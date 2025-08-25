import tensorflow as tf
import numpy as np
from tensorflow.python.framework.errors_impl import NotFoundError
import sys
import concurrent.futures
from time import perf_counter

executor = concurrent.futures.ThreadPoolExecutor(max_workers = 2)
future_blink = None
future_yawn = None

# timing state
future_blink_start = None
future_yawn_start  = None
last_blink_ms = None
last_yawn_ms  = None

def crop_roi(frame, x1, y1, x2, y2):
    h, w = frame.shape[:2]
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()

#sys.path = [p for p in sys.path if '/u/ADAS_project/opencv-3.4.0-install/lib/python3.10/site-packages' not in p]
# === Head Pose Estimation (O-Net / MTCNN landmarks only) ===
WINDOW_NAME = "ONNX MTCNN with Landmarks"
PITCH_UP_THRESHOLD = 0      # degrees (looking down too far)
PITCH_DOWN_THRESHOLD = -160.0   # degrees (looking up too far)

# ---- CONFIG ----
MIRRORED_VIEW = False   # set True if you did cv2.flip(frame, 1) anywhere

# 5-point 3D model (nose forward = +Z). Units arbitrary but consistent.
# Order: [nose, left_eye, right_eye, mouth_left, mouth_right]
_MODEL_POINTS_5 = np.array([
    (  0.0,   0.0,  30.0),   # Nose tip (slightly in front)
    (-35.0,   0.0,   0.0),   # Left eye center
    ( 35.0,   0.0,   0.0),   # Right eye center
    (-25.0, -35.0,   0.0),   # Left mouth corner
    ( 25.0, -35.0,   0.0),   # Right mouth corner
], dtype="double")

# (Optional) light smoothing to stabilize angles
def ema(prev, curr, alpha=0.25):
    return curr if prev is None else (alpha * curr + (1 - alpha) * prev)

_prev_yaw = _prev_pitch = _prev_roll = None

def _draw_axes(img, camera_matrix, dist_coeffs, rvec, tvec, axis_len=80.0):
    axis_3d = np.float32([(0,0,0),(axis_len,0,0),(0,axis_len,0),(0,0,axis_len)])
    axis_2d, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)
    o = tuple(axis_2d[0].ravel().astype(int))
    x = tuple(axis_2d[1].ravel().astype(int))
    y = tuple(axis_2d[2].ravel().astype(int))
    z = tuple(axis_2d[3].ravel().astype(int))
    cv2.line(img, o, x, (0,0,255), 2)
    cv2.line(img, o, y, (0,255,0), 2)
    cv2.line(img, o, z, (255,0,0), 2)

def _overlay_text(frame, yaw, pitch, roll):
    text = f"Yaw:{yaw:6.1f}  Pitch:{pitch:6.1f}  Roll:{roll:6.1f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame, (8, 8), (16 + tw, 16 + th), (0, 0, 0), -1)
    cv2.putText(frame, text, (12, 12 + th),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

import cv2
import dlib
import onnxruntime as ort
import time
import math
from scipy.spatial import distance
import pygame  # ★NEW



# Load ONNX sessions
pnet_sess = ort.InferenceSession("./pnet.onnx")
rnet_sess = ort.InferenceSession("./rnet.onnx")
onet_sess = ort.InferenceSession("./onet.onnx")


model = tf.keras.models.load_model("./blink_best_pnet_05.h5")

yawn_model = tf.keras.models.load_model("./yawn_model_best_parameter.h5")

blink_count = 0

blink_close_frames = 0  # 連続で目を閉じているフレーム数

# ==== Alert sound (non-blocking) ====
pygame.mixer.init()
alert_sound = pygame.mixer.Sound('Alert/alert.mp3')  # カレント/Alert/alert.mp3
ALERT_COOLDOWN = 0.0  # 秒。必要に応じて調整
_last_alert_time = 0.0

def parse_landmarks(landmarks):
    """
    Parses facial landmarks from different input formats (dict or np.ndarray) into a standardized format.
    
    The landmarks can be provided as a dictionary or an ndarray. If a dictionary is used, it should contain
    a 'keypoints' field. If an ndarray is used, it should contain either 10 or 16 values depending on the 
    number of keypoints and format.

    Args:
        landmarks (dict or np.ndarray): Facial landmarks, either as a dictionary with key 'keypoints' or 
                                        as a numpy array of shape (10,) or (16,).
    
    Returns:
        dict: A dictionary containing the facial landmarks with keys: 'nose', 'mouth_right', 'right_eye',
              'left_eye', 'mouth_left'. Each key corresponds to the (x, y) coordinates of that keypoint.
    """
    if isinstance(landmarks, dict):
        if 'keypoints' in landmarks:
            landmarks = landmarks['keypoints']  # Extract 'keypoints' from dict

    if isinstance(landmarks, np.ndarray):
        offset = 0 if landmarks.shape[0] == 10 else 6  # Handle different landmark formats
        landmarks = landmarks.round().astype(int)  # Round coordinates and convert to integers
        landmarks = {
            "nose": [landmarks[offset+2], landmarks[offset+7]],
            "mouth_right": [landmarks[offset+4], landmarks[offset+9]],
            "right_eye": [landmarks[offset+1], landmarks[offset+6]],
            "left_eye": [landmarks[offset+0], landmarks[offset+5]],
            "mouth_left": [landmarks[offset+3], landmarks[offset+8]]
        }

    return landmarks
    
def parse_bbox(bbox, output_as_width_height=True, input_as_width_height=True):
    """
    Parses a bounding box from different formats (dict, list, or ndarray) into a standardized format.
    
    Args:
        bbox (dict, list, np.ndarray): Bounding box in one of the following formats:
                                       - dict with key 'box': [x1, y1, x2, y2]
                                       - list: [x1, y1, x2, y2] or [x1, y1, width, height]
                                       - np.ndarray: Shape (4,) or (5,) where the first value might be an index.
        output_as_width_height (bool): Whether to return the bounding box as [x1, y1, width, height] (default True) or [x1, y1, x2, y2] if False.
        input_as_width_height (bool): Whether the input format of the bounding box is [x1, y1, width, height] (default True) or 
                                      [x1, y1, x2, y2] if False.
                                
    
    Returns:
        np.ndarray: Parsed bounding box in format [x1, y1, width, height] or [x1, y1, x2, y2].
    """
    # Extract box if input is a dict
    if isinstance(bbox, dict):
        bbox = bbox['box']

    # Parse list format
    if isinstance(bbox, list):
        x1, y1, width, height = bbox

        if not input_as_width_height:
            width = width - x1
            height = height - y1

        x2_or_w = width if output_as_width_height else x1 + width
        y2_or_h = height if output_as_width_height else y1 + height

        return np.asarray([x1, y1, x2_or_w, y2_or_h]).round().astype(int)

    # Parse ndarray format
    if isinstance(bbox, np.ndarray):
        offset = 1 if bbox.shape[0] > 4 else 0  # Handle optional first element

        x1, y1, width, height = bbox[offset:offset+4]

        if not input_as_width_height:
            width = width - x1
            height = height - y1

        x2_or_w = width if output_as_width_height else x1 + width
        y2_or_h = height if output_as_width_height else y1 + height

        return np.asarray([x1, y1, x2_or_w, y2_or_h]).round().astype(int)

    raise ValueError("Invalid bbox format. Expected dict, list, or ndarray.")
    
def to_json(bboxes_batch, images_count, input_as_width_height=False, output_as_width_height=True):
    """
    Converts a batch of bounding boxes and facial keypoints into a JSON-friendly format.
    
    This function processes the bounding boxes grouped by unique image IDs, and formats each bounding box
    and its associated keypoints (facial landmarks) into a dictionary structure suitable for JSON serialization.
    
    Args:
        bboxes_batch (np.ndarray): An array of shape (n, 16) where each row represents a bounding box 
                                   and associated keypoints in the following format:
                                   [image_id, x1, y1, x2, y2, confidence, left_eye_x, left_eye_y, right_eye_x, 
                                   right_eye_y, nose_x, nose_y, mouth_left_x, mouth_left_y, mouth_right_x, mouth_right_y].
        images_count (int): Number of different images composed by the batch.
        input_as_width_height (bool, optional): True if format of input bounding boxes is [x1, x2, width, height].
                                                 False if format is [x1, y1, x2, y2].
        output_as_width_height (bool, optional): True to format bounding boxes as [x1, x2, width, height].
                                                 False to format as [x1, y1, x2, y2].
        
    Returns:
        list: A list of lists, where each inner list contains dictionaries for bounding boxes and keypoints 
              for a specific image. Each dictionary has the following structure:
              {
                "box": [x, y, width, height],
                "keypoints": {
                    "nose": [nose_x, nose_y],
                    "mouth_right": [mouth_right_x, mouth_right_y],
                    "right_eye": [right_eye_x, right_eye_y],
                    "left_eye": [left_eye_x, left_eye_y],
                    "mouth_left": [mouth_left_x, mouth_left_y]
                },
                "confidence": confidence_score
              }
    """
    single_element = len(bboxes_batch.shape) == 1

    if single_element:
        bboxes_batch = np.expand_dims(bboxes_batch, axis=0)

    #unique_ids = np.unique(bboxes_batch[:, 0])

    result_batch = []

    # Loop over each unique image ID
    for unique_id in range(images_count):
        result = []
        bboxes_subset = bboxes_batch[bboxes_batch[:, 0] == unique_id]

        # Loop over each bounding box in the subset
        for bbox in bboxes_subset:
            row = {
                "box": parse_bbox(bbox, 
                                  output_as_width_height=output_as_width_height,
                                  input_as_width_height=input_as_width_height).tolist(),
                "confidence": bbox[5]
            }
            result.append(row)

            # If the stages combination allows landmarks, then we append them. Otherwise we don't
            try:
                row["keypoints"] = parse_landmarks(bbox)
            except IndexError:
                pass

        result_batch.append(result)

    return result_batch
    
# --- Load and preprocess input image ---
def preprocess(image):
    #img = cv2.imread(image_path)
    img = cv2.resize(image, (30, 30))  # Ensure correct size
    #img = img.astype('float32') / 255.0  # Normalize to [0,1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension -> (1, 30, 30, 3)
    return img


CNN_CONSEC_FRAMES = 2

# --- Alert function ---
# def check_alert(frame, fps):
#     global blink_close_frames
#     # 2秒以上閉じていたらアラート
#     print(f"blink_close_frames{blink_close_frames}")
#     print(f"FPS{fps}")
#     if blink_close_frames >= 10:
#         if not pygame.mixer.get_busy():  # 連続再生を防止
#             alert_sound.play()
#         cv2.putText(frame, "ALERT!", (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
#                     1.0, (0, 0, 255), 3)

# Initialize variables
eye_closed_start_time = None
alert_played = False

def check_eye_closed_duration(eye_closed):
    global eye_closed_start_time, alert_played
    if eye_closed:
        if eye_closed_start_time is None:
            eye_closed_start_time = time.time()
        else:
            elapsed_time = time.time() - eye_closed_start_time
            text = f"Eyes closed: {elapsed_time:.2f} sec"
            cv2.putText(frame, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            if elapsed_time > 2 and not alert_played:
                alert_sound.play()
                cv2.putText(frame, "ALERT!", (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 255), 3)
                alert_played = True
    else:
        eye_closed_start_time = None
        alert_played = False

    
# --- Predict function ---
def predict_blink(left_eye, right_eye, frame, fps=30):
    global blink_count, blink_detected, blink_close_frames, total_blinks

    left_tensor = preprocess(left_eye)
    right_tensor = preprocess(right_eye)
    left_prediction = model.predict(left_tensor)
    right_prediction = model.predict(right_tensor)

    # 閉じている判定
    closed = (left_prediction[0][0] > left_prediction[0][1] and 
              right_prediction[0][0] > right_prediction[0][1])

    if closed:
        blink_close_frames += 1
        if not blink_detected:
            blink_count += 1
            blink_detected = True
            # cv2.putText(frame, "Blink Detected", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 
            #             0.7, (0, 0, 255), 2)
    else:
        blink_close_frames = 0
        blink_detected = False

    # アラート判定を別関数に委譲
    # check_alert(frame, fps)
    check_eye_closed_duration(closed)

    cv2.putText(frame, f"Total Blinks = {blink_count}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)


def predict_yawn(mouth, frame):
    mouth_tensor = preprocess(mouth)
    mouth_prediction = yawn_model.predict(mouth_tensor)
    global yawn_count, yawn_detected
    print(mouth_prediction[0][0], mouth_prediction[0][1])
    if (mouth_prediction[0][0] < mouth_prediction[0][1]):
        if not yawn_detected:
            yawn_count += 1
            yawn_detected = True
            # cv2.putText(frame, "yawn Detected", (60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        yawn_detected = False

    cv2.putText(frame, f"Total Yawns = {yawn_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
           
         
"""
pnet_model = tf.keras.models.load_model("/u/ADAS_project/pnet_tf")
rnet_model = tf.keras.models.load_model("/u/ADAS_project/rnet_tf")
onet_model = tf.keras.models.load_model("/u/ADAS_project/onet_tf")
"""

# Load TFLite model
#interpreter = tf.lite.Interpreter(model_path="/u/ADAS_project/mediapipe_face-facelandmarkdetector-float.tflite")
#predictor = dlib.shape_predictor('/u/ADAS_project/shape_predictor_68_face_landmarks.dat')
#interpreter.allocate_tensors()

# Input and output
#input_details = interpreter.get_input_details()
#output_details = interpreter.get_output_details()


# Define landmark indices
left_eye_indices = [33, 133, 160, 159, 158, 144, 153, 154, 155, 173]
right_eye_indices = [362, 263, 387, 386, 385, 373, 380, 381, 382, 362]
outer_lip_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
inner_lip_indices = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

# Combine all indices
selected_indices = left_eye_indices + right_eye_indices + outer_lip_indices + inner_lip_indices


# Landmark indices for lip seam
upper_lip_center = 13
lower_lip_center = 14
mouth_left_corner = 78
mouth_right_corner = 308

# Indices to use
lip_seam_indices = [upper_lip_center, lower_lip_center, mouth_left_corner, mouth_right_corner]

def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def image_cropping(x, y, width, height, frame):
    # Ensure coordinates are integers
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + width), int(y + height)

    # Optional: Check bounds to avoid indexing errors
    h, w, _ = frame.shape
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    # Crop the region from the frame
    cropped_face = frame[y1:y2, x1:x2]

    return cropped_face

def resize(image, size):
    #resize
    resized_face = cv2.resize(image, (192, 192))
    return resized_face


def load_image(image, dtype=tf.float32, device="CPU:0"):

    with tf.device(device):
        is_tensor = tf.is_tensor(image) or isinstance(image, np.ndarray)

        if is_tensor:
            decoded_image = image
        else:
            try:
                if isinstance(image, str):
                    image_data = tf.io.read_file(image)  # Read image from file
                else:
                    image_data = image  # Assume image data is provided directly
            except NotFoundError:
                image_data = image  # If file not found, use the input directly

            # Decode the image with 3 channels (RGB)
            decoded_image = tf.image.decode_image(image_data, channels=3, dtype=dtype).numpy()

        # If dtype is float, adjust the image scale
        if dtype in [tf.float16, tf.float32]:
            decoded_image *= 255  # Convert pixel values to [0, 255] if using float data type

    return decoded_image


def load_images_batch(images, dtype=tf.float32, device="CPU:0"):
  
    is_tensor = tf.is_tensor(images[0]) or isinstance(images[0], np.ndarray)
    images_raw = images if is_tensor else [load_image(img, dtype=dtype, device=device) for img in images]
    return images_raw
    
def pad_stack_np(images, justification="center"):

    # Stack the shapes of all images into an array
    sizes_stack = np.stack([img.shape for img in images], axis=0)

    # Find the maximum shape along each dimension
    sizes_max = sizes_stack.max(axis=0, keepdims=True)

    # Calculate the difference in size for padding
    sizes_diff = sizes_max - sizes_stack

    # Calculate if any padding size is odd, to adjust padding
    sizes_mod = sizes_diff % 2
    sizes_diff = sizes_diff - sizes_mod

    # Justification masks for padding alignment
    justification_mask = {
        "top": np.asarray([[[0, 1], [0.5, 0.5], [0, 0]]]),
        "topleft": np.asarray([[[0, 1], [0, 1], [0, 0]]]),
        "topright": np.asarray([[[0, 1], [1, 0], [0, 0]]]),
        "bottom": np.asarray([[[1, 0], [0.5, 0.5], [0, 0]]]),
        "bottomleft": np.asarray([[[1, 0], [0, 1], [0, 0]]]),
        "bottomright": np.asarray([[[1, 0], [1, 0], [0, 0]]]),
        "left": np.asarray([[[0.5, 0.5], [0, 1], [0, 0]]]),
        "right": np.asarray([[[0.5, 0.5], [1, 0], [0, 0]]]),
        "center": np.asarray([[[0.5, 0.5], [0.5, 0.5], [0, 0]]]),
    }

    # Justification adjustments for padding if needed
    justification_pad_mask = {
        "top": "topleft",
        "bottom": "bottomleft",
        "left": "topleft",
        "right": "topright",
        "center": "topleft"
    }

    # Get the correct padding mask based on justification
    pad_mask = justification_mask[justification]
    mod_mask = justification_mask[justification_pad_mask.get(justification, justification)]

    # Calculate the exact padding parameters
    pad_param = (pad_mask * sizes_diff[:,:,None] + mod_mask * sizes_mod[:,:,None]).astype(int)

    # Apply the calculated padding to each image and stack them into a single array
    images_padded = np.stack([np.pad(img, pad) for img, pad in zip(images, pad_param)], axis=0)

    # We keep the original faces to return as extra info
    original_shapes = np.stack([img.shape for img in images], axis=0)

    return images_padded, original_shapes, pad_param
    
def normalize_images(images):
    
    # Normalize the images to the range (-1, 1)
    return (images - 127.5) / 128
    
def standarize_batch(images_raw, normalize=True, justification="center"):
    
    images_result, images_oshapes, pad_param = pad_stack_np(images_raw, justification=justification)

    if normalize:
        images_result = normalize_images(images_result)

    return images_result, images_oshapes, pad_param
    
    
    
def build_scale_pyramid(width, height, min_face_size, scale_factor, min_size=12):
   

    # Find the smallest dimension of the image
    min_dim = min(width, height)

    # Calculate how many scales are needed based on the smallest dimension and the scale factor
    scales_count = round(-((np.log(min_dim / min_size) / np.log(scale_factor)) + 1))

    # Calculate the base scale value (based on the smallest detectable face size)
    m = min_size / min_face_size

    # Generate an array of scales for the pyramid
    return m * scale_factor ** np.arange(scales_count)    
    


def scale_images(images, scale: float=None, new_shape: tuple=None):

    # Extract the shape from the images
    shape = np.asarray(images.shape[-3:-1])

    if scale is None and new_shape is None:
        new_shape = shape

    new_shape = shape * scale if new_shape is None else new_shape

    # Resize the images using the specified scaling factor
    images_scaled = tf.image.resize(images, new_shape, method=tf.image.ResizeMethod.AREA)

    return images_scaled
    
    
    
def apply_scales(images_normalized, scales_groups):
    
    # Select the scale group with the largest number of elements
    selected_scaleset_as_index = np.argmax([x.shape[0] for x in scales_groups])
    largest_scale_group_set = scales_groups[selected_scaleset_as_index]

    # Apply the scales from the largest scale group to the normalized images
    result = [scale_images(images_normalized, scale) for scale in largest_scale_group_set]

    return result, largest_scale_group_set
    

def sort_by_scores(tensor, scores, ascending=True):

    # Get the sorted indices based on the scores
    sorted_indices = np.argsort(scores)

    # Sort the tensor using the sorted indices, reversing if descending
    sorted_tensor = tensor[sorted_indices[::(-2 * int(not ascending) + 1)]]

    return sorted_tensor
    

def upscale_bboxes(bboxes_result, scales):
    if len(bboxes_result) == 0:
        return bboxes_result

    scale_indices = bboxes_result[:, 0].astype(int)

    # Debug: print to see what's wrong
    #print("scale_indices:", scale_indices)
    #print("scales shape:", scales.shape)

    # Fix: clip indices to be within bounds
    scale_indices = np.clip(scale_indices, 0, len(scales)-1)

    scales_bcast = np.expand_dims(scales[scale_indices], axis=-1)
    bboxes_result[:, 1:5] /= scales_bcast
    return bboxes_result

            

def generate_bounding_box(bbox_reg, bbox_class, threshold_face, strides=2, cell_size=12):
  
    #bbox_reg = bbox_reg.numpy()
    #bbox_class = bbox_class.numpy()

    # Create a mask for detected faces based on the threshold for face probability
    confidence_score = bbox_class[:,:,:,1]

    # Find the indices where the detection mask is true (i.e., face detected)
    index_bboxes = np.stack(np.where(confidence_score > threshold_face)) # batch_size, y, x
    filtered_bbox_reg = np.transpose(bbox_reg[index_bboxes[0], index_bboxes[1], index_bboxes[2]], (1,0))

    # Extract the regression values
    reg_x1, reg_y1, reg_x2, reg_y2 = filtered_bbox_reg

    # Convert strides and cell size into arrays for easy broadcasting
    strides = np.asarray([[1], [strides], [strides]])
    cellsize = [np.asarray([[0], [1], [1]]), np.asarray([[0], [cell_size], [cell_size]])]

    # Calculate the top-left and bottom-right corners of the bounding boxes
    bbox_up_left = index_bboxes * strides + cellsize[0]
    bbox_bottom_right = index_bboxes * strides + cellsize[1]

    # Calculate width and height for the bounding boxes
    reg_w = bbox_bottom_right[2] - bbox_up_left[2]  # width of bounding box
    reg_h = bbox_bottom_right[1] - bbox_up_left[1]  # height of bounding box

    # Apply the regression to adjust the bounding box coordinates
    x1 = bbox_up_left[2] + reg_x1 * reg_w  # Adjusted x1
    y1 = bbox_up_left[1] + reg_y1 * reg_h  # Adjusted y1
    x2 = bbox_bottom_right[2] + reg_x2 * reg_w  # Adjusted x2
    y2 = bbox_bottom_right[1] + reg_y2 * reg_h  # Adjusted y2

    # Concatenate the bounding box coordinates and detection information, keeping batch index
    bboxes_result = np.stack([
        index_bboxes[0], x1, y1, x2, y2, confidence_score[index_bboxes[0], index_bboxes[1], index_bboxes[2]]
    ], axis=0).T

    # Sort bounding boxes by score in descending order
    bboxes_result = sort_by_scores(bboxes_result, scores=bboxes_result[:, -1], ascending=False)

    return bboxes_result

    
    

def iou(bboxes, method="union"):

    # Convert the list of bounding boxes to a NumPy array
    bboxes = np.stack(bboxes, axis=0)

    # Calculate the area of each bounding box
    area_bboxes = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

    # Expand dimensions to compute pairwise IoU (N x N matrix)
    bboxes_a = np.expand_dims(bboxes, axis=0)
    bboxes_b = np.expand_dims(bboxes, axis=1)

    # Calculate the intersection coordinates
    row_inter_top = np.maximum(bboxes_a[:, :, 0], bboxes_b[:, :, 0])
    col_inter_left = np.maximum(bboxes_a[:, :, 1], bboxes_b[:, :, 1])
    row_inter_bottom = np.minimum(bboxes_a[:, :, 2], bboxes_b[:, :, 2])
    col_inter_right = np.minimum(bboxes_a[:, :, 3], bboxes_b[:, :, 3])

    # Calculate the intersection area
    height_inter = np.maximum(0, row_inter_bottom - row_inter_top)
    width_inter = np.maximum(0, col_inter_right - col_inter_left)
    area_inter = height_inter * width_inter

    # Compute IoU based on the specified method
    if method == "union":
        # Union: Area of A + Area of B - Intersection
        area_union = area_bboxes[:, None] + area_bboxes[None, :] - area_inter
        iou_matrix = area_inter / area_union
    elif method == "min":
        # Minimum: Area of the smaller box between A and B
        area_min = np.minimum(area_bboxes[:, None], area_bboxes[None, :])
        iou_matrix = area_inter / area_min
    else:
        raise ValueError("Method should be either 'union' or 'min'.")

    return iou_matrix
    

def nms(target_iou, threshold):

    # Step 1: Create a mask for allowed comparisons (upper triangular part of the IoU matrix, excluding the diagonal)
    allowed_mask = np.triu(np.ones((target_iou.shape[0], target_iou.shape[0])), k=1)

    # Step 2: Create a mask for failed comparisons (IoU above the threshold)
    failed_mask = (target_iou > threshold).astype(int)

    # Step 3: Combine the masks and get the indices of the remaining boxes
    result_indexes = np.where((failed_mask * allowed_mask).sum(axis=0) == 0)[0]

    return result_indexes
    

def smart_nms_from_bboxes(bboxes, threshold, column_image_id=0, columns_bbox=slice(1, 5, None), column_confidence=5,
                          method="union", initial_sort=True):
                          
    # Step 0: Sort if required
    if initial_sort:
        bboxes = sort_by_scores(bboxes, scores=bboxes[:, column_confidence], ascending=False)

    # Step 1: Get unique image IDs
    image_ids = np.unique(bboxes[:, 0])

    result = []

    # Step 2: Apply NMS per image
    for image_id in image_ids:
        # Filter bounding boxes for the current image
        target_bboxes = bboxes[bboxes[:, column_image_id] == image_id]

        # Compute the IoU matrix for the bounding boxes
        target_iou = iou(target_bboxes[:, columns_bbox], method=method)

        # Perform NMS and get the indices of the boxes to keep
        target_indexes = nms(target_iou, threshold)

        # Filter the boxes for the image
        target_filtered_bboxes = target_bboxes[target_indexes.astype(int)]

        # Store the result
        result.append(target_filtered_bboxes)

    result = np.concatenate(result, axis=0) if len(result) > 0 else np.empty((0, 6))

    return result
  
  


def resize_to_square(bboxes):
    
    bboxes = bboxes.copy()
    h = bboxes[:, 4] - bboxes[:, 2]  # Height of each bounding box
    w = bboxes[:, 3] - bboxes[:, 1]  # Width of each bounding box
    largest_size = np.maximum(w, h)  # Largest dimension (width or height)

    # Adjust x1 and y1 to center the bounding box and resize to square
    bboxes[:, 1] = bboxes[:, 1] + w * 0.5 - largest_size * 0.5
    bboxes[:, 2] = bboxes[:, 2] + h * 0.5 - largest_size * 0.5
    bboxes[:, 3:5] = bboxes[:, 1:3] + np.tile(largest_size, (2, 1)).T  # Resize x2, y2

    return bboxes

def extract_patches(images_normalized, bboxes_batch, expected_size=(24, 24)):

    # Get the shape of the input images
    shape = images_normalized.shape

    # Normalize the bounding box coordinates to be within [0, 1] relative to image dimensions
    selector = [2, 1, 4, 3]

    bboxes_batch_coords = bboxes_batch[:, selector] / np.asarray([[shape[selector[1]], shape[selector[0]], shape[selector[1]], shape[selector[0]]]])

    # Extract patches from the images using the bounding boxes, resizing them to `expected_size`
    result = tf.image.crop_and_resize(
        images_normalized,                 # Input image tensor
        bboxes_batch_coords,               # Bounding boxes in format [y1, x1, y2, x2], normalized to [0.0, 1.0]
        bboxes_batch[:, 0].astype(int),    # Indices of the images in the batch corresponding to the bounding boxes
        expected_size                      # Size to resize the cropped patches (height, width)
    )

    return result


def replace_confidence(bboxes_batch, new_scores):
    
    bboxes_batch[:, -1] = new_scores[:, -1]
    return bboxes_batch
    

def adjust_bboxes(bboxes_batch, bboxes_offsets):
    
    bboxes_batch = bboxes_batch.copy()
    w = bboxes_batch[:, 3] - bboxes_batch[:, 1] + 1  # Calculate width of each bounding box
    h = bboxes_batch[:, 4] - bboxes_batch[:, 2] + 1  # Calculate height of each bounding box

    sizes = np.stack([w, h, w, h], axis=-1)  # Stack width and height to match bbox_offsets
    bboxes_batch[:, 1:5] += bboxes_offsets * sizes  # Apply offsets to the coordinates

    return bboxes_batch


def pick_matches(bboxes_batch, scores_column=-1, score_threshold=0.7):
    
    return bboxes_batch[np.where(bboxes_batch[:, scores_column] > score_threshold)[0]]
 

def adjust_landmarks(face_landmarks, bboxes_batch):
    
    # Convert face_landmarks to a NumPy array and make a copy
    face_landmarks = face_landmarks.copy()

    # Compute the width and height of each bounding box
    w = bboxes_batch[:, 3:4] - bboxes_batch[:, 1:2] + 1  # Width
    h = bboxes_batch[:, 4:5] - bboxes_batch[:, 2:3] + 1  # Height

    # Adjust the x-coordinates of the landmarks
    face_landmarks[:, 0:5] = w * face_landmarks[:, 0:5] + bboxes_batch[:, 1:2] - 1
    # Adjust the y-coordinates of the landmarks
    face_landmarks[:, 5:10] = h * face_landmarks[:, 5:10] + bboxes_batch[:, 2:3] - 1

    return face_landmarks


def fix_bboxes_offsets(bboxes_batch, pad_param):
    
    bboxes_batch = bboxes_batch.copy()
    images_ids = np.unique(bboxes_batch[:, 0])  # Get unique image IDs

    indexes_bbox_x = [1,3]
    indexes_bbox_y = [2,4]

    indexes_landmarks_x = [6, 7, 8, 9, 10]
    indexes_landmarks_y = [11, 12, 13, 14, 15]


    # Adjust bounding boxes and landmarks for each image based on its padding parameters
    for image_id, pad in zip(images_ids, pad_param):
        selector = bboxes_batch[:, 0] == image_id

        # Adjust the x-coordinates of bounding boxes by subtracting width padding
        bboxes_batch[np.ix_(selector, indexes_bbox_x)] -= pad[1, 0]

        # Adjust the y-coordinates of bounding boxes by subtracting height padding
        bboxes_batch[np.ix_(selector, indexes_bbox_y)] -= pad[0, 0]

        # If stages combinations contain landmarks, we adjust them too
        try:
            # Adjust the x-coordinates of landmarks by subtracting width padding
            bboxes_batch[np.ix_(selector, indexes_landmarks_x)] -= pad[1, 0]

            # Adjust the y-coordinates of landmarks by subtracting height padding
            bboxes_batch[np.ix_(selector, indexes_landmarks_y)] -= pad[0, 0]

        except IndexError:
            pass


    return bboxes_batch

def limit_bboxes(bboxes_batch, images_shapes, limit_landmarks=True):
    
    bboxes_batch_fitted = bboxes_batch.copy()

    # Get the original shapes (height, width) for each image in the batch
    expected_shapes = images_shapes[bboxes_batch_fitted[:, 0].astype(int)]

    # Adjust x1 and x2 to be within [0, width-1]
    bboxes_batch_fitted[:, 1] = np.minimum(np.maximum(bboxes_batch_fitted[:, 1], 0), expected_shapes[:, 1] - 1)
    bboxes_batch_fitted[:, 3] = np.minimum(np.maximum(bboxes_batch_fitted[:, 3], 0), expected_shapes[:, 1] - 1)

    # Adjust y1 and y2 to be within [0, height-1]
    bboxes_batch_fitted[:, 2] = np.minimum(np.maximum(bboxes_batch_fitted[:, 2], 0), expected_shapes[:, 0] - 1)
    bboxes_batch_fitted[:, 4] = np.minimum(np.maximum(bboxes_batch_fitted[:, 4], 0), expected_shapes[:, 0] - 1)

    if limit_landmarks:
        # Adjust x1..x5 of the landmarks to not surpass boundaries
        bboxes_batch_fitted[:, 6:11] = np.minimum(np.maximum(bboxes_batch_fitted[:, 6:11], 0), expected_shapes[:, 1:2] - 1)

        # Adjust y1..y5 of the landmarks to not surpass boundaries
        bboxes_batch_fitted[:, 11:16] = np.minimum(np.maximum(bboxes_batch_fitted[:, 11:16], 0), expected_shapes[:, 0:1] - 1)

    return bboxes_batch_fitted
                        
# ================= PNet =====================
# def run_pnet(img, threshold=0.6, min_face_size=70, scale_factor=0.5): #p-netの縮小倍率 0.709 -> 0.5
#     is_batch = isinstance(img, list)
#     img = img if is_batch else [img]
#     images_raw = load_images_batch(img)
#     images_normalized, images_oshapes, pad_param = standarize_batch(images_raw, justification="center", normalize=True)
    
#     #bboxes_batch = None
    
#     scales_groups = [build_scale_pyramid(shape[1], shape[0], min_face_size=min_face_size, scale_factor=scale_factor) for shape in images_oshapes]
    
#     # 2. Apply the scales to normalized images
#     scales_result, scales_index = apply_scales(images_normalized, scales_groups)
#     batch_size = images_normalized.shape[0]
    
#     # 3. Get proposals bounding boxes and confidence from the model (PNet)
    
#     #pnet_result = [pnet_model(s) for s in scales_result] 
#     #pnet_result = [pnet_sess.run(None, {"input": input_tensor.astype(np.float32)}) for input_tensor in scales_result]
#     pnet_result = [pnet_sess.run(None, {"input": tf.cast(input_tensor, tf.float32).numpy()}) for input_tensor in scales_result]
    
#     #print(pnet_result)
#     # 4. Generate bounding boxes per scale group
#     bboxes_proposals = [generate_bounding_box(result[0], result[1], 0.6) for result in pnet_result]
#     bboxes_batch_upscaled = [upscale_bboxes(bbox, np.asarray([scale] * batch_size)) for bbox, scale in zip(bboxes_proposals, scales_index)]

#     # 5. Apply Non-Maximum Suppression (NMS) per scale group
#     bboxes_nms = [smart_nms_from_bboxes(b, threshold=0.5, method="union", initial_sort=False) for b in bboxes_batch_upscaled]

#     # 6. Concatenate and apply NMS again across all scales
#     bboxes_batch = np.concatenate(bboxes_nms, axis=0) if len(bboxes_nms) > 0 else np.empty((0, 6))
#     bboxes_batch = smart_nms_from_bboxes(bboxes_batch, threshold=0.7, method="union", initial_sort=True)

#     # 7. Resize bounding boxes to square format
#     bboxes_batch = resize_to_square(bboxes_batch)
    
#     #print(bboxes_batch)
#     return bboxes_batch, images_normalized, images_oshapes, pad_param, img

def run_pnet(
    img,
    threshold=0.6,
    min_face_size=70,
    scale_factor=0.5,
    min_box_size=300,   # <--- New parameter: discard faces smaller than this (pixels)
    max_box_size=500
):
    is_batch = isinstance(img, list)
    img = img if is_batch else [img]

    images_raw = load_images_batch(img)
    images_normalized, images_oshapes, pad_param = standarize_batch(
        images_raw, justification="center", normalize=True
    )

    # 1. Build pyramids
    scales_groups = [
        build_scale_pyramid(shape[1], shape[0],
                            min_face_size=min_face_size,
                            scale_factor=scale_factor)
        for shape in images_oshapes
    ]

    # 2. Apply scales
    scales_result, scales_index = apply_scales(images_normalized, scales_groups)
    batch_size = images_normalized.shape[0]

    # 3. Run P-Net
    pnet_result = [
        pnet_sess.run(None, {"input": tf.cast(input_tensor, tf.float32).numpy()})
        for input_tensor in scales_result
    ]

    # 4. Generate bounding boxes
    bboxes_proposals = [
        generate_bounding_box(result[0], result[1], threshold)   # use threshold arg
        for result in pnet_result
    ]
    bboxes_batch_upscaled = [
        upscale_bboxes(bbox, np.asarray([scale] * batch_size))
        for bbox, scale in zip(bboxes_proposals, scales_index)
    ]

    # 5. Per-scale NMS
    bboxes_nms = [
        smart_nms_from_bboxes(b, threshold=0.5, method="union", initial_sort=False)
        for b in bboxes_batch_upscaled
    ]

    # 6. Merge and global NMS
    bboxes_batch = np.concatenate(bboxes_nms, axis=0) if len(bboxes_nms) > 0 else np.empty((0, 6))
    bboxes_batch = smart_nms_from_bboxes(bboxes_batch, threshold=0.7, method="union", initial_sort=True)

    # 7. Resize to square
    bboxes_batch = resize_to_square(bboxes_batch)

    # 8. Exclude too-small, too-large boxes
    if bboxes_batch.shape[0] > 0:
        w = bboxes_batch[:, 2] - bboxes_batch[:, 0]
        h = bboxes_batch[:, 3] - bboxes_batch[:, 1]
        keep_idx = np.where((w >= min_box_size) & (h >= min_box_size) & (w <= max_box_size) & (h <= max_box_size))[0]
        bboxes_batch = bboxes_batch[keep_idx]

    return bboxes_batch, images_normalized, images_oshapes, pad_param, img



def run_rnet(images_normalized, bboxes_batch, threshold_rnet=0.7, nms_rnet=0.7, **kwargs):
    # 1. Extract patches for each bounding box from the normalized images.
    # These patches are resized to the expected input size for RNet (24x24).
    patches = extract_patches(images_normalized, bboxes_batch, expected_size=(24, 24))

    # 2. Pass the extracted patches through RNet to get bounding box offsets and confidence scores.
    
    #bboxes_offsets, scores = rnet_model(patches)
    bboxes_offsets, scores = rnet_sess.run(None, {"input": tf.cast(patches, tf.float32).numpy()}) 
    
    # 3. Replace the confidence of the bounding boxes with the ones provided by RNet.
    bboxes_batch = replace_confidence(bboxes_batch, scores)

    # 4. Adjust the bounding boxes using the offsets predicted by RNet (refinement of the proposals).
    bboxes_batch = adjust_bboxes(bboxes_batch, bboxes_offsets)

    # 5. Filter out bounding boxes based on the new confidence scores and the threshold set for RNet.
    bboxes_batch = pick_matches(bboxes_batch, score_threshold=threshold_rnet)

    # 6. Apply Non-Maximum Suppression (NMS) to remove overlapping boxes based on the refined boxes and scores.
    bboxes_batch = smart_nms_from_bboxes(bboxes_batch, threshold=nms_rnet, method="union", initial_sort=True)

    # 7. Resize bounding boxes to a square format to prepare them for the next stage.
    bboxes_batch = resize_to_square(bboxes_batch)

    return bboxes_batch

def run_onet(images_normalized, bboxes_batch, threshold_onet=0.8, nms_onet=0.7, **kwargs):
    # 1. Extract patches for each bounding box from the normalized images.
    # These patches are resized to the expected input size for ONet (48x48).
    patches = extract_patches(images_normalized, bboxes_batch, expected_size=(48, 48))

    # 2. Pass the extracted patches through ONet to get bounding box offsets, facial landmarks, and confidence scores.
    #bboxes_offsets, face_landmarks, scores = onet_model(patches)
    bboxes_offsets, face_landmarks, scores = onet_sess.run(None, {"input": tf.cast(patches, tf.float32).numpy()}) 
    
    
    # 3. Adjust the landmarks to match the bounding box coordinates relative to the original image.
    face_landmarks = adjust_landmarks(face_landmarks, bboxes_batch)

    # 4. Replace the confidence of the bounding boxes with the ones provided by ONet.
    bboxes_batch = replace_confidence(bboxes_batch, scores)

    # 5. Adjust the bounding boxes using the offsets predicted by ONet (refinement of the proposals).
    bboxes_batch = adjust_bboxes(bboxes_batch, bboxes_offsets)

    # 6. Combine the facial landmarks with the bounding boxes batch tensor.
    bboxes_batch = np.concatenate([bboxes_batch, face_landmarks], axis=-1)

    # 7. Filter out bounding boxes based on the new confidence scores and the threshold set for ONet.
    bboxes_batch = pick_matches(bboxes_batch, scores_column=5, score_threshold=threshold_onet)

    # 8. Apply Non-Maximum Suppression (NMS) to remove overlapping boxes based on the refined boxes, scores, and landmarks.
    bboxes_batch = smart_nms_from_bboxes(bboxes_batch, threshold=nms_onet, method="min", initial_sort=True)

    return bboxes_batch

def detect_pnet(img, minsize, threshold):
    return run_pnet(img, threshold[0])

def detect_rnet(boxes, images_normalized):
    return run_rnet(images_normalized, boxes)
    
def detect_onet(boxes, images_normalized):
    return run_onet(images_normalized, boxes)
    
# ====================== Camera Loop ======================
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS) or 30  # fpsを取得。取得できない場合は30で仮定
print(fps)
BLINK_CLOSE_FRAMES_THRESHOLD = int(fps * 2)  # 2秒連続閉眼でアラート

threshold = [0.2, 0.5, 0.7]

# FPS計算用の変数
frame_count = 0
start_time = time.time()
current_fps = 0

cv2.namedWindow("ONNX MTCNN with Landmarks", cv2.WINDOW_NORMAL)  # allows resizing
cv2.resizeWindow("ONNX MTCNN with Landmarks", 650, 530)         # set desired window size


# EAR threshold & counter
EAR_THRESHOLD = 0.3
EAR_CONSEC_FRAMES = 3
#blink_counter = 0

# MAR threshold
MAR_THRESHOLD = 0.2
MAR_CONSEC_FRAMES = 5
#yawn_counter = 0

# Define outside loop
yawn_display_start_time = 0
YAWN_DISPLAY_DURATION = 3  # seconds

# define blink counters and yawn counters 
blink_counter = 0
total_blinks = 0
blink_detected = False

yawn_count = 0
total_yawn = 0
yawn_detected = False

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(lower_lip, joint_lip, right_mouth_vertex, left_mouth_vertex):
    """
    A = distance.euclidean(mouth[3], mouth[9])   # p51–p59
    B = distance.euclidean(mouth[4], mouth[8])   # p52–p58
    C = distance.euclidean(mouth[5], mouth[7])   # p53–p57
    D = distance.euclidean(mouth[2], mouth[10])  # p50–p60 (horizontal)
    mar = (A + B + C) / (2.0 * D)
    return mar
    """
    A = distance.euclidean(lower_lip[1], joint_lip[1])
    B = distance.euclidean(right_mouth_vertex, left_mouth_vertex)
    return (A/B)

def shape_to_np(shape):
    """Convert dlib shape to a (68, 2) numpy array."""
    coords = np.zeros((68, 2), dtype="double")
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def draw_axes(img, camera_matrix, dist_coeffs, rvec, tvec, axis_len=100.0):
    """Draw 3D axes (X: right, Y: down, Z: forward) originating at nose tip."""
    axis_3d = np.float32([
        (0, 0, 0),                 # origin (nose tip)
        (axis_len, 0, 0),          # X axis
        (0, axis_len, 0),          # Y axis
        (0, 0, axis_len)           # Z axis
    ])
    axis_2d, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)
    o = tuple(axis_2d[0].ravel().astype(int))
    x = tuple(axis_2d[1].ravel().astype(int))
    y = tuple(axis_2d[2].ravel().astype(int))
    z = tuple(axis_2d[3].ravel().astype(int))

    cv2.line(img, o, x, (255, 0, 0), 2)   # X-axis
    cv2.line(img, o, y, (0, 255, 0), 2)   # Y-axis
    cv2.line(img, o, z, (0, 0, 255), 2)   # Z-axis

def overlay_text(frame, yaw, pitch, roll):
    text = f"Yaw:{yaw:6.1f}  Pitch:{pitch:6.1f}  Roll:{roll:6.1f}"
    # (Optional) background for readability
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame, (8, 8), (16 + tw, 16 + th), (0, 0, 0), -1)
    cv2.putText(frame, text, (12, 12 + th),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

output_path = 'output.mp4'  # Change to .mp4 and codec if you want
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can also use 'MJPG' or 'mp4v'
fps = 30.0  # Frames per second
frame_size = (471, 628)  # Change this to match your actual frame size
out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

 
    
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # FPS計算
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1.0:  # 1秒ごとにFPSを更新
        current_fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, images_normalized, images_oshape, pad_param, img = detect_pnet(rgb, 120, threshold)
    boxes = detect_rnet(boxes, images_normalized)
    boxes = detect_onet(boxes, images_normalized)
    
    bboxes_batch = fix_bboxes_offsets(boxes, pad_param)
    bboxes_batch = limit_bboxes(boxes, images_shapes=images_oshape, limit_landmarks=True)

    result = to_json(bboxes_batch, images_count=len(img), output_as_width_height="xywh", input_as_width_height=False)
    result = result[0] if (not True and len(result) > 0) else result
    
    print(result)
    if len(result[0]) == 0:
        # FPSを表示
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (frame.shape[1] - 120, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(WINDOW_NAME, frame)
        out.write(frame)
        # Only 'q' to quit — no baseline capture
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
        continue

    x1 = result[0][0]['box'][0]
    y1 = result[0][0]['box'][1]
    width  = result[0][0]['box'][2] + 10
    height = result[0][0]['box'][3] + 20

    cv2.rectangle(frame, (int(x1), int(y1)), (int(x1 + width), int(y1 + height)), (0, 255, 0), 2)

    # --- Head pose estimation using O-Net keypoints (no dlib) ---
    try:
        kps = result[0][0]['keypoints'] if isinstance(result[0][0], dict) and 'keypoints' in result[0][0] else None
        if kps is not None:
            # read keypoints
            le   = kps.get('left_eye')
            reye = kps.get('right_eye')
            nose = kps.get('nose')
            ml   = kps.get('mouth_left')
            mr   = kps.get('mouth_right')

            # If you mirrored the frame earlier, swap L/R to keep *subject-left/right* correct
            if MIRRORED_VIEW:
                le, reye = reye, le
                ml, mr   = mr, ml

            # Fallback nose as mid-eye if missing
            if (nose is None) and (le is not None and reye is not None):
                nose = ((le[0] + reye[0]) * 0.5, (le[1] + reye[1]) * 0.5)

            if all(v is not None for v in [le, reye, nose, ml, mr]):
                # ORDER MUST MATCH _MODEL_POINTS_5
                image_points = np.array([
                    (float(nose[0]), float(nose[1])),   # nose
                    (float(le[0]),   float(le[1])),     # left eye
                    (float(reye[0]), float(reye[1])),   # right eye
                    (float(ml[0]),   float(ml[1])),     # mouth left
                    (float(mr[0]),   float(mr[1])),     # mouth right
                ], dtype="double")

                # Camera intrinsics
                h, w = frame.shape[:2]
                focal_length = float(w)
                center = (w/2.0, h/2.0)
                camera_matrix = np.array([[focal_length, 0, center[0]],
                                        [0, focal_length, center[1]],
                                        [0, 0, 1]], dtype="double")
                dist_coeffs = np.zeros((4,1), dtype="double")

                # Initial pose with EPNP (>=4 pts), then refine with LM
                success, rvec, tvec = cv2.solvePnP(
                    _MODEL_POINTS_5, image_points, camera_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_EPNP
                )
                if success and hasattr(cv2, "solvePnPRefineLM"):
                    rvec, tvec = cv2.solvePnPRefineLM(
                        _MODEL_POINTS_5, image_points, camera_matrix, dist_coeffs, rvec, tvec
                    )

                if success:
                    R, _ = cv2.Rodrigues(rvec)

                    # Cheirality sanity check: ensure the modeled nose is in front of camera
                    # Project the model nose (0,0,30) to camera coords and check Z
                    nose_cam = (R @ _MODEL_POINTS_5[0].reshape(3,1) + tvec).ravel()
                    if nose_cam[2] < 0:
                        # Flip pose if it's behind camera (rare but can fix “backwards” solutions)
                        R[:, :2] = -R[:, :2]  # flip X and Y axes
                        rvec, _ = cv2.Rodrigues(R)

                    # --- Euler extraction (camera x-right, y-down, z-forward) ---
                    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
                    singular = sy < 1e-6
                    if not singular:
                        pitch = np.degrees(np.arctan2(R[2,1], R[2,2]))   # +up/-down
                        yaw   = np.degrees(np.arctan2(-R[2,0], sy))     # +left/-right
                        roll  = np.degrees(np.arctan2(R[1,0], R[0,0]))  # +tilt left
                    else:
                        pitch = np.degrees(np.arctan2(-R[1,2], R[1,1]))
                        yaw   = np.degrees(np.arctan2(-R[2,0], sy))
                        roll  = 0.0

                    # If you show a mirrored preview, invert yaw & roll so the text matches your visual intuition
                    if MIRRORED_VIEW:
                        yaw  = -yaw
                        roll = -roll

                    # Smooth angles a bit for readability
                    yaw   = float(ema(_prev_yaw,   yaw))
                    pitch = float(ema(_prev_pitch, pitch))
                    roll  = float(ema(_prev_roll,  roll))
                    _prev_yaw, _prev_pitch, _prev_roll = yaw, pitch, roll

                    # ----- robust nose direction line with pitch-consistent sign -----
                    TOWARD_CAMERA = True   # keep True if you want the line to come out of the face
                    LINE_LEN = 140.0

                    # 1) Nose (object space) and forward axis (object +Z) in CAMERA space
                    nose_obj = _MODEL_POINTS_5[0].reshape(3, 1)          # (3,1)
                    nose_cam = (R @ nose_obj + tvec)                     # (3,1)
                    fwd_cam  = (R @ np.array([[0.0], [0.0], [1.0]]))     # (3,1)
                    fwd_cam  = fwd_cam / max(1e-9, np.linalg.norm(fwd_cam))   # normalize

                    # 2) Choose a base sign so "toward camera" reduces Z; else away increases Z
                    base_sign = -1.0 if TOWARD_CAMERA else 1.0
                    if TOWARD_CAMERA and fwd_cam[2, 0] < 0:
                        # if forward already points toward camera (negative z), keep sign positive
                        base_sign = 1.0

                    # 3) Build a tentative endpoint in camera space
                    end_cam = nose_cam + base_sign * LINE_LEN * fwd_cam

                    # 4) Project nose & end to pixels
                    def project_cam_to_px(Xc, K):
                        x, y, z = float(Xc[0,0]), float(Xc[1,0]), float(Xc[2,0])
                        z = z if z > 1e-6 else 1e-6
                        u = (K[0,0]*x/z) + K[0,2]
                        v = (K[1,1]*y/z) + K[1,2]
                        return np.array([u, v], dtype=np.float32)

                    p1_px = project_cam_to_px(nose_cam, camera_matrix)
                    p2_px = project_cam_to_px(end_cam,  camera_matrix)

                    # 5) Ensure the 2D direction aligns with pitch semantics:
                    #    pitch > 0 (face up) => line should go up on screen (dir_y < 0)
                    #    pitch < 0 (face down) => line should go down (dir_y > 0)
                    dir2d = p2_px - p1_px
                    if (pitch > 0 and dir2d[1] > 0) or (pitch < 0 and dir2d[1] < 0):
                        # flip the 3D direction and re-project once
                        end_cam = nose_cam - base_sign * LINE_LEN * fwd_cam
                        p2_px   = project_cam_to_px(end_cam, camera_matrix)
                        dir2d   = p2_px - p1_px  # refreshed

                    # 6) Anchor at detected 2D nose but use the corrected direction
                    image_nose = np.array([image_points[0][0], image_points[0][1]], dtype=np.float32)
                    p2_final = (image_nose + dir2d).astype(int)
                    p1_final = image_nose.astype(int)

                    cv2.line(frame, tuple(p1_final), tuple(p2_final), (255, 0, 0), 2)
                    # ----- end nose direction line -----

                    # Axes + overlays
                    _draw_axes(frame, camera_matrix, dist_coeffs, rvec, tvec, axis_len=10)
                    _overlay_text(frame, yaw, pitch, roll)

                    # Pitch threshold alerts
                    alert_text = None
                    if pitch > PITCH_UP_THRESHOLD:
                        alert_text = f"ALERT:  FACE DOWN!!({pitch:.1f}°)"
                    elif pitch > PITCH_DOWN_THRESHOLD:
                        alert_text = f"ALERT:  FACE UP!!({pitch:.1f}°)"
                    if alert_text:
                        cv2.putText(frame, alert_text, (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
                        print(alert_text)
    except Exception as _e:
        # fail-soft
        pass

    # (eye/yawn regions & visualization)
    left_eye_top_x =  result[0][0]['keypoints']['left_eye'][0]  - 32
    left_eye_top_y =  result[0][0]['keypoints']['left_eye'][1]  - 32
    left_eye_bottom_x = result[0][0]['keypoints']['left_eye'][0] + 32
    left_eye_bottom_y = result[0][0]['keypoints']['left_eye'][1] + 32
    cv2.rectangle(frame, (int(left_eye_top_x), int(left_eye_top_y)), (int(left_eye_bottom_x), int(left_eye_bottom_y)), (0,255,0), 2)

    right_eye_top_x =  result[0][0]['keypoints']['right_eye'][0]  - 32
    right_eye_top_y =  result[0][0]['keypoints']['right_eye'][1]  - 32
    right_eye_bottom_x = result[0][0]['keypoints']['right_eye'][0] + 30
    right_eye_bottom_y = result[0][0]['keypoints']['right_eye'][1] + 30
    cv2.rectangle(frame, (int(right_eye_top_x), int(right_eye_top_y)), (int(right_eye_bottom_x), int(right_eye_bottom_y)), (0,255,0), 2)

    # mouth crop
    mx1 = min(result[0][0]['keypoints']['mouth_left'][0],  result[0][0]['keypoints']['mouth_right'][0]) - 10
    my1 = min(result[0][0]['keypoints']['mouth_left'][1],  result[0][0]['keypoints']['mouth_right'][1]) - 40 + 23
    mx2 = max(result[0][0]['keypoints']['mouth_left'][0],  result[0][0]['keypoints']['mouth_right'][0]) + 10
    my2 = max(result[0][0]['keypoints']['mouth_left'][1],  result[0][0]['keypoints']['mouth_right'][1]) + 40 + 13
    cv2.rectangle(frame, (mx1, my1), (mx2, my2), (0, 255, 0), 2)

    # predict_blink(
    #     frame[int(left_eye_top_y):int(left_eye_bottom_y), int(left_eye_top_x):int(left_eye_bottom_x)],
    #     frame[int(right_eye_top_y):int(right_eye_bottom_y), int(right_eye_top_x):int(right_eye_bottom_x)],
    #     frame
    # )
    # predict_yawn(frame[int(my1):int(my2), int(mx1):int(mx2)], frame)

    # Build ROIs (use your existing coords: left_eye_top_y, etc.)
    left_eye  = crop_roi(frame, left_eye_top_x,  left_eye_top_y,  left_eye_bottom_x,  left_eye_bottom_y)
    right_eye = crop_roi(frame, right_eye_top_x, right_eye_top_y, right_eye_bottom_x, right_eye_bottom_y)
    mouth     = crop_roi(frame, mx1, my1, mx2, my2)
    # #cv2.imwrite("./yawn_crop.jpg", frame[int(y1):int(y2), int(x1):int(x2)])
    # if future_blink is None or future_blink.done():
    #     future_blink = executor.submit(predict_blink,left_eye,right_eye,frame)
    # if future_yawn is None or future_yawn.done():
    #     future_yawn = executor.submit(predict_yawn,mouth,frame)

    # # blink_detectedの状態を左下に表示
    # blink_status = "True" if blink_detected else "False"
    # blink_color = (0, 0, 255) if blink_detected else (0, 255, 0)  # 赤：検出中、緑：非検出
    # cv2.putText(frame, f"Blink Detected: {blink_status}", (20, frame.shape[0] - 50), 
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, blink_color, 2)

    # # yawn_detectedの状態も左下に表示
    # yawn_status = "True" if yawn_detected else "False" 
    # yawn_color = (0, 0, 255) if yawn_detected else (0, 255, 0)  # 赤：検出中、緑：非検出
    # cv2.putText(frame, f"Yawn Detected: {yawn_status}", (20, frame.shape[0] - 20), 
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, yawn_color, 2)

    # Submit jobs (non-blocking)
    if future_blink is None or future_blink.done():
        future_blink = executor.submit(predict_blink, left_eye, right_eye, frame)
        future_blink_start = perf_counter()

    if future_yawn is None or future_yawn.done():
        future_yawn = executor.submit(predict_yawn, mouth, frame)
        future_yawn_start = perf_counter()

    # Collect timings when each job completes (non-blocking)
    # if future_blink is not None and future_blink.done() and future_blink_start is not None:
    #     try:
    #         _ = future_blink.result()  # pull exceptions if any (won’t block because it's done)
    #     except Exception as e:
    #         print("predict_blink error:", e)
    #     last_blink_ms = (perf_counter() - future_blink_start) * 1000.0
    #     future_blink_start = None  # reset so the next submit re-times

    # if future_yawn is not None and future_yawn.done() and future_yawn_start is not None:
    #     try:
    #         _ = future_yawn.result()
    #     except Exception as e:
    #         print("predict_yawn error:", e)
    #     last_yawn_ms = (perf_counter() - future_yawn_start) * 1000.0
    #     future_yawn_start = None

    # ---- existing overlays ----
    # Blink status (left-bottom)
    blink_status = "True" if blink_detected else "False"
    blink_color = (0, 0, 255) if blink_detected else (0, 255, 0)
    cv2.putText(frame, f"Blink Detected: {blink_status}", (20, frame.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, blink_color, 2)

    # Yawn status (left-bottom)
    yawn_status = "True" if yawn_detected else "False"
    yawn_color = (0, 0, 255) if yawn_detected else (0, 255, 0)
    cv2.putText(frame, f"Yawn Detected: {yawn_status}", (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, yawn_color, 2)

    # ---- new: show last execution times (right-bottom) ----
    h, w = frame.shape[:2]
    if last_blink_ms is not None:
        cv2.putText(frame, f"Blink time: {last_blink_ms:.1f} ms",
                    (w - 280, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    if last_yawn_ms is not None:
        cv2.putText(frame, f"Yawn time:  {last_yawn_ms:.1f} ms",
                    (w - 280, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # FPSを画面右上に表示
    cv2.putText(frame, f"FPS: {current_fps:.1f}", (frame.shape[1] - 120, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow(WINDOW_NAME, frame)
    out.write(frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
try:
    pygame.mixer.quit()
except:
    pass