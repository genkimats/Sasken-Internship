import cv2
import os
import numpy as np
import dlib
from keras.models import load_model
import onnxruntime as ort
import tensorflow as tf
import numpy as np
from tensorflow.python.framework.errors_impl import NotFoundError
import keras
import math
import keras_tuner as kt

pnet_sess = ort.InferenceSession("./pnet.onnx")
rnet_sess = ort.InferenceSession("./rnet.onnx")
onet_sess = ort.InferenceSession("./onet.onnx")

trial_num = 4

model = keras.models.load_model(f"./blinknet_models/blink_model_trained.h5")
# model = keras.models.load_model(f"./blinknet_models/blink_model_trained_{trial_num}.h5")
# model = keras.models.load_model(f"./blinknet_models/blink_model_finetuned_5.h5")

blink_count = 0

#region Functions

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
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None in preprocess()")
    # imgage = cv2.resize(image, (30, 30))  # Ensure correct size
    #image = image.astype('float32') / 255.0  # Normalize to [0,1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension -> (1, 30, 30, 3)
    return image


CNN_CONSEC_FRAMES = 2
# --- Predict function ---
         
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
def run_pnet(img, threshold=0.6, min_face_size=20, scale_factor=0.709):
    is_batch = isinstance(img, list)
    img = img if is_batch else [img]
    images_raw = load_images_batch(img)
    images_normalized, images_oshapes, pad_param = standarize_batch(images_raw, justification="center", normalize=True)
    
    #bboxes_batch = None
    
    scales_groups = [build_scale_pyramid(shape[1], shape[0], min_face_size=min_face_size, scale_factor=scale_factor) for shape in images_oshapes]
    
    # 2. Apply the scales to normalized images
    scales_result, scales_index = apply_scales(images_normalized, scales_groups)
    batch_size = images_normalized.shape[0]
    
    # 3. Get proposals bounding boxes and confidence from the model (PNet)
    
    #pnet_result = [pnet_model(s) for s in scales_result] 
    #pnet_result = [pnet_sess.run(None, {"input": input_tensor.astype(np.float32)}) for input_tensor in scales_result]
    pnet_result = [pnet_sess.run(None, {"input": tf.cast(input_tensor, tf.float32).numpy()}) for input_tensor in scales_result]
    
    #print(pnet_result)
    # 4. Generate bounding boxes per scale group
    bboxes_proposals = [generate_bounding_box(result[0], result[1], 0.6) for result in pnet_result]
    bboxes_batch_upscaled = [upscale_bboxes(bbox, np.asarray([scale] * batch_size)) for bbox, scale in zip(bboxes_proposals, scales_index)]

    # 5. Apply Non-Maximum Suppression (NMS) per scale group
    bboxes_nms = [smart_nms_from_bboxes(b, threshold=0.5, method="union", initial_sort=False) for b in bboxes_batch_upscaled]

    # 6. Concatenate and apply NMS again across all scales
    bboxes_batch = np.concatenate(bboxes_nms, axis=0) if len(bboxes_nms) > 0 else np.empty((0, 6))
    bboxes_batch = smart_nms_from_bboxes(bboxes_batch, threshold=0.7, method="union", initial_sort=True)

    # 7. Resize bounding boxes to square format
    bboxes_batch = resize_to_square(bboxes_batch)
    
    #print(bboxes_batch)
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

def predict_blink(left_eye, right_eye):
    if left_eye.size == 0 or right_eye.size == 0:
        return -1
    left_tensor = preprocess(left_eye)  # Your preprocessing function here
    right_tensor = preprocess(right_eye)
    left_prediction = model.predict(left_tensor)
    right_prediction = model.predict(right_tensor)
    print("Closed prob:", left_prediction[0][0], right_prediction[0][0])
    print("Open prob:", left_prediction[0][1], right_prediction[0][1])
    if (left_prediction[0][0] > left_prediction[0][1] and right_prediction[0][0] > right_prediction[0][1]):
        # Eye closed
        return 1
    else:
        # Eye open
        return 0

#endregion

# Directories
data_dirs = {
    'closed': 'blink_dataset/train/closed',
    'open': 'blink_dataset/train/open'
}

# Counters
total = 0
correct = 0

class_totals = {"closed": 0, "open": 0}
class_correct = {"closed": 0, "open": 0}


threshold = [0.2, 0.5, 0.7]

for label, dir_path in data_dirs.items():
    for filename in os.listdir(dir_path):
        only_name = os.path.splitext(filename)[0]
        extention = os.path.splitext(filename)[1]
        only_name_lst = only_name.split("_")
        side = only_name_lst[-1]
        if side == "right":
            continue
        pair_filename = "_".join(only_name_lst[:-1])
        left_filename = pair_filename + "_left" + extention
        right_filename = pair_filename + "_right" + extention
        
        left_img_path = os.path.join(dir_path, left_filename)
        right_img_path = os.path.join(dir_path, right_filename)
        left_eye_img = cv2.imread(left_img_path)
        right_eye_img = cv2.imread(right_img_path)
        
        if right_eye_img is None or left_eye_img is None:
            continue
        
        predicted_class = predict_blink(left_eye_img, right_eye_img)
        
        if predicted_class == -1:
            print(filename)
            continue
        
        predicted_label = 'closed' if predicted_class == 1 else 'open'
        print(label, "->", predicted_label)
        
        total += 1
        class_totals[label] += 1
        if predicted_label == label:
            correct += 1
            class_correct[label] += 1

accuracy = (correct / total) * 100 if total > 0 else 0
print(f"Total samples: {total}")
print(f"Correct predictions: {correct}")
print(f"Overall Accuracy: {accuracy:.2f}%")

for cls in ["closed", "open"]:
    if class_totals[cls] > 0:
        acc = (class_correct[cls] / class_totals[cls]) * 100
    else:
        acc = 0
    print(f"Class '{cls}': {class_correct[cls]}/{class_totals[cls]} correct ({acc:.2f}% accuracy)")
