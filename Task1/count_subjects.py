import os

directory = 'data/'
sub_dirs = ["drowsy", "non_drowsy"]

for sub_dir in sub_dirs:
    subject_counts = {}
    path = directory + sub_dir
    for filename in os.listdir(path):
        if os.path.isfile(os.path.join(path, filename)):
            subject = filename[0]
            if subject_counts.get(subject) == None:
                subject_counts[subject] = 1
            else:
                subject_counts[subject] += 1

    print(len(subject_counts.keys()))
    for subject, count in subject_counts.items():
        print(f"Subject {subject}: {count} files")
    print("")
