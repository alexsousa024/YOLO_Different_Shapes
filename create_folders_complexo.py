import os
import shutil
from sklearn.model_selection import train_test_split

# ---- Configuration ----
source_images_path = 'dataset_inicial/images'
source_labels_path = 'dataset_inicial/labels'
output_dir = 'split_dataset_500'

img_ext = '.png'

# ---- Split Ratios ----
train_ratio = 0.70
val_ratio = 0.30


random_state = 42
force_overwrite = True

# ---- Helper Functions ----
def save_list_to_file(filepath, data_list):
    with open(filepath, 'w') as f:
        for item in data_list:
            f.write(f"{item}\n")
    print(f"Saved {len(data_list)} filenames to {filepath}")

def make_dirs(path):
    if os.path.exists(path) and force_overwrite:
        shutil.rmtree(path)
    os.makedirs(path)
    os.makedirs(os.path.join(path, 'images'))
    os.makedirs(os.path.join(path, 'labels'))
    print(f"Created directory structure under: {path}")

def copy_files(file_list, dest_dir):
    for file in file_list:
        img_file = file + img_ext
        label_file = file + '.txt'

        shutil.copy(os.path.join(source_images_path, img_file), os.path.join(dest_dir, 'images', img_file))
        shutil.copy(os.path.join(source_labels_path, label_file), os.path.join(dest_dir, 'labels', label_file))

# ---- Main Logic ----
# List all image base filenames (without extension)
all_images = sorted([os.path.splitext(f)[0] for f in os.listdir(source_images_path) if f.endswith(img_ext)])[:500]

# First split into train and temp (val + test)
train_images, val_images = train_test_split(all_images, train_size=train_ratio, random_state=random_state)

# Create directories
make_dirs(os.path.join(output_dir, 'train'))
make_dirs(os.path.join(output_dir, 'val'))


# Copy files to respective folders
copy_files(train_images, os.path.join(output_dir, 'train'))
copy_files(val_images, os.path.join(output_dir, 'val'))


# Optionally save file lists
save_list_to_file(os.path.join(output_dir, 'train_list.txt'), train_images)
save_list_to_file(os.path.join(output_dir, 'val_list.txt'), val_images)


print("Dataset splitting and copying completed.")
