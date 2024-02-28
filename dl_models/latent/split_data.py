import os
import shutil
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image

# Define the source and target directories
source_dir = '/data/albert/302_latent_data'
base_dir = '/data/albert/302_latent_data_split'

# the original scanned finger print data split is here
non_latent_dir = '/data/therealgabeguo/fingerprint_data/sd302_split'

def get_subfolder_names(dir_path):
    return [dir_path + "/" + name for name in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, name))]

train = get_subfolder_names(non_latent_dir + "/train")
test = get_subfolder_names(non_latent_dir + "/test")
val = get_subfolder_names(non_latent_dir + "/val")

# Ensure the existence of target directories for train, test, and val
for split in ['train', 'test', 'val']:
    os.makedirs(os.path.join(base_dir, split), exist_ok=True)

if False:
    # List all subfolders in the source directory
    subfolders = [f.path for f in os.scandir(source_dir) if f.is_dir()]
    # Shuffle the list for randomness
    np.random.shuffle(subfolders)

    # Split the data
    train_val, test = train_test_split(subfolders, test_size=0.2, random_state=42)  # 80% for training + validation, 20% for testing
    train, val = train_test_split(train_val, test_size=0.125, random_state=42)  # 0.125 * 80% = 10% for validation

def convert_image_to_8bit(image_path):
    img = Image.open(image_path)
    img = img.convert('L')  # convert image to 8-bit grayscale
    img.save(image_path)

# Function to move subfolders to target directories
def move_subfolders(subfolders, target_dir):
    skips = 0
    for folder in subfolders:
        # Extract the name of the subfolder
        folder_name = os.path.basename(folder)

        from_latent_folder = source_dir + "/" + folder_name

        if not os.path.exists(from_latent_folder):
            print("skipping...", folder)
            skips += 1
            continue

        # Define the target path for the subfolder
        target_path = os.path.join(target_dir, folder_name)
        # Move the subfolder to the target directory
        # if not os.path.exists(target_path):
        #     os.makedirs(target_path)

        shutil.copytree(from_latent_folder, target_path)
        print(f"Copy to {folder} to {target_path}")

        # Convert all images in the moved subfolder to 8-bit
        for filename in os.listdir(target_path):
            if filename.endswith('.png'):  # or whatever file type your images are
                image_path = os.path.join(target_path, filename)
                convert_image_to_8bit(image_path)
                print("converted to 8-bit:", image_path)
        
        if toy:
            break

    print("people in orginal split but not in latent...", skips)

# Move the subfolders to their respective new directories
toy = False
move_subfolders(train, os.path.join(base_dir, 'train'))
move_subfolders(test, os.path.join(base_dir, 'test'))
move_subfolders(val, os.path.join(base_dir, 'val'))
