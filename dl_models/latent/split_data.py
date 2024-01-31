import os
import shutil
from sklearn.model_selection import train_test_split
import numpy as np

# Define the source and target directories
source_dir = '/data/albert/latent_302/latent/train'
base_dir = '/data/albert/latent_302/latent'

# Ensure the existence of target directories for train, test, and val
for split in ['train', 'test', 'val']:
    os.makedirs(os.path.join(base_dir, split), exist_ok=True)

# List all subfolders in the source directory
subfolders = [f.path for f in os.scandir(source_dir) if f.is_dir()]
# Shuffle the list for randomness
np.random.shuffle(subfolders)

# Split the data
train_val, test = train_test_split(subfolders, test_size=0.2, random_state=42)  # 80% for training + validation, 20% for testing
train, val = train_test_split(train_val, test_size=0.125, random_state=42)  # 0.125 * 80% = 10% for validation

# Function to move subfolders to target directories
def move_subfolders(subfolders, target_dir):
    for folder in subfolders:
        # Extract the name of the subfolder
        folder_name = os.path.basename(folder)
        # Define the target path for the subfolder
        target_path = os.path.join(target_dir, folder_name)
        # Move the subfolder to the target directory
        shutil.move(folder, target_path)
        print(f"Moved {folder} to {target_path}")

# Move the subfolders to their respective new directories
move_subfolders(train, os.path.join(base_dir, 'train'))
move_subfolders(test, os.path.join(base_dir, 'test'))
move_subfolders(val, os.path.join(base_dir, 'val'))
