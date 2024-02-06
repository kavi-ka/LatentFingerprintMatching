import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def move_and_open_image(source_path, target_path, save_path):
    # Move the image
    shutil.move(source_path, target_path)
    print(f"Moved image from {source_path} to {target_path}")

    # Open and display the image
    img = mpimg.imread(target_path)
    plt.imshow(img)
    plt.show()

    # Save the image
    plt.imsave(save_path, img)
    print(f"Saved image to {save_path}")

# Usage

# image_name = "00002302_7C_X_563_CA_D800_1098PPI_16BPC_1CH_LP02_1.png"
# source_path = '/data/albert/latent_302/latent_8bit/train/00002302/' + image_name
# target_path = '/home/albert/crystal/LatentFingerprintMatching/dl_models/latent/inspect_images/move/' + image_name
# save_path = '/home/albert/crystal/LatentFingerprintMatching/dl_models/latent/inspect_images/save/' + image_name
# move_and_open_image(source_path, target_path, save_path)


import os

file_path = '/data/albert/latent_302/latent_8bit_toy/train/00002488/00002488_7D_X_135_CA_D800_1074PPI_16BPC_1CH_LP01_1.png'

if os.access(file_path, os.R_OK):
    print("File is accessible.")
else:
    print("File is not accessible.")

file_size = os.path.getsize(file_path)

print(f"The size of the file is {file_size} bytes.")