import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



import os

def access():

    file_path = '/data/albert/latent_302/latent_8bit_toy/train/00002488/00002488_7D_X_135_CA_D800_1074PPI_16BPC_1CH_LP01_1.png'

    if os.access(file_path, os.R_OK):
        print("File is accessible.")
    else:
        print("File is not accessible.")

    file_size = os.path.getsize(file_path)

    print(f"The size of the file is {file_size} bytes.")

import matplotlib.pyplot as plt

def testing_image_save():
    # Sample data
    x = [1, 2, 3, 4, 5]
    y = [2, 3, 5, 7, 11]

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot data
    ax.plot(x, y)

    # Set labels
    ax.set_xlabel('X-axis label')
    ax.set_ylabel('Y-axis label')
    ax.set_title('Sample Plot')

    # Save the plot
    plt.savefig('plot.png')

    # Show the plot
    plt.show()

import os
from PIL import Image, ImageOps, ImageEnhance

# Specify the source and destination directories
latent_src_dir = "/data/albert/latent_302/latent_8bit/train/00002302/"
latent_dst_dir = "/home/albert/crystal/LatentFingerprintMatching/dl_models/latent/inspect_images/move"

real_src_dir = "/data/therealgabeguo/fingerprint_data/sd302_split/train/00002502"
real_dst_dir = "/home/albert/crystal/LatentFingerprintMatching/dl_models/latent/inspect_images/move_real"

filter_src = "/data/albert/302_latent_data/00002360"
filter_dst = "/home/albert/crystal/LatentFingerprintMatching/dl_models/latent/inspect_images/move_filter"

from delete_and_filter import is_blank

# def copy_images_for_inspect(src_dir, dst_dir):

#     # Check if the destination directory exists
#     if not os.path.exists(dst_dir):
#         # If not, create it
#         os.makedirs(dst_dir)
#     else:
#         # If it does, clear it
#         for filename in os.listdir(dst_dir):
#             file_path = os.path.join(dst_dir, filename)
#             try:
#                 if os.path.isfile(file_path) or os.path.islink(file_path):
#                     os.unlink(file_path)
#                 elif os.path.isdir(file_path):
#                     shutil.rmtree(file_path)
#             except Exception as e:
#                 print('Failed to delete %s. Reason: %s' % (file_path, e))

#     # Get a list of all PNG files in the source directory
#     files = [f for f in os.listdir(src_dir) if f.endswith('.png')]

#     # Process the first 5 images
#     for file in files[:100]:
#         print(file)
#         full_path = os.path.join(src_dir, file)

#         if is_blank(full_path):
#             # Open the image file
#             img = Image.open(os.path.join(src_dir, file))

#             # Convert the image to 8-bit (256 color) mode
#             img = img.convert('P', dither=Image.NONE)

#             # Save the image to the destination directory
#             img.save(os.path.join(dst_dir, file))
#         else:
#             # Open the image file
#             img = Image.open(os.path.join(src_dir, file))

#             # Convert the image to 8-bit (256 color) mode
#             img = img.convert('P', dither=Image.NONE)

#             # Save the image to the destination directory
#             img.save(os.path.join(dst_dir, file))
        


# copy_images_for_inspect(latent_src_dir, latent_dst_dir)
# copy_images_for_inspect(real_src_dir, real_dst_dir)
# copy_images_for_inspect(filter_src, latent_dst_dir)



def copy_images_for_inspect(src_dir, dst_dir):

    # Check if the destination directory exists
    if not os.path.exists(dst_dir):
        # If not, create it
        os.makedirs(dst_dir)
    else:
        # If it does, clear it
        for filename in os.listdir(dst_dir):
            file_path = os.path.join(dst_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    print("dir prepared...")
    # Create subdirectories for blank and non-blank images
    blank_dir = os.path.join(dst_dir, 'blank')
    non_blank_dir = os.path.join(dst_dir, 'non_blank')
    os.makedirs(blank_dir, exist_ok=True)
    os.makedirs(non_blank_dir, exist_ok=True)

    # Get a list of all PNG files in the source directory
    files = [f for f in os.listdir(src_dir) if f.endswith('.png')]
    contrast_factor = 2
    # Process the first 5 images
    for file in files:
        print(111, file)
        full_path = os.path.join(src_dir, file)

        # Open the image file
        img = Image.open(full_path).convert('L')
        img = ImageOps.autocontrast(img)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_factor)

        # Convert the image to 8-bit (256 color) mode
        img = img.convert('P', dither=Image.NONE)

        

        # Save the image to the appropriate subdirectory
        if is_blank(full_path):
            img.save(os.path.join(blank_dir, file))
        else:
            img.save(os.path.join(non_blank_dir, file))

copy_images_for_inspect(filter_src, latent_dst_dir)