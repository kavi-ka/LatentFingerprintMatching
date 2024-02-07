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
from PIL import Image

# Specify the source and destination directories
src_dir = "/data/albert/latent_302/latent_8bit/train/00002302/"
dst_dir = "/home/albert/crystal/LatentFingerprintMatching/dl_models/latent/inspect_images/move"

# Get a list of all PNG files in the source directory
files = [f for f in os.listdir(src_dir) if f.endswith('.png')]

# Process the first 5 images
for file in files[:5]:
    print(file)
    # Open the image file
    img = Image.open(os.path.join(src_dir, file))

    # Convert the image to 8-bit (256 color) mode
    img = img.convert('P', dither=Image.NONE)

    # Save the image to the destination directory
    img.save(os.path.join(dst_dir, file))