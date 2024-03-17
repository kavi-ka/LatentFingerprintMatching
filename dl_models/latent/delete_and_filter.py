import os
from PIL import Image
import numpy as np

'''
Given a parent directory, checks all images in that directory's subdirectories
If an image is blank, it prints the image path to log_file

Also filters to only the necessary finger images (ending in 1),
prints images to be deleted in log_file2
'''

# The parent directory's path
parent_dir = '/data/albert/latent_302/latent_8bit/train'

# name of log file created by script
log_file = 'blank_img_list.txt' 
log_file2 = 'unwanted_imgs.txt'

def is_blank(img_pth, threshold=235):
    try:
        with Image.open(img_pth) as img:
            img_array = np.array(img).flatten()
            image_sum = np.average(img_array)
            # print(image_sum)
            if image_sum >= threshold:
                # print(222, img_pth)
                return True
            return False
    except Exception as e:
        print(f"error processing {img_pth}: {e}")
        return False

def find_blanks(dir):
    imgs = []
    for root, dirs, files in os.walk(dir):
        for f in files:
            if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                f_path = os.path.join(root, f)
                if is_blank(f_path):
                    imgs.append(f_path)

    return imgs

def get_subdirs(dir):
    subdirs = next(os.walk(dir))[1]
    return subdirs

# returns non _1 images
def filter_imgs(dir):
    imgs = []
    s = set()
    print(dir)
    for root, dirs, files in os.walk(dir):
        for f in files:
            if not f.lower().endswith(('_1.png', '_1.jpg', '_1.jpeg', '_1.bmp', '_1.gif')):
                imgs.append(f)
    return imgs


# subdirs = get_subdirs(parent_dir)

# white_imgs, del_imgs = [],[]
# for sbdr in subdirs:
#     curr_dir = parent_dir + "/" + sbdr
#     white_imgs += find_blanks(curr_dir)
#     del_imgs += filter_imgs(curr_dir)

# if os.path.exists(log_file):
#     os.remove(log_file)

# with open(log_file, 'w') as f: 
#     for img in white_imgs:
#          f.write(f"{img} \n")

# with open(log_file2, 'w') as f: 
#     for img in del_imgs:
#          f.write(f"{img} \n")
    
