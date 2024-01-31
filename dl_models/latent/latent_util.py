import matplotlib.pyplot as plt
import os
from torchvision import transforms

def plot_and_save_ldr(ldr_name, ldr):
    imgs, ids, fngr_nums = next(iter(ldr))
    # Convert tensor to PIL Image for displaying if not already in PIL format
    to_pil = transforms.ToPILImage()
    img_pil = to_pil(imgs[0].squeeze())

    # Now display and save using PIL handling
    plt.imshow(img_pil, cmap='gray')
    plt.title(f'ID: {ids[0]}, Finger: {fngr_nums[0]}')
    
    # Ensure the directory is correctly referenced relative to the current script's location
    save_dir = os.path.join(os.path.dirname(__file__), 'test')
    os.makedirs(save_dir, exist_ok=True)

    # Save the plot
    plt.savefig(os.path.join(save_dir, f'{ldr_name}_ID_{ids[0]}_Finger_{fngr_nums[0]}.png'))
