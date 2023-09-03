import torch
import torch.functional as F
import numpy as np
import torchvision
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
from pytorch_grad_cam import GradCAM, AblationCAM

import argparse
from tqdm import tqdm

import sys
sys.path.append("../")
sys.path.append('../dl_models')
from dl_models.embedding_models import EmbeddingNet
from dl_models.parameterized_multiple_finger_tester import load_data, create_output_dir, DEFAULT_OUTPUT_ROOT, ALL_FINGERS

# Thanks https://github.com/jacobgil/pytorch-grad-cam
# Thanks https://jacobgil.github.io/pytorch-gradcam-book/Pixel%20Attribution%20for%20embeddings.html

class SimilarityToConceptTarget:
    def __init__(self, features):
        self.features = features
        return
    
    def __call__(self, model_output):
        #cos = torch.nn.CosineSimilarity(dim=0)
        #res = cos(model_output.squeeze(), self.features.squeeze())
        res = -torch.sqrt(torch.sum((model_output - self.features).pow(2)))
        #print(res)
        return res
    
class DissimilarityToConceptTarget:
    def __init__(self, features):
        self.features = features
        return
    
    def __call__(self, model_output):
        #cos = torch.nn.CosineSimilarity(dim=0)
        #res = cos(model_output.squeeze(), self.features.squeeze())
        res = torch.sqrt(torch.sum((model_output - self.features).pow(2)))
        #print(res)
        return res

def create_float_img(tensor_image):
    image_float = tensor_image.squeeze().cpu().numpy()
    image_float = (image_float - image_float.min()) / (image_float.max() - image_float.min())
    image_float = np.transpose(image_float, (1, 2, 0))
    return image_float

if __name__ == "__main__":
    parser = argparse.ArgumentParser('parameterized_multiple_finger_tester.py')
    parser.add_argument('--dataset', help='Path to folders containing images', 
                        default='/data/therealgabeguo/fingerprint_data/sd302_split', type=str)
    parser.add_argument('--weights', help='Path to model weights', 
                        default='/data/therealgabeguo/most_recent_experiment_reports/jan_08_resnet18Final/weights_2023-01-07_11:06:28.pth',
                        type=str)
    parser.add_argument('--output_root', help='Root directory for output',
                        default='/data/therealgabeguo/gradcam_outputs', nargs='?', \
                        type=str)

    args = parser.parse_args()

    dataset = args.dataset
    weights = args.weights
    output_dir = create_output_dir(args.output_root)

    print(args)

    fingerprint_dataset, test_dataset, test_dataloader = load_data(
        the_data_folder=args.dataset, \
        num_anchors=1, num_pos=1, num_neg=1, scale_factor=1, \
        diff_fingers_across_sets=True, diff_fingers_within_set=True, \
        diff_sensors_across_sets=True, same_sensor_within_set=True, \
        possible_fgrps=ALL_FINGERS
    )

    pretrained_model = EmbeddingNet()
    pretrained_model.load_state_dict(torch.load(args.weights, map_location=torch.device('cuda')))
    pretrained_model = pretrained_model.cuda()

    print(pretrained_model.feature_extractor)

    target_layers = [
        pretrained_model.feature_extractor[7][1],
        pretrained_model.feature_extractor[6][1]
    ]
    cam = GradCAM(
        model=pretrained_model,
        target_layers=target_layers,
        use_cuda=True
    )

    data_iter = iter(test_dataloader)
    for i in range(5):#tqdm(range(len(test_dataloader))):
        # test_images is 3 (anchor, pos, neg) * N (number of sample images) * image_size (1*3*224*224)
        test_images, _, test_filepaths = next(data_iter)
        test_images = [torch.unsqueeze(curr_img[0], 0).cuda() for curr_img in test_images]
        test_filepaths = [curr_filepath[0] for curr_filepath in test_filepaths]
        # 0th image is anchor, 1st image is positive, 2nd image is negative
        anchor_embedding = pretrained_model(test_images[0])
        print(anchor_embedding.shape)
        pos_sim_targets = [SimilarityToConceptTarget(pretrained_model(test_images[1]))]
        neg_sim_targets = [SimilarityToConceptTarget(pretrained_model(test_images[2]))]
        pos_dis_targets = [DissimilarityToConceptTarget(pretrained_model(test_images[1]))]
        neg_dis_targets = [DissimilarityToConceptTarget(pretrained_model(test_images[2]))]

        anchor_image_float = create_float_img(test_images[0])
        pos_image_float = create_float_img(test_images[1])
        neg_image_float = create_float_img(test_images[2])

        pos_grayscale_sim_cam = cam(input_tensor=test_images[0], targets=pos_sim_targets, aug_smooth=True, eigen_smooth=True)[0, :]
        pos_sim_cam_image = show_cam_on_image(anchor_image_float, pos_grayscale_sim_cam, use_rgb=True, colormap=cv2.COLORMAP_HOT)

        neg_grayscale_sim_cam = cam(input_tensor=test_images[0], targets=neg_sim_targets, aug_smooth=True, eigen_smooth=True)[0, :]
        neg_sim_cam_image = show_cam_on_image(anchor_image_float, neg_grayscale_sim_cam, use_rgb=True, colormap=cv2.COLORMAP_HOT)

        pos_grayscale_dis_cam = cam(input_tensor=test_images[0], targets=pos_dis_targets, aug_smooth=True, eigen_smooth=True)[0, :]
        pos_dis_cam_image = show_cam_on_image(anchor_image_float, pos_grayscale_dis_cam, use_rgb=True, colormap=cv2.COLORMAP_OCEAN)

        neg_grayscale_dis_cam = cam(input_tensor=test_images[0], targets=neg_dis_targets, aug_smooth=True, eigen_smooth=True)[0, :]
        neg_dis_cam_image = show_cam_on_image(anchor_image_float, neg_grayscale_dis_cam, use_rgb=True, colormap=cv2.COLORMAP_OCEAN)
        
        # plt.imshow(pos_heatmap)
        # plt.savefig('dummy.png')
        
        num_rows = 2
        fig, axes = plt.subplots(num_rows + 1, 3, figsize=(3 * 5, 5 * (num_rows + 1)))

        for row_num in range(num_rows + 1):
            for col_num in range(3):
                axes[row_num, col_num].axis('off')

        axes[0, 1].imshow(pos_image_float); axes[0, 1].set_title('Same Person:\nComparison Fingerprint')
        axes[0, 2].imshow(neg_image_float); axes[0, 2].set_title('Different Person\n:Comparison Fingerprint')

        axes[1, 0].imshow(anchor_image_float); axes[1, 0].set_title('Reference Fingerprint:\nSimilarity Heatmap')
        axes[1, 1].imshow(pos_sim_cam_image)
        axes[1, 2].imshow(neg_sim_cam_image)
        axes[2, 0].imshow(anchor_image_float); axes[2, 0].set_title('Reference Fingerprint:\nDissimilarity Heatmap')
        axes[2, 1].imshow(pos_dis_cam_image)
        axes[2, 2].imshow(neg_dis_cam_image)
        

        plt.savefig('sample_saliency_map_{}.png'.format(i))

