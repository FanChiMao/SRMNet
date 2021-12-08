"""
## Reference from: Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch
import utils

from SRMNet import SRMNet
from skimage import img_as_ubyte
import scipy.io as sio

parser = argparse.ArgumentParser(description='Image Denoising using SRMNet')

parser.add_argument('--input_dir', default='D:/NCHU/Dataset/Denoise/Real-world noise/SIDD/test', type=str,
                    help='Directory of validation images')
parser.add_argument('--result_dir', default='./test_results/SIDD/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./checkpoints/SRMNet_real_denoise/models/model_bestPSNR.pth', type=str,
                    help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--save_images', default=False, help='Save denoised images in result directory')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

result_dir = os.path.join(args.result_dir, 'mat')
utils.mkdir(result_dir)

if args.save_images:
    result_dir_img = os.path.join(args.result_dir, 'png')
    utils.mkdir(result_dir_img)

model_restoration = SRMNet()

utils.load_checkpoint(model_restoration, args.weights)
print("===>Testing using weights: ", args.weights)
model_restoration.cuda()
# model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

# Process data                           # BenchmarkNoisyBlocksSrgb.mat     ValidationNoisyBlocksSrgb.mat
filepath = os.path.join(args.input_dir, 'ValidationNoisyBlocksSrgb.mat')
img = sio.loadmat(filepath)

Inoisy = np.float32(np.array(img['ValidationNoisyBlocksSrgb']))  # ValidationNoisyBlocksSrgb
Inoisy /= 255.
restored = np.zeros_like(Inoisy)

import time
start = time.time()
with torch.no_grad():
    for i in tqdm(range(40)):
        for k in range(32):  # 32
            noisy_patch = torch.from_numpy(Inoisy[i, k, :, :, :]).unsqueeze(0).permute(0, 3, 1, 2).cuda()
            restored_patch = model_restoration(noisy_patch)
            restored_patch = torch.clamp(restored_patch, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0)
            restored[i, k, :, :, :] = restored_patch

            if args.save_images:
                save_file = os.path.join(result_dir_img, '%04d_%02d.png' % (i + 1, k + 1))
                utils.save_img(save_file, img_as_ubyte(restored_patch))

# save denoised data

print('Process time each patch:', (time.time() - start)/1280)
sio.savemat(os.path.join(result_dir, 'Idenoised.mat'), {"Idenoised": restored, })
