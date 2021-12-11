# Selective Residual M-Net for Real Image Denoising  
> Abstract : Image restoration is a low-level vision task which is restoring
the degraded images to noise-free images. With the success
of deep neural networks, especially the convolutional
neural networks suppress the traditional restoration methods
and become the mainstream in the computer vision. To advance
the performance of denoising algorithms, we propose a
blind real image denoising network (SRMNet) by employing
a hierarchical architecture improved from U-Net. We
use a selective kernel with residual block on the hierarchical
structural named M-Net to enrich the multi-scale semantic
information. Furthermore, our SRMNet has competitive
performance results on three synthetic and two realworld
noisy datasets in terms of quantitative metrics and
visual quality.

## Network Architecture  
<table>
  <tr>
    <td colspan="2"><img src = "https://i.imgur.com/GYeypta.png" alt="SRMNet" width="800"> </td>  
  </tr>
  <tr>
    <td colspan="2"><p align="center"><b>Overall Framework of SRMNet</b></p></td>
  </tr>
  
  <tr>
    <td> <img src = "https://i.imgur.com/z6Vds87.png" width="400"> </td>
    <td> <img src = "https://i.imgur.com/eaLejBK.png" width="400"> </td>
  </tr>
  <tr>
    <td><p align="center"><b>Selective Residual Block (SRB)</b></p></td>
    <td><p align="center"> <b>Resizing Block</b></p></td>
  </tr>
</table>


## Quick Run  
To test the pre-trained models of Denoising on your own images, run
```
python demo.py --input_dir images_folder_path --result_dir save_images_here --weights path_to_models
```
Here is an example to perform Deraindrop:
```
python demo.py --input_dir './demo_samples/deraindrop' --result_dir './demo_results' --weights './pretrained_model/deraindrop_model.pth'
```

**All pre-trained models can be downloaded at [pretrained_model/README.md](pretrained_model/README.md) or [here](https://github.com/FanChiMao/CMFNet/releases)**
## Train  
To train the restoration models of Deraindrop, Dehaze and Deblurring. You should check the following components:  
- `training.yaml`:  
  ```
  # Training configuration
  GPU: [0,1,2,3]

  VERBOSE: False

  MODEL:
    MODE: 'Deblur'

  # Optimization arguments.
  OPTIM:
    BATCH: 2
    EPOCHS: 150
    # NEPOCH_DECAY: [10]
    LR_INITIAL: 2e-4
    LR_MIN: 1e-6
    # BETA1: 0.9

  TRAINING:
    VAL_AFTER_EVERY: 1
    RESUME: False
    TRAIN_PS: 256
    VAL_PS: 256
    TRAIN_DIR: './datasets/deraindrop/train'       # path to training data
    VAL_DIR: './datasets/deraindrop/test' # path to validation data
    SAVE_DIR: './checkpoints'           # path to save models and images

  ```
- Dataset:  
  The preparation of dataset in more detail, see [datasets/README.md](datasets/README.md).  
  
- Train:  
  If the above path and data are all correctly setting, just simply run:  
  ```
  python train.py
  ```  


