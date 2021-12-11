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
    <td colspan="2"><img src = "https://i.ibb.co/3WRbpYv/CMFNet.png" alt="CMFNet" width="800"> </td>  
  </tr>
  <tr>
    <td colspan="2"><p align="center"><b>Overall Framework of CMFNet</b></p></td>
  </tr>
  
  <tr>
    <td> <img src = "https://i.ibb.co/FBx9QLy/UNet.png" width="400"> </td>
    <td> <img src = "https://i.ibb.co/W0yk5hn/MSC.png" width="400"> </td>
  </tr>
  <tr>
    <td><p align="center"><b>Branch U-Net architecture</b></p></td>
    <td><p align="center"> <b>Mixed Skip Connection (MSC)</b></p></td>
  </tr>
</table>
