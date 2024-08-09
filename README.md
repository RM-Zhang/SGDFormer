# SGDFormer: One-stage transformer-based architecture for cross-spectral stereo image guided denoising

### [Paper](https://www.sciencedirect.com/science/article/pii/S1566253524003816) [Arxiv](https://arxiv.org/abs/2404.00349)

## Abstract
Cross-spectral image guided denoising has shown its great potential in recovering clean images with rich details, such as using the near-infrared image to guide the denoising process of the visible one. To obtain such image pairs, a feasible and economical way is to employ a stereo system, which is widely used on mobile devices. In this case, it is necessary to first model the stereo correspondence between two images before guided denoising. Current works attempt to generate an aligned guidance image to handle the disparity. However, due to occlusion, spectral differences and noise degradation, the aligned guidance image generally exists ghosting and artifacts, leading to an unsatisfactory denoised result. To address this issue, we propose a one-stage transformer-based architecture, named SGDFormer, for cross-spectral Stereo image Guided Denoising. The architecture integrates the correspondence modeling and feature fusion of stereo images into a unified network. Our transformer block contains a noise-robust cross-attention (NRCA) module and a spatially variant feature fusion (SVFF) module. The NRCA module captures the long-range correspondence of two images in a coarse-to-fine manner to alleviate the interference of noise. The SVFF module further enhances salient structures and suppresses harmful artifacts through dynamically selecting useful information. Thanks to the above design, our SGDFormer can restore artifact-free images with fine structures, and achieves state-of-the-art performance on various datasets. Additionally, our SGDFormer can be extended to handle other unaligned cross-model guided restoration tasks such as guided depth super-resolution.


## Environment
Our SGDFormer is built upon Python 3.7.0 with PyTorch 1.11.0 and CUDA-11.3. Additionally, the [natten](https://github.com/SHI-Labs/NATTEN) (0.14.6+torch1110cu113) and [MultiScaleDeformableAttention](https://github.com/fundamentalvision/Deformable-DETR) (1.0) libraries should be installed.


## Evaluation
We provide 3 demos to evaluate the example paired images from the Flickr1024 dataset, the Kitti Stereo 2015 dataset, the PittsStereo-RGBNIR dataset, respectively.

```Shell
# Evaluation on the Flickr1024 dataset
python test_denoising_Flickr.py

# Evaluation on the Kitti Stereo 2015 dataset
python test_denoising_Kitti.py

# Evaluation on the PittsStereo-RGBNIR dataset
python test_denoising_PittsStereo.py
```


## License
This project is released under the Apache 2.0 license.


## Contact
If you have any other problems, feel free to post questions in the issues section or contact Runmin Zhang (runmin_zhang@zju.edu.cn).


## Acknowledgement
This work is mainly based on [SANet](https://github.com/lustrouselixir/SANet), [NAFNet](https://github.com/megvii-research/NAFNet), [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR), [NATTEN](https://github.com/SHI-Labs/NATTEN) and [IGEV](https://github.com/gangweiX/IGEV), we thank the authors for the contribution.

Moreover, our work is mainly evaluated on the Flickr1024, Kitti Stereo 2015, PittsStereo-RGBNIR, and Dark Flash Stereo Datasets. We thank the authors of the open datasets for their contributions.

## Citation

If you find this project helpful, please consider citing the following paper:
```bibtex
@article{zhang2025sgdformer,
    title = {SGDFormer: One-stage transformer-based architecture for cross-spectral stereo image guided denoising},
    author = {Zhang, Runmin and Yu, Zhu and Sheng, Zehua and Ying, Jiacheng and Cao, Si-Yuan and Chen, Shu-Jie and Yang, Bailin and Li, Junwei and Shen, Hui-Liang},
    journal = {Information Fusion},
    volume = {113},
    pages = {102603},
    year = {2025},
}
```