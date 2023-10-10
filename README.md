# Judging a video by its bitstream cover

This repository is the official implementation of [Judging a video by its bitstream cover](https://arxiv.org/pdf/2309.07361v1.pdf).

## Overview

Classifying videos into distinct categories, such as Sport and Music Video, is crucial for multimedia understanding and retrieval, especially
in an age where an immense volume of video content is constantly being generated. Traditional methods require video decompression to extract pixel-level features like color, texture, and motion, thereby
increasing computational and storage demands. Moreover, these methods often suffer from performance degradation in low-quality videos. We present a novel approach that examines only the post-compression
bitstream of a video to perform classification, eliminating the need for bitstream. We validate our approach using a custombuilt data set comprising over 29,000 YouTube video clips, totaling 6,000 hours and spanning 11 distinct categories. Our preliminary evaluations indicate precision, accuracy, and recall rates well over 80%.

## Requirements
All experiments use the PyTorch library. We recommend installing the following package versions:

* &nbsp;&nbsp; python=3.7 

* &nbsp;&nbsp; pytorch=1.6.0

* &nbsp;&nbsp; torchvision=0.7.0

Dependency packages can be installed using following command:
```
pip install -r requirements.txt
```

## Dateset
### Download
We created a large data set consisting of 29,142 video clips, each containing at least 3,000 frames.
[Download](https://tinyurl.com/ bitstream-video-data)



### Data preprocess
Transcoded the input video to 1.5，1.2，1.0，0.8，0.5 Mbps using the FFmpeg open source H.264/AVC encoder with the same encoding settings.

```
python preprocess.py
```

(Optional) Split the training set into k-fold for the **cross-validation** experiment.

```
python split.py
```

## Training
D2-Net can be trained by running following command:

```
sh tool/train.sh
```
In our experiments, D2-Net is built on the lightweight backbone [U2-Net](https://arxiv.org/abs/1909.06012). Training D2-Net requires at least one V100 GPU with 32G memory. The defualt hyperparameters are set in train.py. Running the training code will generate logs files and saved models in a directory name logs and ckpts, respectively.

## Inference
D2-Net can be tested with a saved model using following command:
```
sh tool/inference.sh
```
The inference code will test all 15 cases with missing modalities together.

## Citation
```
@article{yang2022d2,
  title={D2-Net: Dual Disentanglement Network for Brain Tumor Segmentation with Missing Modalities},
  author={Yang, Qiushi and Guo, Xiaoqing and Chen, Zhen and Woo, Peter YM and Yuan, Yixuan},
  journal={IEEE Transactions on Medical Imaging},
  year={2022},
  publisher={IEEE}
}
```

## Acknowledgement
1. The implementation is based on the repo: [DMFNet](https://github.com/China-LiuXiaopeng/BraTS-DMFNet)
