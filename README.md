# cv_models

Implement common deep networks for computer vision with pytorch from scratch.

[![issues](https://img.shields.io/github/issues/SmartPolarBear/cv_models)](https://github.com/SmartPolarBear/cv_models/issues)
[![forks](https://img.shields.io/github/forks/SmartPolarBear/cv_models)](https://github.com/SmartPolarBear/cv_models/fork)
[![stars](https://img.shields.io/github/stars/SmartPolarBear/cv_models)](https://github.com/SmartPolarBear/cv_models/stargazers)
[![license](https://img.shields.io/github/license/SmartPolarBear/cv_models)](https://github.com/SmartPolarBear/cv_models/blob/master/LICENSE)
[![twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2F___zirconium___)](https://twitter.com/___zirconium___)

## Dependencies

### Packages

- Pytorch 1.12.0

### Environment

- Python 3.10

## Backbones

### Common Backbones

| Model              | Status | Paper                                                                                                                                                                                |
|--------------------|--------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ConvNext           | 🔄️    | [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)                                                                                                                          |
| MLP-Mixer          | ✅      | [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601)                                                                                                    |
| ResNet             | ✅      | []()                                                                                                                                                                                 |
| ResNeXt            | 🔄️    | []()                                                                                                                                                                                 |
| BoTNet             | 🔄️    | [Bottleneck Transformers for Visual Recognition](https://openaccess.thecvf.com//content/CVPR2021/papers/Srinivas_Bottleneck_Transformers_for_Visual_Recognition_CVPR_2021_paper.pdf) |
| Swin Transformer   | 🔄️    | []()                                                                                                                                                                                 |
| Vision Transformer | ✅      | [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929v2)                                                                     |
| Xception           | ✅      | [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)                                                                                    |

### Lightweight/Mobile Backbones

| Model      | Status | Paper |
|------------|--------|-------|
| MobileNet  | ✅      | []() |
| MobileViT  | 🔄️    | []() |
| ShuffleNet | ✅      | []() |

## Segmentation

| Model       | Status | Paper |
|-------------|--------|-------|
| UNet        | 🔄️    | []()  |
| DeepLab V3+ | 🔄️    | []()  |

## Detection

No detailed plan yet.

## 3D

No detailed plan yet.

## Plugin-in Modules

No detailed plan yet.

## License

Copyright (c) 2022 SmartPolarBear

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.