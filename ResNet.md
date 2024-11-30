# ResNet 

**殘差網路**（residual network，**ResNet**）是一種深度[卷積神經網路](https://hackmd.io/@qavit/convolutional_neural_network)架構，於 2015 年由 Kaiming He 等人提出[^he15]，它以「殘差學習」的概念解決了隨著網路深度加深時出現的[梯度消失](https://hackmd.io/@qavit/vanishing_gradient_problem)和退化問題。


## 模型結構

### 殘差塊

**殘差塊**（residual block）是一種 skip connection

> ![](https://upload.wikimedia.org/wikipedia/commons/b/ba/ResBlock.png)
> **殘差塊**（residual block）示意圖。Credit: [Wikimedia](https://commons.wikimedia.org/wiki/File:ResBlock.png)



> ![](https://production-media.paperswithcode.com/methods/Screen_Shot_2020-09-25_at_10.26.40_AM_SAB79fQ.png =50%x)
> Credit: [^he15], Fig. 5

## 特性

## 發展與成就

- ILSVRC 2015 影像分類、檢測、定位任務冠軍[^ilsvrc2015]
- MS COCO 2015 影像檢測、語義分割冠軍
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) — highly-cited paper

---

## 實作

### PyTorch + torchvision
- `torchvision.models` 中的 [ResNet](https://pytorch.org/vision/main/models/resnet.html)
    - [`torchvision.models.resnet18`](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html)
    - `torchvision.models.resnet34`
    - `torchvision.models.resnet50`
    - `torchvision.models.resnet101`
    - `torchvision.models.resnet152`

#### CIFAR10

1. ==![](https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg =35x) **Google Colab:** [`CNN_ResNet_CIFAR10_hello.ipynb`](https://colab.research.google.com/drive/1g9TEfyow-3LxrlPWJYwCcyhMObJHEEPq?usp=sharing)==

### TensorFlow Keras

- [`tensorflow.keras.applications`](https://www.tensorflow.org/api_docs/python/tf/keras/applications)
    - [`tf.keras.applications.resnet`](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet)
    - [`tf.keras.applications.resnet50`](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50)
    - [`tf.keras.applications.resnet_v2`](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet_v2)
        - `ResNet101V2()`, `ResNet152V2()`, `ResNet50V2()`

---

# 延伸閱讀

- https://paperswithcode.com/method/resnet

<!-- Footnotes & References -->
[^he15]: He et al. (2015). Deep Residual Learning for Image Recognition. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385). 

[^ilsvrc2015]: [Large Scale Visual Recognition Challenge 2015 (ILSVRC2015)](https://web.archive.org/web/20230704172431/https://image-net.org/challenges/LSVRC/2015/results). ImageNet.