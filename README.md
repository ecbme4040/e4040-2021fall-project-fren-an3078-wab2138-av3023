# Squeeze-and-Excitation Networks
This report summarizes the findings of the original Squeeze-and-Excitation Networks paper and shows a reproduction of the results with Tensorflow [1]. Convolutional neural networks are widely used for image classification in models such as Resnet models. This report investigates the effectiveness of Squeeze-and-Excitation blocks whose aim is to strengthen the inter-channel relationship by rescaling them. The original paper showed that it consistently increased the classification accuracy with various data sets such as Image Net. This paper is going to show how we were able to increase the accuracy of CNN models with the Squeeze-and-Excitation method. Because of technical barriers, slightly different models and data sets were used. We also tried to analyze the effect of different parameters of  Squeeze-and-Excitation blocks on the models, matching the paper’s findings with varying success.

[1] J. Hu, L. Shen, S. Albanie, G. Sun, and E. Wu, Squeeze-and-Excitation Networks. 2019.
## Download the models
Models are ordered by test category folders (link below) 
```
Drive directory tree
full_models/
├── ablation/
│   ├── SE_identity.h5
│   ├── SE_post.h5
│   └── SE_pre.h5
├── cifar10/
│   ├── inception weights/
│   │   ├── inception_v3_custom_cifar10.ckpt.data-00000-of-00001
│   │   └── inception_v3_custom_cifar10.ckpt.index
│   ├── resnet_best.h5
│   ├── SE-inception weights/
│   └── SE_resnet_best.h5
├── cifar100/
│   ├── resnet_best.h5
│   └── SE_resnet_best.h5
├── ratio/
│   ├── se_resnet_ratio_1.h5
│   ├── se_resnet_ratio_2.h5
│   ├── se_resnet_ratio_32.h5
│   ├── se_resnet_ratio_4.h5
│   └── se_resnet_ratio_8.h5
├── stage/
│   ├── stage2.h5
│   ├── stage3.h5
│   └── stage4.h5
└── tinyImageNet/
    ├── resnet_18_aug_best.h5
    ├── resnet_18_best.h5
    ├── resnet_34_aug_best.h5
    ├── resnet_34_best.h5
    ├── resnet_50_aug_best.h5
    ├── resnet_50_best.h5
    ├── SE-resnet_18_aug_best.h5
    ├── SE-resnet_18_best.h5
    ├── SE-resnet_18best.h5
    ├── SE-resnet_34_aug_best.h5
    ├── SE-resnet_34_best.h5
    ├── SE-resnet_50_aug_best.h5
    └── SE-resnet_50_best.h5
```
[Lion drive link](https://drive.google.com/drive/folders/15kpxrvAyOuMmiqZfGMf_imL12vHBefLx?usp=sharing)
(https://drive.google.com/drive/folders/15kpxrvAyOuMmiqZfGMf_imL12vHBefLx?usp=sharing)


## Jupyter noteooks descriptions
### 0. Results overview
- **Load and Test Models.ipynb** loads all the saved models and computes the top-1,3 and 5 accuracy on the associated test data sets.
### 1. CIFAR10 
- **ResNet with CIFAR10.ipynb** Shows the training process and results of ResNet et SE-Resnet models on CIFAR-10
- **ResNeXt with CIFAR10.ipynb** Shows the training process and results of ResNeXt et SE-ResneXt models on CIFAR-10
- **InceptionV3 with CIFAR10-100.ipynb** Shows the training process and results of InceptionV3 et SE-InceptionV3 models on CIFAR-10
### 2. CIFAR 100
- **ResNet with CIFAR100.ipynb** Shows the training process and results of ResNet et SE-Resnet models on CIFAR-100
- **ResNeXt with CIFAR100.ipynb** Shows the training process and results of ResNeXt et SE-ResneXt models on CIFAR-100
- **InceptionV3 with CIFAR10-100.ipynb** Shows the training process and results of InceptionV3 et SE-InceptionV3 models on CIFAR-100
### 3. Tiny ImageNet
- **ResNet18 with tinyImageNet.ipynb** Shows the training process and results of ResNet-18 et SE-Resnet-18 models on Tiny ImageNet with and without data augmentation
- **ResNet34 with tinyImageNet.ipynb** Shows the training process and results of ResNet-34 et SE-Resnet-34 models on Tiny ImageNet with and without data augmentation
- **ResNet50 with tinyImageNet.ipynb** Shows the training process and results of ResNet-50 et SE-Resnet-50 models on Tiny ImageNet with and without data augmentation
### 4. Other tests on SE-block parameters
- **analysis_ablation.ipynb** shows the ablation study tests (Different SE block integrations)
- **analysis_activation.ipynb** shows the different activation distributions accross channels for all stages and block ids
- **analysis_inference.ipynb** shows the inference speed performance of different ResNet et SE-ResNet models with Tiny ImageNet
- **analysis_ratio.ipynb** shows the effect of the ratio parameter on the accuracies
- **analysis_stage.ipynb** shows the impact of each stage where SE blocks are added.

### ./utils python files

Description is also in the ./utils directory 

- **custom_resnet.py**: builds custom ResNet models with the specified input, output sizes, stages, block multiplicity, and kernel sizes.
- **custom_ResNeXt.py**: builds custom ResNeXt models
- **SE_resnet.py**: build custom SE-ResNet models with the specified input, output sizes, stages, block multiplicity, and kernel sizes.
- **SE_ResNeXt.py**: build custom SE-ResNeXt models
- **add_SE.py**: adds SE blocks to any models by entering a list of layers where the SE blocks go.
- **ablation_resnet.py**: builds custom SE-Resnet models with different SE-block integration methods: Standard, POST, PREO and Identity
- **evaluate_model.py**: evaluate model accuracy with Top-n accuray parameters
- **train_xxx.py**: all the training functions for the CIFAR, Tiny ImageNet and ratio tests on different models.

### ./img folder
Contains graphs of the models of the ablation test and stages test
### ./figures folder
Contains miscellaneous figures such as the GCP screenshots

# Download the data sets
- **CIFAR-10-100** https://www.cs.toronto.edu/~kriz/cifar.html or use the Tensorflow included functions.
- **Tiny ImageNet** http://cs231n.stanford.edu/tiny-imagenet-200.zip

Linux commands:
```
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
rm tiny-imagenet-200.zip
```
# Organization of this directory

```
/
├── analysis_ablation.ipynb
├── analysis_activation.ipynb
├── analysis_inference.ipynb
├── analysis_ratio.ipynb
├── analysis_stage.ipynb
├── figures/
│   ├── inceptionv3.png
│   ├── screen_GCP_1.PNG
│   ├── screen_GCP_2.PNG
│   └── screen_GCP_3.PNG
├── img/
│   ├── ablation_identity.png
│   ├── ablation_post.png
│   ├── ablation_pre.png
│   ├── ablation_standard.png
│   ├── stage_2.png
│   ├── stage_3.png
│   └── stage_4.png
├── InceptionV3 with CIFAR10-100.ipynb
├── Load and Test Models.ipynb
├── README.md
├── ResNet with CIFAR10.ipynb
├── ResNet with CIFAR100.ipynb
├── ResNet18 with tinyImageNet.ipynb
├── ResNet34 with tinyImageNet.ipynb
├── ResNet50 with tinyImageNet.ipynb
├── ResNeXt with CIFAR10.ipynb
├── ResNeXt with CIFAR100.ipynb
├── Untitled.ipynb
├── utils/
│   ├── ablation_resnet.py
│   ├── add_SE.py
│   ├── custom_resnet.py
│   ├── custom_ResNeXt.py
│   ├── evaluate_model.py
│   ├── README.md
│   ├── SE_inception_v3.py
│   ├── SE_resnet.py
│   ├── SE_ResNeXt.py
│   ├── tiny_imageNet.py
│   ├── train_CIFAR_all.py
│   ├── train_CIFAR_ResNet.py
│   ├── train_CIFAR_ResNeXt.py
│   ├── train_ratio.py
│   └── train_TinyImageNet_SE_ResNet.py
└── Wael InceptionV3.ipynb
```


