# Squeeze-and-Excitation Networks
This report summarizes the findings of the original Squeeze-and-Excitation Networks paper and shows a reproduction of the results with Tensorflow [1]. Convolutional neural networks are widely used for image classification in models such as Resnet models. This report investigates the effectiveness of Squeeze-and-Excitation blocks whose aim is to strengthen the inter-channel relationship by rescaling them. The original paper showed that it consistently increased the classification accuracy with various data sets such as Image Net. This paper is going to show how we were able to increase the accuracy of CNN models with the Squeeze-and-Excitation method. Because of technical barriers, slightly different models and data sets were used. We also tried to analyze the effect of different parameters of  Squeeze-and-Excitation blocks on the models, matching the paper’s findings with varying success.

## Download the models
Models are ordered by test category folders

[Lion drive link](https://drive.google.com/drive/folders/15kpxrvAyOuMmiqZfGMf_imL12vHBefLx?usp=sharing)
(https://drive.google.com/drive/folders/15kpxrvAyOuMmiqZfGMf_imL12vHBefLx?usp=sharing)


## Jupyter noteooks descriptions
### 0. Results overview
- **Load and Test Models.ipynb** loads all the saved models and computes the top-1,3 and 5 accuracy on the associated test data sets.
### 1. CIFAR10 
- **ResNet with CIFAR10.ipynb** Shows the training process and results of ResNet et SE-Resnet models on CIFAR-10
- **ResNeXt with CIFAR10.ipynb** Shows the training process and results of ResNeXt et SE-ResneXt models on CIFAR-10
### 2. CIFAR 100
- **ResNet with CIFAR100.ipynb** Shows the training process and results of ResNet et SE-Resnet models on CIFAR-100
- **ResNeXt with CIFAR100.ipynb** Shows the training process and results of ResNeXt et SE-ResneXt models on CIFAR-100
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

## Utils python files

Description is also in the ./utils directory 

- **custom_resnet.py**: builds custom ResNet models with the specified input, output sizes, stages, block multiplicity, and kernel sizes.
- **custom_ResNeXt.py**: builds custom ResNeXt models
- **SE_resnet.py**: build custom SE-ResNet models with the specified input, output sizes, stages, block multiplicity, and kernel sizes.
- **SE_ResNeXt.py**: build custom SE-ResNeXt models
- **add_SE.py**: adds SE blocks to any models by entering a list of layers where the SE blocks go.
- **ablation_resnet.py**: builds custom SE-Resnet models with different SE-block integration methods: Standard, POST, PREO and Identity
- **evaluate_model.py**: evaluate model accuracy with Top-n accuray parameters
- **train_xxx.py**: all the training functions for the CIFAR, Tiny ImageNet and ratio tests on different models.

# Organization of this directory
To be populated by students, as shown in previous assignments.
Create a directory/file tree

[1] J. Hu, L. Shen, S. Albanie, G. Sun, and E. Wu, Squeeze-and-Excitation Networks. 2019.
