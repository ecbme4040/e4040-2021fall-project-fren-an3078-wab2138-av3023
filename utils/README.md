## Utils files description

- **custom_resnet.py**: builds custom ResNet models with the specified input, output sizes, stages, block multiplicity, and kernel sizes.
- **custom_ResNeXt.py**: builds custom ResNeXt models
- **SE_resnet.py**: build custom SE-ResNet models with the specified input, output sizes, stages, block multiplicity, and kernel sizes.
- **SE_ResNeXt.py**: build custom SE-ResNeXt models
- **add_SE.py**: adds SE blocks to any models by entering a list of layers where the SE blocks go.
- **ablation_resnet.py**: builds custom SE-Resnet models with different SE-block integration methods: Standard, POST, PREO and Identity
- **evaluate_model.py**: evaluate model accuracy with Top-n accuray parameters
- **train_xxx.py**: all the training functions for the CIFAR, Tiny ImageNet and ratio tests on different models.
- **tiny_imageNet.py**: gives consistent label order when switching environment or os.
