"""
Author: Antonin Vidon

This files defines functions to build custom ResNeXt models.

"""

from tensorflow.keras.layers import MaxPool2D, GlobalAvgPool2D,Add, ReLU, Dense,Input, Conv2D, BatchNormalization, Concatenate
from tensorflow.keras import Model
import tensorflow as tf

# stages of convolution go from 1 to 5 as in https://arxiv.org/pdf/1611.05431.pdf
# grouped convolutions are only present in stages 2 to 5

def custom_ResNeXt(ResNeXt_model=[],custom_input=(224,224,3),n_classes=1000,cardinality=32,model_name='custom_ResNeXt'):

    # default model is resnet50
    if len(ResNeXt_model)==0:
        model_name='custom_ResNeXt50_32x4d'
        # first array of ResNeXt defines the parameters for conv1 and maxpool in conv2
        ResNeXt_model= [[64,7,2],[3,2]], [[1,3,1],[128,128,256],3], [[1,3,1],[256,256,512],4], [[1,3,1],[512,512,1024],6], [[1,3,1],[1024,1024,2048],3]
        

    #layers with 2dconv, batch norm, and relu activation
    def conv_and_activation_layer(x, filters, kernel_size, strides=1,name=''):
            x = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = 'same',name = name+'_conv')(x)
            x = BatchNormalization(name=name+'_bn')(x)
            x = ReLU(name=name+'_relu')(x)
            return x

    def transform_layer(x, filters, kernel_sizes, strides, name = ''):
        # why strides is plural here ?
        x = conv_and_activation_layer(x, filters=filters[0], kernel_size = kernel_sizes[0], strides = strides[0], name = name + '_caal_1')
        x = conv_and_activation_layer(x, filters=filters[1], kernel_size = kernel_sizes[1], strides = strides[1], name = name + '_caal_2')
        return x

    def split_layer(x, filters, kernel_sizes, strides, name = ''):
        all_splits = list()
        for i in range(cardinality) :
            splits = transform_layer(x, filters = filters, kernel_sizes = kernel_sizes, strides = strides, name = name + '_split' + str(i + 1))
            all_splits.append(splits)
        concatenated = Concatenate(axis = 3, name = name + '_concat')(all_splits)
        return concatenated

    def transition_layer(x, filters, kernel_size, strides, name = ''):
        # last layer of grouped ResNeXt block, there is no activation
        x = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = 'same',name = name+'transition_conv')(x)
        x = BatchNormalization(name=name+'_bn')(x)
        return x

    input = Input(custom_input)
    
    # get parameters for conv1 and maxpool of conv2
    conv_pool_params = ResNeXt_model[0]

    x = conv_and_activation_layer(input, filters=conv_pool_params[0][0], kernel_size=conv_pool_params[0][1], strides=conv_pool_params[0][2], name = 'conv1')
    x = MaxPool2D(pool_size = conv_pool_params[1][0], strides = conv_pool_params[1][1], padding = 'same',name='conv2_pool')(x)

    stage_num = 2

    for stage in ResNeXt_model[1:]:

        for indexblock in range(stage[-1]):
            
            skip_x = x # memorizing input to add it back to the result of the transformation

            # Downsampling of conv3, 4, and 5 is done by stride-2 convolutions in the 3×3 layer of the first block in each stage [1,2,1]
            if (stage_num == 2 or indexblock != 0):
                # if we are not in one of the first block of conv3, conv4 or conv5
                # then the stride of the 3x3 filter in the middle of the block is 1
                strides = [1,1,1]
            else:
                # if we are in one of the first block of conv3, conv4 or conv5
                # then the stride of the 3x3 filter in the middle of the block is 2
                strides = [1,2,1]
            
            x = split_layer(x, filters = [stage[1][0] // cardinality, stage[1][1] // cardinality], kernel_sizes = stage[0][0:2], strides = strides[0:2], 
                        name = 'conv' + str(stage_num ) + '_block' + str(indexblock + 1))
            x = transition_layer(x, filters = stage[1][2], kernel_size = stage[0][2],  strides = strides[2], 
                        name = 'conv' + str(stage_num) + '_block' + str(indexblock + 1))

            if indexblock  == 0: # otherwise skip and add directly
                skip_x = Conv2D(filters = stage[1][2], kernel_size = stage[0][2], strides = strides[1], padding = 'same', # stride is 1 for conv2 (maxpool right before and 2 for conv3,4,5)
                        name = 'conv' + str(stage_num ) + '_block' + str(indexblock + 1) + '_skip_conv')(skip_x) 
                skip_x = BatchNormalization(name='conv' + str(stage_num ) + '_block' + str(indexblock + 1) + 'skip_bn')(skip_x)
             
            x = Add(name='conv' + str(stage_num ) + '_block' + str(indexblock + 1) + '_add')([skip_x,x])
            x = tf.nn.relu(x)

        stage_num += 1

    x = GlobalAvgPool2D(name='pool_out')(x)
    output = Dense(n_classes, activation ='softmax',name='output_layer')(x)

    custom_model = Model(inputs=input, outputs=output)
    custom_model._name=model_name

    return custom_model

# prints the resnet layer descriptions
def custom_ResNeXt_summary(res_model=[]):
    model=custom_ResNeXt(res_model)
    model.summary()

# builds ResNeXt50
# https://arxiv.org/pdf/1611.05431.pdf
def custom_ResNeXt50(input_shape=(224,224,3),n_classes=1000,cardinality=32,model_name='custom_ResNext50_32x4d'):
    return custom_ResNeXt([],input_shape,n_classes,cardinality=32,model_name=model_name)

# builds ResNeXt29
# https://arxiv.org/pdf/1611.05431.pdf
def custom_ResNeXt29(input_shape=(224,224,3),n_classes=1000,cardinality=8,model_name='custom_ResNext29_8x64d'):
    return  custom_ResNeXt([[[64,3,2],[3,2]], [[1,3,1],[512,512,256],3], [[1,3,1],[1024,1024,512],3], [[1,3,1],[2048,2048,1024],3]],
                           input_shape,n_classes,cardinality=8,model_name=model_name)