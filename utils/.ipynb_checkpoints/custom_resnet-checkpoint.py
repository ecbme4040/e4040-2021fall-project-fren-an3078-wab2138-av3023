"""
Author: Anh-Vu Nguyen

This files defines functions to build custom resnet models.

"""
from tensorflow.keras.layers import MaxPool2D, GlobalAvgPool2D,Add, ReLU, Dense,Input, Conv2D, BatchNormalization
from tensorflow.keras import Model
from tensorflow import keras

###### THIS IS OUR CUSTOM RESNET MODEL GENERATOR #####################
# Inputs:
# - res_model: list of resnet blocks parmater (eg: [1,3,1],[64,64,256],3] first resnet block of resnet50)  
# containing list of kernel sizes ([1,3,1]), filter sizes ([64,64,256]) ,
# and multiplicity of resnet blocks (3)
# - custom_input: input shape of model 
# - n_classes: number of classes
# - model_name: model name
# - debug: prints layers shape if true
#
#  Defaults parameters correspond to resnet50
#
#
# Output : tensorflow model if the described resnet model
######################################################################



def custom_resnet(res_model=[],custom_input=(224,224,3),n_classes=1000,model_name='custom_resnet',debug=False):

    # default model is resnet50
    if len(res_model)==0:
        model_name='custom_resnet50'
        
        #resnet50 parameters
        #https://pytorch.org/assets/images/resnet.png
        res_model=[[1,3,1],[64,64,256],3],[[1,3,1],[128,128,512],4],[[1,3,1],[256,256,1024],6],[[1,3,1],[512,512,2048],3]

    #layers with 2dconv, batch norm, and relu activation
    def conv_and_activation_block(x, filters, kernel_size, strides=1,name=''):
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding = 'same',name=name+'_conv')(x)
        x = BatchNormalization(name=name+'_bn')(x)
        x = ReLU(name=name+'_relu')(x)
        return x
    
     #block with the 2Dconv instead of the skip connection
    def first_resnet_block(input_x, filters,kernels, big_stride,name): 
        
        #sequence with multiple conv
        
        #first conv has can have a stride 2
        x = conv_and_activation_block(input_x, filters=filters[0], kernel_size=kernels[0], strides=big_stride,name=name+'_1')  
        
        #other conv layers have stride 1
        for i in range(len(filters[1:-1])): 
            x = conv_and_activation_block(x, filters=filters[1+i], kernel_size=kernels[i+1], strides=1,name=name+'_'+str(i+2)) 
            
        #last conv layers not followed by activation
        x = Conv2D(filters=filters[-1], kernel_size=kernels[-1], strides=1,padding = 'same',name=name+'_'+str(len(filters))+'_conv')(x)
        if debug:
            print('first block: ',x.shape)
        x = BatchNormalization(name=name+'_'+str(len(filters))+'_bn')(x) 

        #sequence with only one conv  
        shortcut = Conv2D(filters=filters[-1], kernel_size=kernels[-1], strides=big_stride,padding = 'same',name=name+'_0_conv')(input_x)     
        shortcut = BatchNormalization(name=name+'_0_bn')(shortcut) 
        
            
        #add the two sequences
        x = Add(name=name+'_add')([shortcut,x])    #skip connection     
        x = ReLU(name=name+'_relu')(x)    
            
        return x
    
    #block with the skip connection
    def block_with_skip(input_x, filters,kernels,name):
        
        #first conv has can have a stride 2
        x = conv_and_activation_block(input_x, filters=filters[0], kernel_size=kernels[0], strides=1,name=name+'_1')
        #other conv layers have stride 1
        for i in range(len(filters[1:-1])): 
            x = conv_and_activation_block(x, filters=filters[1+i], kernel_size=kernels[i+1], strides=1,name=name+'_'+str(i+2))     
            
        
         #last conv layers not followed by activation
        x = Conv2D(filters=filters[-1], kernel_size=kernels[-1], strides=1,padding = 'same',name=name+'_'+str(len(filters))+'_conv')(x)     
        x = BatchNormalization(name=name+'_'+str(len(filters))+'_bn')(x)
        if debug:
            print('INPUT SHAPE  ',input_x.shape)
            print('conv shape   ',x.shape)       
        
        
        
        #skip node
        x = Add(name=name+'_add')([input_x,x])    
        x = ReLU(name=name+'_relu')(x)
        return x


    
    #resnet block
    def resnet_block(x,params,big_stride,name): #params=[[1,3,1],[64,64,256],3] FOR EXAMPLE
        multiplicity=params[2]-1
        filters=params[1]
        kernels=params[0]

        x = first_resnet_block(x, filters,kernels, big_stride=big_stride,name=name+'_block1')
        for i in range(multiplicity):
            x = block_with_skip(x,filters,kernels,name=name+'_block'+str(i+2))
        return x
    
    
    
    # Building the custom model all paddings to same!
    input = Input(custom_input)

    x = conv_and_activation_block(input, filters=64, kernel_size=7, strides=2,name='conv1')
    if debug:
        print('after first layer   ',x.shape)
    x = MaxPool2D(pool_size = 3, strides =2, padding = 'same',name='pool1')(x)
    if debug:
        print('after maxpool  ',x.shape)
    first_res_block=True
    for i,res_size in enumerate(res_model):
        # The first restnet block have of strides set to 1 because of the previous maxpool!
        if first_res_block:
            first_res_block=False
            x=resnet_block(x,res_size,big_stride=1,name="conv"+str(i+2))#kernels, filters, multiplicity
        else:
            x=resnet_block(x,res_size,big_stride=2,name="conv"+str(i+2))

    x = GlobalAvgPool2D(name='pool_out')(x)

    output = Dense(n_classes, activation ='softmax',name='output_layer')(x)

    custom_model = Model(inputs=input, outputs=output)
    custom_model._name=model_name
    return custom_model

# prints the resnet layer descriptions
def custom_resnet_summary(res_model=[],input_shape=(224,224,3),n_classes=1000,name='custom_resnet',debug=False):
    model=custom_resnet(res_model,input_shape,n_classes,name,debug)
    model.summary()
    
# builds resnet18
# https://pytorch.org/assets/images/resnet.png for more information
def custom_resnet18(input_shape=(224,224,3),n_classes=1000,name='custom_resnet18',debug=False):
    return  custom_resnet([[[3,3],[64,64],2],[[3,3],[128,128],2],[[3,3],[256,256],2],[[3,3],[512,512],2]],input_shape,n_classes,name,debug)

# builds resnet34
# https://pytorch.org/assets/images/resnet.png for more information
def custom_resnet34(input_shape=(224,224,3),n_classes=1000,name='custom_resnet34',debug=False):
    return  custom_resnet([[[3,3],[64,64],3],[[3,3],[128,128],4],[[3,3],[256,256],6],[[3,3],[512,512],3]],input_shape,n_classes,name,debug)

def custom_resnet50(input_shape=(224,224,3),n_classes=1000,name='custom_resnet50',debug=False):
    return  custom_resnet([],input_shape,n_classes,name,False)

# builds resnet101
# https://pytorch.org/assets/images/resnet.png for more information
def custom_resnet101(input_shape=(224,224,3),n_classes=1000,name='custom_resnet101',debug=False):
    return  custom_resnet([[1,3,1],[64,64,256],3],[[1,3,1],[128,128,512],4],[[1,3,1],[256,256,1024],23],[[1,3,1],[512,512,2048],3],input_shape,n_classes,name,debug)

# builds resnet152
# https://pytorch.org/assets/images/resnet.png for more information
def custom_resnet152(input_shape=(224,224,3),n_classes=1000,name='custom_resnet152',debug=False):
    return  custom_resnet([[1,3,1],[64,64,256],3],[[1,3,1],[128,128,512],4],[[1,3,1],[256,256,1024],36],[[1,3,1],[512,512,2048],3],input_shape,n_classes,name,debug)
