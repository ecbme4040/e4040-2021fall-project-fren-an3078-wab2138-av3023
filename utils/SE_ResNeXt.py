"""
Author: Anh-Vu Nguyen

This files defines functions to build custom ResNeXt models with SE blocks.

"""

from utils.custom_ResNeXt import custom_ResNeXt,custom_ResNeXt29,custom_ResNeXt50
import tensorflow as tf

from tensorflow.keras.layers import MaxPool2D, GlobalAvgPool2D, Dense,Input, BatchNormalization
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D,Flatten
from tensorflow.keras import Model

####### add SE layers to model ##########
#        input: tensorflow model
#        l: list l of layer names where SE layers are going to go after
#        ratio : SE ratio
#        modelname: name of new model
#
# Output: modified model with SE layers at specified locations
# See https://stackoverflow.com/questions/49492255/how-to-replace-or-insert-intermediate-layer-in-keras-model to understand the function
##########################################

def add_SE(model, l,ratio,modelname=''):
    if modelname=='':
        modelname='SE_'+model.name
    
    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                        {layer_name: [layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
            {model.layers[0].name: model.input})

    # Iterate over all layers after the input
    model_outputs = []
    for layer in model.layers[1:]:

        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux] 
                for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Insert SE layers if name matches the argument list
        if layer.name in l:
            #input of SE
            x = layer(layer_input)
            
            # SE layer name base
            lay_name=layer.name
            lay_name=lay_name[0:lay_name.index('block')+6]
            
            ### adding SE layers
            out_dim=layer.output_shape[-1]
            #squeeze layer
            F_sq = GlobalAveragePooling2D(name=lay_name+'_SE_squeeze')(x) #squeeze
            #Excitation with 2 fully connected layers
            F_ex = Dense(out_dim / ratio,activation='relu', name=lay_name+'_SE_fc1')(F_sq)
            F_ex = Dense(out_dim,activation='sigmoid', name=lay_name+'_SE_fc2')(F_ex)
            #Output : rescaling
            F_ex = tf.reshape(F_ex, [-1,1,1,out_dim])
            x=tf.keras.layers.multiply([x,F_ex], name=lay_name+'_SE_scaling')
            
           
            
        else:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

        # Renaming model
        if layer.name in model.output_names:
            model_outputs.append(x)
        my_final_model= Model(inputs=model.inputs, outputs=model_outputs)

        my_final_model._name='SE_'+model._name
    return my_final_model


######## Returns the SE version of a custom ResNeXt ###########
# input : 
# - add_after_layers: list of layer names where the SE blocks are going to be added after
# - res_model: list of ResNeXt blocks parmater (eg: [1,3,1],[64,64,256],3] first ResNeXt block of ResNeXt50)  
# containing list of kernel sizes ([1,3,1]), filter sizes ([64,64,256]) ,
# and multiplicity of ResNeXt blocks (3)
# - custom_input: input shape of model 
# - n_classes: number of classes
# - SE: Add squeeze and excitation layers if SE!=0 with ratio = SE (https://arxiv.org/abs/1709.01507)
# - model_name: model name
# - debug: prints layers shape if true
#
# Output: Returns the SE version of the specified custom ResNeXt
############################################################## 


# enter any layer where we want to insert SE blocks
def SE_custom_ablation(add_after_layers=[],res_model=[],custom_input=(224,224,3),n_classes=1000,ratio=16,model_name='custom_ResNeXt',debug=False):
    
    base_ResNeXt = custom_ResNeXt(res_model,custom_input,n_classes,model_name,debug)
    return add_SE(base_ResNeXt, add_after_layers ,ratio,modelname=name)


def SE_custom_ResNeXt(res_model=[],custom_input=(224,224,3),n_classes=1000,ratio=16,model_name='custom_ResNeXt',debug=False):
    
    base_ResNeXt = custom_ResNeXt(res_model,custom_input,n_classes,model_name,debug)
    
    l=[l.name for l in base_ResNeXt.layers]
    list_se=[]
    for i,name in enumerate(l):
        if(i+1<len(l) and 'add' in l[i+1]):
            list_se.append(name)
    return add_SE(base_ResNeXt, list_se ,ratio,modelname=name)

# the last 5 functions return the SE versions of the ResNeXt29-50 models
def SE_ResNeXt29(input_shape=(224,224,3),n_classes=1000,ratio=16,name='SE_ResNeXt29'):
    base_ResNeXt = custom_ResNeXt29(input_shape,n_classes,'ResNeXt29')
    l=[l.name for l in base_ResNeXt.layers]
    list_se=[]
    for i,name in enumerate(l):
        if(i+1<len(l) and 'add' in l[i+1]):
            list_se.append(name)
    return add_SE(base_ResNeXt, list_se ,ratio,modelname=name)

# the last 5 functions return the SE versions of the ResNeXt29-50 models
def SE_ResNeXt50(input_shape=(224,224,3),n_classes=1000,ratio=16,name='SE_ResNeXt50'):
    base_ResNeXt = custom_ResNeXt29(input_shape,n_classes,'ResNeXt50')
    l=[l.name for l in base_ResNeXt.layers]
    list_se=[]
    for i,name in enumerate(l):
        if(i+1<len(l) and 'add' in l[i+1]):
            list_se.append(name)
    return add_SE(base_ResNeXt, list_se ,ratio,modelname=name)


