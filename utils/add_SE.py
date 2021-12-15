"""
Author: Anh-Vu Nguyen

This files defines functions to build custom resnet models with SE blocks.

"""

import tensorflow as tf

from tensorflow.keras.layers import MaxPool2D, GlobalAvgPool2D, Dense,Input, BatchNormalization
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D,Flatten
from tensorflow.keras import Model

####### add SE layers to model ##########
# input: model
#        l: list l of layer names where SE layers are going to go after
#        ratio : SE ratio
#        modelname: name of new model
#
# Output: modified model with SE layers at specified locations
# See https://stackoverflow.com/questions/49492255/how-to-replace-or-insert-intermediate-layer-in-keras-model to understand the function
##########################################

def add_SE(model, l,ratio,modelname=None):
    se_num =1
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
            lay_name=layer.name
            lay_name=lay_name[0:lay_name.index('block')+6]
            ### adding SE layers
            out_dim=layer.output_shape[-1]
            #squeeze layer
            F_sq = GlobalAveragePooling2D(name=lay_name+'_SE_squeeze')(x) #squeeze
            #Excitation with 2 fully connected layers
            F_ex = Dense(out_dim / ratio,activation='relu', name='SE'+str(se_num)+'_dense_relu')(F_sq)
            F_ex = Dense(out_dim,activation='sigmoid', name='SE'+str(se_num)+'_dense_sig')(F_ex)
            #Output : rescaling
            F_ex = tf.reshape(F_ex, [-1,1,1,out_dim])
            x=keras.layers.multiply([x,F_ex], name='SE'+str(se_num)+'_scaling')
            
            se_num+=1 # for naming
            print('added SE after ',layer.name)
        else:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

        # Renaming model
        if layer.name in model.output_names:
            model_outputs.append(x)
        my_final_model= Model(inputs=model.inputs, outputs=model_outputs)
        if (modelname!=None):
            my_final_model._name = modelname
        else:
            my_final_model._name='SE_'+model._name
    return my_final_model



