import tensorflow as tf
from tensorflow.keras.layers import Conv2D, AvgPool2D, MaxPool2D, Activation, BatchNormalization, GlobalAveragePooling2D, Dense, Dropout, Flatten

class inception_v3_custom:
    """Our own implementation for inception_v3. The main thing to be remembered about this model is that it implements new blocks that for a same input 
    perform several convolutions in parallels, with different parameters (kernels/strides/number of channels).
    """
    def __init__(self, aux_classifier=True, se_network=False, data_base='cifar-10', input_shape=None, n_classes=None, ratio = 16, image_shape = 200):
        """
        Parameters
        ----------
        aux_classifier : bool, optional
            implement or not an auxiliary classifier helping reduce overfitting (present in original inception_v3), by default True
        se_network : bool, optional
            implement or not Squeeze-Excitation block after each convolution, by default False
        data_base : str, optional
            'cifar-10', 'cifar-100' or 'imagenet', by default 'cifar-10'
        input_shape : list, optional
            if you use any other databases than CIFAR-10/100 or Imagenet, manually enter the input shape (e.g. (299,299,3)), by default None
        n_classes : int, optional
            if you use any other databases than CIFAR-10/100 or Imagenet, manually enter the number of classes (e.g. 100), by default None
        image_shape : int, optional
            only serves when the image entered is too small, we resize the input images to size image_shape x image_shape, by default 200
        """
        self.aux_classifier = aux_classifier
        self.se_network = se_network
        if se_network:
            self.ratio = ratio
        self.image_shape = tf.cast(image_shape, 'int64')
        self.data_base = data_base


        #we choose the right formats for each database
        if data_base == 'cifar-10':
            self.input_shape = (32,32,3)
            self.n_classes = 10
        elif data_base == 'cifar-100':
            self.input_shape = (32,32,3)
            self.n_classes = 100
        elif data_base == 'imagenet':
            self.input_shape = (299,299,3)
            self.n_classes = 1000
        elif input_shape != None and n_classes != None:
            self.input_shape = input_shape
            self.n_classes = n_classes
        else:
            raise NotImplementedError("Please choose either a valid database for this model ('cifar-10', 'cifar-100' or 'imagenet') or correct shape of inputs/number of classes.")

    @staticmethod
    def SE_block(input, ratio = 16):
        """Squeeze-Excitation block.

        Parameters
        ----------
        input : tensor
        ratio : int, optionnal
            ratio for the Excitation transformation, by default 16
        """
        out_dim = input.shape[-1]
        #squeeze layer
        F_sq = GlobalAveragePooling2D()(input) #squeeze
        #Excitation with 2 fully connected layers
        F_ex = Dense(out_dim / ratio, activation='relu')(F_sq)
        F_ex = Dense(out_dim, activation='sigmoid')(F_ex)
        #Output : rescaling
        F_ex = tf.reshape(F_ex, [-1, 1, 1, out_dim])
        output = tf.keras.layers.multiply([input, F_ex])
        return output


    def Conv2D_Stack(self, input, filters, kernel_size = 1, strides = 1, padding = 'same', activation = 'relu') :
        """A Conv2D layer followed by a Batch normalization, possibly an SE block, and an activation.

        Parameters
        ----------
        input : tensor
        filters : int
        kernel_size : int, optional
            by default 1
        strides : int, optional
            by default 1
        padding : str, optional
            by default 'same'
        activation : str, optional
            by default 'relu'

        Returns
        -------
        output
            
        """
        input_shape = tf.shape(input)
        output = Conv2D(filters=filters, kernel_size= kernel_size, strides = strides, padding = padding, input_shape=input_shape[1:])(input)
        output = BatchNormalization()(output)
        output = Activation(activation)(output)
        return output

    # Inception Block A
    def block_A(self, input, number = 1):
        #number = number of the block A in inception (in total: 3 blocks)

        filters_branch_4 = 64 - (number == 1) * 32 #32 filters for first block A, 64 for the others

        branch_1 = self.Conv2D_Stack(input, filters = 64, kernel_size = (1,1), activation = 'relu')

        branch_2 = self.Conv2D_Stack(input, filters = 48, kernel_size = (1,1), activation = 'relu')
        branch_2 = self.Conv2D_Stack(branch_2, filters = 64, kernel_size = (5,5), activation = 'relu')

        branch_3 = self.Conv2D_Stack(input, filters = 64 , kernel_size = (1,1), activation = 'relu')
        branch_3 = self.Conv2D_Stack(branch_3, filters = 96, kernel_size = (3,3), activation = 'relu')
        branch_3 = self.Conv2D_Stack(branch_3, filters = 96, kernel_size = (3,3), activation = 'relu')

        branch_4 = AvgPool2D(pool_size = (3,3), strides = (1,1), padding = 'same')(input)
        branch_4 = self.Conv2D_Stack(branch_4, filters = filters_branch_4, kernel_size = (1,1), activation = 'relu')

        output = tf.concat(axis=3, values=[branch_1, branch_2, branch_3, branch_4])
        
        if self.se_network:
            output = self.SE_block(output, self.ratio)
        return output
        
    # Inception Block B
    #Note: At each use of Block B, Tensorflow implementaton might take values slightly different for the kernel
    #size, I might take that into consideration later on

    def block_B(self, input, number = 1):

        filters_inter = (number == 1) * 128 + (number in (2,3) ) * 160 + (number == 4) * 192 #Intermediate Conv2D layers of first block B have 128 filters, then 160, 160, 192 for 2nd, 3rd and 4th instance of block B

        branch_1 = self.Conv2D_Stack(input, filters = 192, kernel_size = (1,1), activation = 'relu')

        branch_2 = AvgPool2D(pool_size = (3,3), strides = (1,1), padding = 'same')(input)
        branch_2 = self.Conv2D_Stack(branch_2, filters = 192, kernel_size = (1,1), activation = 'relu')

        branch_3 = self.Conv2D_Stack(input, filters = filters_inter, kernel_size = (1,1), activation = 'relu')
        branch_3 = self.Conv2D_Stack(branch_3, filters = filters_inter, kernel_size = (1,7), activation = 'relu')
        branch_3 = self.Conv2D_Stack(branch_3, filters = 192, kernel_size = (7,1), activation = 'relu')

        branch_4 = self.Conv2D_Stack(input, filters = filters_inter , kernel_size = (1,1), activation = 'relu')
        branch_4 = self.Conv2D_Stack(branch_4, filters = filters_inter, kernel_size = (7,1), activation = 'relu')
        branch_4 = self.Conv2D_Stack(branch_4, filters = filters_inter, kernel_size = (1,7), activation = 'relu')
        branch_4 = self.Conv2D_Stack(branch_4, filters = filters_inter, kernel_size = (7,1), activation = 'relu')
        branch_4 = self.Conv2D_Stack(branch_4, filters = 192, kernel_size = (1,7), activation = 'relu')

        output = tf.concat(axis=3, values=[branch_1, branch_2, branch_3, branch_4])
        
        if self.se_network:
            output = self.SE_block(output, self.ratio)
        return output

    # Inception Block C
    def block_C(self, input):
        #block_C is called 2 times in incetpion, with the exact same parameters

        branch_1 = self.Conv2D_Stack(input, filters = 320, kernel_size = (1,1), activation = 'relu')

        branch_2 = self.Conv2D_Stack(input, filters = 384, kernel_size = (1,1), activation = 'relu')
        branch_2_a = self.Conv2D_Stack(branch_2, filters = 384, kernel_size = (1,3), activation = 'relu')
        branch_2_b = self.Conv2D_Stack(branch_2, filters = 384, kernel_size = (3,1), activation = 'relu')

        branch_3 = self.Conv2D_Stack(input, filters = 448 , kernel_size = (1,1), activation = 'relu')
        branch_3 = self.Conv2D_Stack(branch_3, filters = 384, kernel_size = (3,3), activation = 'relu')
        branch_3_a = self.Conv2D_Stack(branch_3, filters = 384, kernel_size = (3,1), activation = 'relu')
        branch_3_b = self.Conv2D_Stack(branch_3, filters = 384, kernel_size = (1,3), activation = 'relu')

        branch_4 = AvgPool2D(pool_size = (3,3), strides = (1,1), padding = 'same')(input)
        branch_4 = self.Conv2D_Stack(branch_4, filters = 192, kernel_size = (1,1), activation = 'relu')

        output = tf.concat(axis=3, values=[branch_1, branch_2_a, branch_2_b, branch_3_a, branch_3_b, branch_4])

        if self.se_network:
            output = self.SE_block(output, self.ratio)
        return output

    # Inception Reduction A
    def reduction_block_A(self, input):
        branch_1 = MaxPool2D(pool_size = (3,3), strides = (2,2), padding = 'valid')(input)

        branch_2 = self.Conv2D_Stack(input, kernel_size = (3,3), filters = 384, strides = 2, padding = 'valid', activation = 'relu')

        branch_3 = self.Conv2D_Stack(input, filters = 64, kernel_size = (1,1), activation = 'relu')
        branch_3 = self.Conv2D_Stack(branch_3, filters = 96, kernel_size = (3,3), activation = 'relu')
        branch_3 = self.Conv2D_Stack(branch_3, filters = 96, kernel_size = (3,3), strides = 2, padding = 'valid', activation = 'relu')

        output = tf.concat(axis=3, values=[branch_1, branch_2, branch_3])
        
        if self.se_network:
            output = self.SE_block(output, self.ratio)
        return output

    # Inception Reduction B
    def reduction_block_B(self, input):
        branch_1 = MaxPool2D(pool_size = (3,3), strides = (2,2), padding = 'valid')(input)

        branch_2 = self.Conv2D_Stack(input, kernel_size = (1,1), filters = 192, activation = 'relu')
        branch_2 = self.Conv2D_Stack(branch_2, kernel_size = (3,3), filters = 320, strides = 2, padding = 'valid', activation = 'relu')

        branch_3 = self.Conv2D_Stack(input, filters = 192, kernel_size = (1,1), activation = 'relu')
        branch_3 = self.Conv2D_Stack(branch_3, filters = 192, kernel_size = (1,7), activation = 'relu')
        branch_3 = self.Conv2D_Stack(branch_3, filters = 192, kernel_size = (7,1), activation = 'relu')
        branch_3 = self.Conv2D_Stack(branch_3, filters = 192, kernel_size = (3,3), strides = 2, padding = 'valid', activation = 'relu')

        output = tf.concat(axis=3, values=[branch_1, branch_2, branch_3])

        if self.se_network:
            output = self.SE_block(output, self.ratio)
        return output 

    def forward(self, input) :
        #Inception forward

        if self.input_shape[0] < self.image_shape: # if the image is too small (smaller than 100), we'll end up totally erasing the image via the pooling/convolutional 
        #layers, it is crucial to have images large enough
            x = tf.image.resize(input, size = (self.image_shape, self.image_shape))

        x = self.Conv2D_Stack(x, kernel_size = (3,3), filters = 32, strides = 2, padding = 'valid', activation = 'relu')
        x = self.Conv2D_Stack(x, kernel_size = (3,3), filters = 32, strides = 1, padding = 'valid', activation = 'relu')
        x = self.Conv2D_Stack(x, kernel_size = (3,3), filters = 64, strides = 1, padding = 'valid', activation = 'relu')
        x = MaxPool2D(pool_size = (3,3), strides = (2,2), padding = 'valid')(x)

        x = self.Conv2D_Stack(x, kernel_size = (1,1), filters = 80, strides = 1, padding = 'valid', activation = 'relu')
        x = self.Conv2D_Stack(x, kernel_size = (3,3), filters = 192, strides = 1, padding = 'valid', activation = 'relu')
        x = MaxPool2D(pool_size = (3,3), strides = (2,2), padding = 'valid')(x)

        x = self.block_A(x, number = 1)
        x = self.block_A(x, number = 2)
        x = self.block_A(x, number = 3)

        x = self.reduction_block_A(x)

        x = self.block_B(x, number = 1)
        x = self.block_B(x, number = 2)
        x = self.block_B(x, number = 3)
        x = self.block_B(x, number = 4)

        ###auxiliary classifier for better performance (only useful for backward propagation)

        if self.aux_classifier:
            out1 = AvgPool2D(pool_size = (5,5), strides = 3, padding = 'same')(x)
            out1 = self.Conv2D_Stack(out1, filters = 128, kernel_size = 1, strides = 1, padding = 'same')
            out1 = self.Conv2D_Stack(out1, filters = 768, kernel_size = (5,5), strides = 1, padding = 'same')
            out1 = Flatten()(out1)
            initializer = tf.keras.initializers.TruncatedNormal(0.001)
            out1 = Dense(self.n_classes, activation = 'softmax', name = 'auxiliary_classifier', kernel_initializer = initializer)(out1)

        ###
        x = self.reduction_block_B(x)
        x = self.block_C(x)
        x = self.block_C(x)
        x = GlobalAveragePooling2D(name = 'average_pooling')(x)
        x = Dropout(rate = 0.2)(x)
        out2 = Dense(self.n_classes, activation = 'softmax', name = 'prediction_layer')(x)
        if self.aux_classifier:
            return [out1, out2]
        else:
            return out2

    def Model(self, name = "inception_v3_custom"):
        input = tf.keras.layers.Input(shape = tf.convert_to_tensor(self.input_shape))  ####### normally  would be input = tf.keras.layers.Input(shape = (299, 299, 3))
        #to use our model as a classic tf.keras model
        out = self.forward(input)
        return tf.keras.models.Model(input, out, name = name)