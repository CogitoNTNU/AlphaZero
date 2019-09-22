# import the necessary packages
# import the necessary packages
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import add
from keras.regularizers import l2
from keras import backend as K

class ResNet:
    @staticmethod
    def residual_block(data, 
                        filters, 
                        stride, 
                        chan_dim, 
                        red=False, 
                        reg=0.01,
                        bnEps=2e-5, 
                        bnMom=0.9, 
                        use_bias=True):
        # the shortcut branch of the ResNet module should be
        # initialize as the input (identity) data
        shortcut = data
    
        # the first block of the ResNet module are the 1x1 CONVs
        bn1 = BatchNormalization(axis=chan_dim, epsilon=bnEps, momentum=bnMom)(data)
        act1 = Activation("relu")(bn1)
        conv1 = Conv2D(
            filters=int(filters * 0.25),
            kernel_size=(1, 1), 
            use_bias=False, 
            kernel_regularizer=l2(reg)
            )(act1)
        
        # the second block of the ResNet module are the 3x3 CONVs
        bn2 = BatchNormalization(axis=chan_dim, epsilon=bnEps, momentum=bnMom)(conv1)
        act2 = Activation("relu")(bn2)
        conv2 = Conv2D(
            filters=int(filters * 0.25),
            kernel_size=(3, 3), 
            strides=stride, 
            padding="same", 
            use_bias=False, 
            kernel_regularizer=l2(reg)
            )(act2)

        # the third block of the ResNet module is another set of 1x1 CONVs
        bn3 = BatchNormalization(axis=chan_dim, epsilon=bnEps,momentum=bnMom)(conv2)
        act3 = Activation("relu")(bn3)
        conv3 = Conv2D(
            filters=filters, 
            kernel_size=(1, 1), 
            use_bias=False, 
            kernel_regularizer=l2(reg)
            )(act3)

        # if we are to reduce the spatial size, apply a CONV layer to the shortcut
        if red:
            shortcut = Conv2D(
                filters=filters,
                kernel_size=(1, 1), 
                strides=stride, 
                use_bias=False, 
                kernel_regularizer=l2(reg)
                )(act1)

        # add together the shortcut and the final CONV
        x = add([conv3, shortcut])

        # return the addition as the output of the ResNet module
        return x

    @staticmethod
    def policy_head(data, chan_dim, bnEps,bnMom):

        conv1 = Conv2D(
            filters=2,
            kernel_size=(1, 1), 
            strides=(1,1),
            use_bias=False, 
            )(data)
        bn1 = BatchNormalization(
            axis=chan_dim, 
            epsilon=bnEps, 
            momentum=bnMom)(conv1)
        act1 = Activation("relu")(bn1)
        x = Flatten()(act1)
        dn1 = Dense(
            7,
            activation='linear')(x)
        return dn1
    
    @staticmethod
    def value_head(data, 
    chan_dim, 
    bnEps, 
    bnMom):

        conv1 = Conv2D(
            filters=1,
            kernel_size=(1, 1), 
            strides=(1,1),
            use_bias=False, 
            )(data)
        bn1 = BatchNormalization(
            axis=chan_dim, 
            epsilon=bnEps, 
            momentum=bnMom)(conv1)
        act1 = Activation("relu")(bn1)
        x = Flatten()(act1)
        dn1 = Dense(
            256,
            activation='relu')(x)
        dn2 = Dense(
            1,
            activation='tanh'
        )(dn1)
        return dn2

    @staticmethod
    def build(height, width, depth, filters, policy_output_dim, classes, reg=0.01, bnEps=2e-5, bnMom=0.9, num_res_blocks=19,
              use_bias=True):
        # initialize the input shape to be "channels last" and the
        # channels dimension itself
        inputShape = (height, width, depth)
        chan_dim = -1

        # set the input and apply BN
        input = Input(shape=inputShape)

        # apply CONV => BN => ACT => POOL to reduce spatial size
        x = Conv2D(
            filters=filters[0], 
            kernel_size=(3, 3), 
            strides=(1,1),
            use_bias=False,
            padding="same", 
            kernel_regularizer=l2(reg)
            )(input)
        x = BatchNormalization(
            axis=chan_dim, 
            epsilon=bnEps,
            momentum=bnMom
            )(x)
        x = Activation("relu")(x)

        # loop over the number of stages
        for _ in range(num_res_blocks):
            x = ResNet.residual_block(
                data=x,
                filters=256,
                stride=(1,1),
                chan_dim= chan_dim,                
            )

        # apply BN => ACT => POOL
        x = BatchNormalization(axis=chan_dim, epsilon=bnEps,
                momentum=bnMom)(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((8, 8))(x)
        
        # softmax classifier
        x = Flatten()(x)
        x = Dense(classes, kernel_regularizer=l2(reg))(x)
        x = Activation("softmax")(x)

        #Policy head
        pol_head = ResNet.policy_head(x, chan_dim, bnEps, bnMom)

        #Value head
        val_head = ResNet.value_head(x, chan_dim, bnEps, bnMom)
        
        # create the model
        model = Model(input, [pol_head, val_head], name="resnet")
        
        # return the constructed network architecture
        return model
