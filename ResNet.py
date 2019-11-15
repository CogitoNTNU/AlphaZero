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
from keras.initializers import TruncatedNormal


class ResNet:
    @staticmethod
    def residual_block(data,
                       filters,
                       strides,
                       chan_dim,
                       red=False,
                       reg=0.0001,
                       bn_eps=2e-5,
                       bn_mom=0.9,
                       use_bias=True):
        """
        Residual block based on the AlphaZero paper
        :param data: Input data
        :param filters: Number of filter for convolution layer
        :param strides: Stride for convolution layer
        :param chan_dim: Channel dimension
        :param red: Whether to reduce the spatial size
        :param reg: Kernel regularization
        :param bn_eps: Small float added to avoid divinding by zero
        :param bn_mom: Momentum for the moving axis
        :param use_bias: If use_bias is True, a bias vector is created and added to the outputs
        :return: Residual block
        """
        # the shortcut branch of the ResNet module should be
        # initialize as the input (identity) data
        shortcut = data

        conv1 = Conv2D(filters=int(filters),
                       kernel_size=(3, 3),
                       padding="same",
                       use_bias=True,
                       kernel_initializer=TruncatedNormal(stddev=0.05),
                       kernel_regularizer=l2(reg)
                       )(data)
        # the first block of the ResNet module are the 1x1 CONVs
        bn1 = BatchNormalization(axis=chan_dim,
                                 epsilon=bn_eps,
                                 momentum=bn_mom)(conv1)
        act1 = Activation("relu")(bn1)

        conv2 = Conv2D(
            filters=int(filters),
            kernel_size=(3, 3),
            strides=strides,
            padding="same",
            use_bias=True,
            kernel_initializer=TruncatedNormal(stddev=0.05),
            kernel_regularizer=l2(reg)
        )(act1)
        # The second block of the ResNet module are the 3x3 CONVs
        bn2 = BatchNormalization(axis=chan_dim, epsilon=bn_eps, momentum=bn_mom)(conv2)


        # if we are to reduce the spatial size, apply a CONV layer to the shortcut
        if red:
            shortcut = Conv2D(
                filters=filters,
                kernel_size=(1, 1),
                strides=strides,
                use_bias=True,
                kernel_initializer=TruncatedNormal(stddev=0.05),
                kernel_regularizer=l2(reg)
            )(act1)

        # Add together the shortcut and the final convolutional layer
        x = add([bn2, shortcut])

        act2 = Activation("relu")(x)

        # Return the addition as the output of the ResNet module
        return act2

    @staticmethod
    def policy_head(data,
                    chan_dim,
                    policy_output_dim,
                    bn_eps,
                    bn_mom):
        """
        Policy head gives out the predictions for the possible actions
        :param data: Input tensor
        :param chan_dim: Channel dimension
        :param policy_output_dim: Dimension of policy output
        :param bn_eps: Small float added to avoid divinding by zero
        :param bn_mom: Momentum for the moving axis
        :return: Matrix of percentages for different moves
        """
        conv1 = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            use_bias=True,
            padding="same",
            kernel_initializer=TruncatedNormal(stddev=0.05)
        )(data)
        bn1 = BatchNormalization(
            axis=chan_dim,
            epsilon=bn_eps,
            momentum=bn_mom)(conv1)
        act1 = Activation("relu")(bn1)
        x = Flatten()(act1)
        dn1 = Dense(
            policy_output_dim,
            activation='linear')(x)
        return dn1

    @staticmethod
    def value_head(data,
                   chan_dim,
                   bn_eps,
                   bn_mom):
        """
        Value head: gives out the state of the board
        :param data: Input tensor
        :param chan_dim: Channel dimension
        :param bn_eps: Small float added to avoid divinding by zero
        :param bn_mom: Momentum for the moving axis
        :return: An integer between [-1,1] with the state of the game
        """

        conv1 = Conv2D(
            filters=32,
            kernel_size=(1, 1),
            strides=(1, 1),
            use_bias=True,
            padding="same",
            kernel_initializer=TruncatedNormal(stddev=0.05),
        )(data)
        bn1 = BatchNormalization(
            axis=chan_dim,
            epsilon=bn_eps,
            momentum=bn_mom)(conv1)
        act1 = Activation("relu")(bn1)
        x = Flatten()(act1)
        dn1 = Dense(
            256,
            activation='relu',
            kernel_initializer=TruncatedNormal(stddev=0.05))(x)
        dn2 = Dense(
            1,
            activation='tanh'
        )(dn1)
        return dn2

    @staticmethod
    def build(height,
              width,
              depth,
              filters,
              policy_output_dim,
              reg=0.0001,
              bn_eps=2e-5,
              bn_mom=0.9,
              num_res_blocks=2,
              use_bias=True):
        """
        Build method for ResNet
        :param height: Height of input
        :param width: Width of input
        :param depth: Channel dimension (Depth of input)
        :param filters: Number of filters
        :param policy_output_dim: The number of possible moves (Classes)
        :param reg: Kernel regularization
        :param bn_eps: Small float added to avoid divinding by zero
        :param bn_mom: Momentum for the moving axis
        :param num_res_blocks: Number of residual block
        :param use_bias: If use_bias is True, a bias vector is created and added to the outputs
        :return: ResNet model
        """
        # Initialize the input shape to be "channels last" and the
        # channels dimension itself
        inputShape = (height, width, depth)
        chan_dim = -1

        # Set the input and apply BN
        input_data = Input(shape=inputShape)

        # Apply CONV => BN => ACT => POOL to reduce spatial size
        x = Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            use_bias=True,
            padding="same",
            kernel_initializer=TruncatedNormal(stddev=0.05),
            kernel_regularizer=l2(reg)
        )(input_data)
        x = BatchNormalization(
            axis=chan_dim,
            epsilon=bn_eps,
            momentum=bn_mom
        )(x)
        x = Activation("relu")(x)

        # Add num_res_blocks to the model
        for _ in range(num_res_blocks):
            x = ResNet.residual_block(
                data=x,
                filters=filters,
                strides=(1, 1),
                chan_dim=chan_dim,
                red=False
            )

        # Policy head
        pol_head = ResNet.policy_head(x,
                                      chan_dim=chan_dim,
                                      policy_output_dim=policy_output_dim,
                                      bn_eps=bn_eps,
                                      bn_mom=bn_mom)

        # Value head
        val_head = ResNet.value_head(x,
                                     chan_dim=chan_dim,
                                     bn_eps=bn_eps,
                                     bn_mom=bn_mom)

        # create the model
        model = Model(input_data, [pol_head, val_head], name="resnet")

        # return the constructed network architecture
        return model

    def get_resnet_4_in_a_row(self):
        """
        Method for return the model for 4 in a row
        :return: ResNet model for 4 in a row
        """
        return self.build(height=6,
                          width=7,
                          depth=2,
                          filters=256,
                          policy_output_dim=7)

    def get_resnet_tictactoe(self):
        """
        Method for returning model for tictactoe
        :return: ResNet model for tictactoe.
        """
        return self.build(height=3,
                          width=3,
                          depth=2,
                          filters=256,
                          policy_output_dim=9)



