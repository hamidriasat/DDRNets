# ------------------------------------------------------------------------------
# Written by Hamid Ali (hamidriasat@gmail.com)
# ------------------------------------------------------------------------------
import tensorflow.keras.layers as layers


def conv3x3(out_planes, stride=1):
    """
    creates a 3*3 conv with given filters and stride
    :param out_planes:
    :param stride:
    :return:
    """
    return layers.Conv2D(kernel_size=(3, 3), filters=out_planes, strides=stride, padding="same",
                         use_bias=False)


basicblock_expansion = 1
bottleneck_expansion = 2


def basic_block(x_in, planes, stride=1, downsample=None, no_relu=False):
    """
    Creates a residual block with two 3*3 conv's in paper it's represented by RB block
    :param x_in:
    :param planes:
    :param stride:
    :param downsample:
    :param no_relu:
    :return:
    """
    residual = x_in

    x = conv3x3(planes, stride)(x_in)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = conv3x3(planes, )(x)
    x = layers.BatchNormalization()(x)

    if downsample is not None:
        residual = downsample

    # x += residual
    x = layers.Add()([x, residual])

    if not no_relu:
        x = layers.Activation("relu")(x)

    return x


def bottleneck_block(x_in, planes, stride=1, downsample=None, no_relu=True):
    """
    creates a bottleneck block of 1*1 -> 3*3 -> 1*1
    :param x_in:
    :param planes:
    :param stride:
    :param downsample:
    :param no_relu:
    :return:
    """
    residual = x_in

    x = layers.Conv2D(filters=planes, kernel_size=(1, 1), use_bias=False)(x_in)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters=planes, kernel_size=(3, 3), strides=stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters=planes * bottleneck_expansion, kernel_size=(1, 1), use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if downsample is not None:
        residual = downsample

    # x += residual
    x = layers.Add()([x, residual])

    if not no_relu:
        x = layers.Activation("relu")(x)

    return x


def make_layer(x_in, block, inplanes, planes, blocks_num, stride=1, expansion=1):
    """
    apply multiple RB or RBB blocks.
    :param x_in: input tensor
    :param block: block to apply it can be RB or RBB
    :param inplanes: input tensor channes
    :param planes: output tensor channels
    :param blocks_num: number of time block to applied
    :param stride: stride
    :param expansion: expand last dimension
    :return:
    """
    downsample = None
    if stride != 1 or inplanes != planes * expansion:
        downsample = layers.Conv2D(((planes * expansion)), kernel_size=(1, 1), strides=stride, use_bias=False)(x_in)
        downsample = layers.BatchNormalization()(downsample)
        downsample = layers.Activation("relu")(downsample)

    x = block(x_in, planes, stride, downsample)
    for i in range(1, blocks_num):
        if i == (blocks_num - 1):
            x = block(x, planes, stride=1, no_relu=True)
        else:
            x = block(x, planes, stride=1, no_relu=False)

    return x
