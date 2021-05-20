import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models

"""
creates a 3*3 conv with given filters and stride
"""
def conv3x3(out_planes, stride=1):
    return layers.Conv2D(kernel_size=(3,3), filters=out_planes, strides=stride, padding="same",
                       use_bias=False)

"""
Creates a residual block with two 3*3 conv's
in paper it's represented by RB block
"""
basicblock_expansion = 1
def basic_block(x_in, planes, stride=1, downsample=None, no_relu=False):
    residual = x_in

    x = conv3x3(planes, stride)(x_in)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = conv3x3(planes,)(x)
    x = layers.BatchNormalization()(x)

    if downsample is not None:
        residual = downsample

    # x += residual
    x = layers.Add()([x, residual])

    if not no_relu:
        x = layers.Activation("relu")(x)

    return x

"""
creates a bottleneck block of 1*1 -> 3*3 -> 1*1
"""
bottleneck_expansion = 2
def bottleneck_block(x_in, planes, stride=1, downsample=None, no_relu=True):
    residual = x_in

    x = layers.Conv2D(filters=planes, kernel_size=(1,1), use_bias=False)(x_in)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters=planes, kernel_size=(3,3), strides=stride, padding="same",use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters=planes* bottleneck_expansion, kernel_size=(1,1), use_bias=False)(x)
    x= layers.BatchNormalization()(x)

    if downsample is not None:
        residual = downsample

    # x += residual
    x = layers.Add()([x, residual])

    if not no_relu:
        x = layers.Activation("relu")(x)

    return  x

# Deep Aggregation Pyramid Pooling Module
def DAPPPM(x_in, branch_planes, outplanes):
    input_shape = tf.keras.backend.int_shape(x_in)
    height = input_shape[1]
    width = input_shape[2]
    # Average pooling kernel size
    kernal_sizes_height = [5, 9, 17, height]
    kernal_sizes_width =  [5, 9, 17, width]
    # Average pooling strides size
    stride_sizes_height = [2, 4, 8, height]
    stride_sizes_width =  [2, 4, 8, width]
    x_list = []

    # y1
    scale0 = layers.BatchNormalization()(x_in)
    scale0 = layers.Activation("relu")(scale0)
    scale0 = layers.Conv2D(branch_planes, kernel_size=(1,1), use_bias=False, )(scale0)
    x_list.append(scale0)

    for i in range( len(kernal_sizes_height)):
        # first apply average pooling
        temp = layers.AveragePooling2D(pool_size=(kernal_sizes_height[i],kernal_sizes_width[i]),
                                       strides=(stride_sizes_height[i],stride_sizes_width[i]),
                                       padding="same")(x_in)
        temp = layers.BatchNormalization()(temp)
        temp = layers.Activation("relu")(temp)
        # then apply 1*1 conv
        temp = layers.Conv2D(branch_planes, kernel_size=(1, 1), use_bias=False, )(temp)
        # then resize using bilinear
        temp = tf.image.resize(temp, size=(height,width), )
        # add current and previous layer output
        temp = layers.Add()([temp, x_list[i]])
        temp = layers.BatchNormalization()(temp)
        temp = layers.Activation("relu")(temp)
        # at the end apply 3*3 conv
        temp = layers.Conv2D(branch_planes, kernel_size=(3, 3), use_bias=False, padding="same")(temp)
        # y[i+1]
        x_list.append(temp)

    # concatenate all
    combined = layers.concatenate(x_list, axis=-1)

    combined = layers.BatchNormalization()(combined)
    combined = layers.Activation("relu")(combined)
    combined = layers.Conv2D(outplanes, kernel_size=(1, 1), use_bias=False, )(combined)

    shortcut = layers.BatchNormalization()(x_in)
    shortcut = layers.Activation("relu")(shortcut)
    shortcut = layers.Conv2D(outplanes, kernel_size=(1, 1), use_bias=False, )(shortcut)

    # final = combined + shortcut
    final = layers.Add()([combined, shortcut])

    return final

"""
Segmentation head 
3*3 -> 1*1 -> rescale
"""
def segmentation_head(x_in, interplanes, outplanes, scale_factor=None):
    x = layers.BatchNormalization()(x_in)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(interplanes, kernel_size=(3, 3), use_bias=False, padding="same")(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(outplanes, kernel_size=(1, 1), use_bias=range, padding="valid")(x)

    if scale_factor is not None:
        input_shape = tf.keras.backend.int_shape(x)
        height2 = input_shape[1] * scale_factor
        width2 = input_shape[2] * scale_factor
        x = tf.image.resize(x, size =(height2, width2))

    return x

"""
apply multiple RB or RBB blocks.
x_in: input tensor
block: block to apply it can be RB or RBB
inplanes: input tensor channes
planes: output tensor channels
blocks_num: number of time block to applied
stride: stride
expansion: expand last dimension
"""
def make_layer(x_in, block, inplanes, planes, blocks_num, stride=1, expansion=1):
    downsample = None
    if stride != 1 or inplanes != planes * expansion:
        downsample = layers.Conv2D(((planes * expansion)), kernel_size=(1, 1),strides=stride, use_bias=False)(x_in)
        downsample = layers.BatchNormalization()(downsample)
        downsample = layers.Activation("relu")(downsample)

    x = block(x_in, planes, stride, downsample)
    for i in range(1, blocks_num):
        if i == (blocks_num - 1):
            x = block(x, planes, stride=1, no_relu=True)
        else:
            x = block(x, planes, stride=1, no_relu=False)

    return x


"""
ddrnet 23 slim
input_shape : shape of input data
layers_arg : how many times each Rb block is repeated
num_classes: output classes
planes: filter size kept throughout model
spp_planes: DAPPM block output dimensions
head_planes: segmentation head dimensions
scale_factor: scale output factor
augment: whether auxiliary loss is added or not
"""
def ddrnet_23_slim(input_shape=[1024,2048,3], layers_arg=[2, 2, 2, 2], num_classes=19, planes=32, spp_planes=128,
                   head_planes=64, scale_factor=8,augment=False):

    x_in = layers.Input(input_shape)

    highres_planes = planes * 2
    input_shape = tf.keras.backend.int_shape(x_in)
    height_output = input_shape[1] // 8
    width_output = input_shape[2] // 8

    layers_inside = []

    # 1 -> 1/2 first conv layer
    x = layers.Conv2D(planes, kernel_size=(3, 3),strides=2, padding='same')(x_in)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    # 1/2 -> 1/4 second conv layer
    x = layers.Conv2D(planes, kernel_size=(3, 3), strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # layer 1
    # 1/4 -> 1/4 first basic residual block not mentioned in the image
    x = make_layer(x, basic_block, planes, planes, layers_arg[0], expansion=basicblock_expansion)
    layers_inside.append(x)

    # layer 2
    # 2 High :: 1/4 -> 1/8 storing results at index:1
    x = layers.Activation("relu")(x)
    x = make_layer(x, basic_block, planes, planes*2, layers_arg[1], stride=2, expansion=basicblock_expansion)
    layers_inside.append(x)

    """
    For next layers 
    x:  low branch
    x_: high branch
    """

    # layer 3
    # 3 Low :: 1/8 -> 1/16 storing results at index:2
    x = layers.Activation("relu")(x)
    x = make_layer(x, basic_block, planes*2, planes*4, layers_arg[2], stride=2, expansion=basicblock_expansion)
    layers_inside.append(x)
    # 3 High :: 1/8 -> 1/8 retrieving from index:1
    x_ = layers.Activation("relu")(layers_inside[1])
    x_ = make_layer(x_, basic_block, planes*2, highres_planes, 2, expansion=basicblock_expansion)

    # Fusion 1
    # x -> 1/16 to 1/8, x_ -> 1/8 to 1/16
    # High to Low
    x_temp = layers.Activation("relu")(x_)
    x_temp =  layers.Conv2D(planes*4, kernel_size=(3, 3), strides=2, padding='same', use_bias=False)(x_temp)
    x_temp = layers.BatchNormalization()(x_temp)
    x = layers.Add()([x, x_temp])
    # Low to High
    x_temp = layers.Activation("relu")(layers_inside[2])
    x_temp = layers.Conv2D(highres_planes, kernel_size=(1,1), use_bias=False)(x_temp)
    x_temp = layers.BatchNormalization()(x_temp)
    x_temp = tf.image.resize(x_temp, (height_output, width_output)) # 1/16 -> 1/8
    x_ = layers.Add()([x_, x_temp]) # next high branch input, 1/8

    if augment:
        temp_output = x_  # Auxiliary loss from high branch

    # layer 4
    # 4 Low :: 1/16 -> 1/32 storing results at index:3
    x = layers.Activation("relu")(x)
    x = make_layer(x, basic_block, planes * 4, planes * 8, layers_arg[3], stride=2, expansion=basicblock_expansion)
    layers_inside.append(x)
    # 4 High :: 1/8 -> 1/8
    x_ = layers.Activation("relu")(x_)
    x_ = make_layer(x_, basic_block, highres_planes, highres_planes, 2, expansion=basicblock_expansion)

    # Fusion 2 :: x_ -> 1/32 to 1/8, x -> 1/8 to 1/32 using two conv's
    # High to low
    x_temp = layers.Activation("relu")(x_)
    x_temp = layers.Conv2D(planes * 4, kernel_size=(3, 3), strides=2, padding='same', use_bias=False)(x_temp)
    x_temp = layers.BatchNormalization()(x_temp)
    x_temp = layers.Activation("relu")(x_temp)
    x_temp = layers.Conv2D(planes * 8, kernel_size=(3, 3), strides=2, padding='same', use_bias=False)(x_temp)
    x_temp = layers.BatchNormalization()(x_temp)
    x = layers.Add()([x, x_temp])
    # Low to High
    x_temp = layers.Activation("relu")(layers_inside[3])
    x_temp = layers.Conv2D(highres_planes, kernel_size=(1, 1), use_bias=False)(x_temp)
    x_temp = layers.BatchNormalization()(x_temp)
    x_temp = tf.image.resize(x_temp, (height_output, width_output))
    x_ = layers.Add()([x_, x_temp])

    # layer 5
    # 5 High :: 1/8 -> 1/8
    x_ = layers.Activation("relu")(x_)
    x_ = make_layer(x_, bottleneck_block, highres_planes, highres_planes, 1, expansion=bottleneck_expansion)
    x = layers.Activation("relu")(x)
    # 5 Low :: 1/32 -> 1/64
    x = make_layer(x, bottleneck_block,  planes * 8, planes * 8, 1, stride=2, expansion=bottleneck_expansion)

    # Deep Aggregation Pyramid Pooling Module
    x = DAPPPM(x, spp_planes, planes * 4)

    # resize from 1/64 to 1/8
    x = tf.image.resize(x, (height_output, width_output))

    x_ = layers.Add()([x, x_])

    x_ = segmentation_head( (x_), head_planes, num_classes, scale_factor)

    # apply softmax at the output layer
    x_ = tf.nn.softmax(x_)

    if augment:
        x_extra = segmentation_head( temp_output, head_planes, num_classes, scale_factor) # without scaling
        x_extra = tf.nn.softmax(x_extra)
        model_output = [x_, x_extra]
    else:
        model_output = x_

    model = models.Model(inputs=[x_in], outputs=[model_output])

    # set weight initializers
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel_initializer = tf.keras.initializers.he_normal()
        if hasattr(layer, 'depthwise_initializer'):
            layer.depthwise_initializer = tf.keras.initializers.he_normal()

    return model

if __name__ == "__main__":
    """## Model Compilation"""
    INPUT_SHAPE = [1024, 2048, 3]
    OUTPUT_CHANNELS = 19
    with tf.device("cpu:0"):
        # create model
        ddrnet_model = ddrnet_23_slim( num_classes=OUTPUT_CHANNELS, input_shape =INPUT_SHAPE, )
        optimizer = tf.keras.optimizers.SGD(momentum=0.9, lr=0.045)
        # compile model
        ddrnet_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), optimizer=optimizer,
                          metrics=['accuracy'])
        # show model summary in output
        ddrnet_model.summary()
        # save model architecture as png
        tf.keras.utils.plot_model(ddrnet_model, show_layer_names=True, show_shapes=True)
        # save model
        ddrnet_model.save("temp.hdf5")
