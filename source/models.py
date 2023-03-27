import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
from keras import layers, models, Input
from keras.applications.resnet import ResNet50
import keras.backend as K
import tensorflow as tf
import numpy as np
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Define the input size
INPUT_SIZE = 512

# Bilinear resize factor
RESIZE_FACTOR = 2


def resize_bilinear(layer):
    return tf.image.resize(layer, size=[K.int_shape(layer)[1] * RESIZE_FACTOR,
                                        K.int_shape(layer)[2] * RESIZE_FACTOR])


class EAST:
    def __init__(self):
        input_tensor = Input(shape=(INPUT_SIZE, INPUT_SIZE, 3), name='input_t')
        resnet = ResNet50(input_tensor=input_tensor, weights='imagenet', include_top=False, pooling=None)
        '''
        Extract feature maps from three different resolutions of tensors
        Fusion feature maps for EAST formed of 16, 32, and 64 unit shaped tensors
        Stem feature maps for EAST formed of twice of each fusion map size
        '''

        fusion_tensor_16 = resnet.get_layer('conv5_block3_out').output
        fusion_tensor_32 = resnet.get_layer('conv4_block6_out').output
        fusion_tensor_64 = resnet.get_layer('conv3_block4_out').output
        fusion_tensor_128 = resnet.get_layer('conv2_block2_out').output

        # Stage 1

        stem_tensor_32 = layers.Lambda(resize_bilinear, name='stem_1')(fusion_tensor_16)
        concat_block_1 = K.concatenate([stem_tensor_32, fusion_tensor_32], axis=3)

        conv1x1_block_1 = layers.Conv2D(128, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(1e-5))(
            concat_block_1)
        batchnorm1_block_1 = layers.BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(conv1x1_block_1)
        activation1_block_1 = layers.Activation('relu')(batchnorm1_block_1)

        conv3x3_block1 = layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(1e-5))(
            activation1_block_1)
        batchnorm2_block_1 = layers.BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(conv3x3_block1)
        activation2_block_1 = layers.Activation('relu')(batchnorm2_block_1)

        # Stage 2

        stem_tensor_64 = layers.Lambda(resize_bilinear, name='stem_2')(activation2_block_1)
        concat_block_2 = K.concatenate([stem_tensor_64, fusion_tensor_64], axis=3)

        conv1x1_block_2 = layers.Conv2D(128, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(1e-5))(
            concat_block_2)
        batchnorm1_block_2 = layers.BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(conv1x1_block_2)
        activation1_block_2 = layers.Activation('relu')(batchnorm1_block_2)

        conv3x3_block1 = layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(1e-5))(
            activation1_block_2)
        batchnorm2_block_2 = layers.BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(conv3x3_block1)
        activation2_block_2 = layers.Activation('relu')(batchnorm2_block_2)

        # Stage 3

        stem_tensor_128 = layers.Lambda(resize_bilinear, name='stem_3')(activation2_block_2)
        concat_block_3 = K.concatenate([stem_tensor_128, fusion_tensor_128], axis=3)

        conv1x1_block_3 = layers.Conv2D(128, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(1e-5))(
            concat_block_3)
        batchnorm1_block_3 = layers.BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(conv1x1_block_3)
        activation1_block_3 = layers.Activation('relu')(batchnorm1_block_3)

        conv3x3_block1 = layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(1e-5))(
            activation1_block_3)
        batchnorm2_block_3 = layers.BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(conv3x3_block1)
        activation2_block_3 = layers.Activation('relu')(batchnorm2_block_3)

        # Output

        ultimate_conv = layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(1e-5),
                                      activation='relu')(activation2_block_3)

        score_map = layers.Conv2D(1, (1, 1), activation='sigmoid', name='score_map')(ultimate_conv)

        rbox_geo_map = layers.Conv2D(4, (1, 1), activation='sigmoid', name='rbox_geo_map')(ultimate_conv)
        rbox_geo_map_relative = layers.Lambda(lambda x: x * INPUT_SIZE)(rbox_geo_map)

        rbox_angle_map = layers.Conv2D(1, (1, 1), activation='sigmoid', name='rbox_angle_map')(ultimate_conv)
        rbox_angle_map_radians = layers.Lambda(lambda x: (x - 0.5) * np.pi / 2)(rbox_angle_map)

        rbox_map = K.concatenate([rbox_geo_map_relative, rbox_angle_map_radians], axis=3)

        model = models.Model(inputs=input_tensor, outputs=[score_map, rbox_map])

        self.model = model


if __name__ == '__main__':
    east = EAST()
