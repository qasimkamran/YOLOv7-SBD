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

# Yaml directories
HYP_DIR = os.path.abspath('hyp')
DATA_DIR = os.path.abspath('data')
CFG_DIR = os.path.abspath('cfg')

# Results directories
RESULTS_DIR = os.path.abspath('results')


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

        stem_16 = resnet.get_layer('conv5_block3_out').output
        stem_32 = resnet.get_layer('conv4_block6_out').output
        stem_64 = resnet.get_layer('conv3_block4_out').output
        stem_128 = resnet.get_layer('conv2_block2_out').output

        # Stage 1

        fusion_32 = layers.Lambda(resize_bilinear, name='stem_1')(stem_16)
        concat_block_1 = K.concatenate([fusion_32, stem_32], axis=3)

        conv1x1_block_1 = layers.Conv2D(128, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(1e-5))(
            concat_block_1)
        batchnorm1_block_1 = layers.BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(conv1x1_block_1)
        activation1_block_1 = layers.Activation('relu')(batchnorm1_block_1)

        conv3x3_block1 = layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(1e-5))(
            activation1_block_1)
        batchnorm2_block_1 = layers.BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(conv3x3_block1)
        activation2_block_1 = layers.Activation('relu')(batchnorm2_block_1)

        # Stage 2

        fusion_64 = layers.Lambda(resize_bilinear, name='stem_2')(activation2_block_1)
        concat_block_2 = K.concatenate([fusion_64, stem_64], axis=3)

        conv1x1_block_2 = layers.Conv2D(64, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(1e-5))(
            concat_block_2)
        batchnorm1_block_2 = layers.BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(conv1x1_block_2)
        activation1_block_2 = layers.Activation('relu')(batchnorm1_block_2)

        conv3x3_block1 = layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(1e-5))(
            activation1_block_2)
        batchnorm2_block_2 = layers.BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(conv3x3_block1)
        activation2_block_2 = layers.Activation('relu')(batchnorm2_block_2)

        # Stage 3

        fusion_128 = layers.Lambda(resize_bilinear, name='stem_3')(activation2_block_2)
        concat_block_3 = K.concatenate([fusion_128, stem_128], axis=3)

        conv1x1_block_3 = layers.Conv2D(32, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(1e-5))(
            concat_block_3)
        batchnorm1_block_3 = layers.BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(conv1x1_block_3)
        activation1_block_3 = layers.Activation('relu')(batchnorm1_block_3)

        conv3x3_block1 = layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(1e-5))(
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


class SimpleHTR:
    def __int__(self):
        input_tensor = Input(shape=(INPUT_SIZE, INPUT_SIZE, 1), name='input_t')

        cnn_in4d = keras.layers.Reshape((INPUT_SIZE, INPUT_SIZE, 1))(input_tensor)

        # CNN

        kernel1 = layers.Conv2D(filters=32, kernel_size=(5, 5), padding='SAME', activation=None,
                                kernel_initializer=tf.random.truncated_normal(stddev=0.1))(cnn_in4d)
        conv1_norm = layers.BatchNormalization()(kernel1)
        relu1 = layers.Activation('relu')(conv1_norm)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='VALID')(relu1)

        kernel2 = layers.Conv2D(filters=64, kernel_size=(5, 5), padding='SAME', activation=None,
                                kernel_initializer=tf.random.truncated_normal(stddev=0.1))(pool1)
        conv2_norm = layers.BatchNormalization()(kernel2)
        relu2 = layers.Activation('relu')(conv2_norm)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='VALID')(relu2)

        kernel3 = layers.Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation=None,
                                kernel_initializer=tf.random.truncated_normal(stddev=0.1))(pool2)
        conv3_norm = layers.Normalization()(kernel3)
        relu3 = layers.Activation('relu')(conv3_norm)
        pool3 = layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='VALID')(relu3)

        kernel4 = layers.Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation=None,
                                kernel_initializer=tf.random.truncated_normal(stddev=0.1))(pool3)
        conv4_norm = layers.BatchNormalization()(kernel4)
        relu4 = layers.Activation('relu')(conv4_norm)
        pool4 = layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='VALID')(relu4)

        kernel5 = layers.Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation=None,
                                kernel_initializer=tf.random.truncated_normal(stddev=0.1))(pool4)
        conv5_norm = layers.BatchNormalization()(kernel5)
        relu5 = layers.Activation('relu')(conv5_norm)
        pool5 = layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='VALID')(relu5)

        model = models.Model(inputs=input_tensor, outputs=pool5)

        self.net = model


class OCR:
    OCR_INPUT_SIZE = (200, 100)

    def __init__(self):
        cnn = self.cnn_model()
        rnn = self.rnn_model(52)  # Lower and upper case letters (26 + 26)
        ocr = self.ocr_model(cnn, rnn)
        self.model = ocr

    # Define the CNN architecture
    def cnn_model(self):
        inputs = Input(shape=(self.OCR_INPUT_SIZE[1], self.OCR_INPUT_SIZE[0], 3))
        x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Flatten()(x)
        outputs = layers.Dense(128, activation='relu')(x)
        cnn = models.Model(inputs=inputs, outputs=outputs)
        return cnn

    # Define the RNN architecture
    def rnn_model(self, num_classes):
        inputs = Input(shape=(None, 128))
        x = layers.LSTM(256, return_sequences=True)(inputs)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(256)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(128, activation='relu')(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        rnn = models.Model(inputs=inputs, outputs=outputs)
        return rnn

    # Combine the CNN and RNN models
    def ocr_model(self, cnn, rnn):
        inputs = Input(shape=(None, self.OCR_INPUT_SIZE[1], self.OCR_INPUT_SIZE[0], 3))
        x = layers.TimeDistributed(cnn)(inputs)
        # x = layers.TimeDistributed(layers.Reshape((-1, 128)))(x)
        x = layers.Reshape((-1, 128))(x)
        outputs = rnn(x)
        ocr = models.Model(inputs=inputs, outputs=outputs)
        ocr.summary()
        return ocr


class SimpleOCR:
    OCR_INPUT_SIZE = (80, 160, 1)
    RECOGNITION_CLASSES = 53
    MAX_LENGTH = 200

    def __init__(self):
        input_img = layers.Input(shape=self.OCR_INPUT_SIZE, name='input_1')

        # CNN for feature extraction
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_32')(input_img)
        x = layers.MaxPooling2D((2, 2), padding='same', name='pool_1')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_64')(x)
        x = layers.MaxPooling2D((2, 2), padding='same', name='pool_2')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_128')(x)
        x = layers.MaxPooling2D((2, 2), padding='same', name='pool_3')(x)

        # Prepare output for RNN
        x = layers.Reshape(target_shape=(self.MAX_LENGTH, 128), name='reshape_1')(x)
        x = layers.Masking(mask_value=0.0, name='masking_1')(x)

        # RNN for sequence recognition
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, recurrent_activation='sigmoid', unroll=True, name='lstm_1'), name='bidirectional_1')(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, recurrent_activation='sigmoid', unroll=True, name='lstm_2'), name='bidirectional_2')(x)

        # Dense layer and output
        x = layers.Dense(64, activation='relu', name='dense_1')(x)
        x = layers.Dropout(0.2, name='dropout_1')(x)
        output = layers.Dense(self.RECOGNITION_CLASSES, activation='softmax', name='output_1')(x)

        self.model = models.Model(inputs=input_img, outputs=output)

        self.model.summary()


'''
from yolov7_package import Yolov7Detector


class YOLOv7:
    def __int__(self):
        self.model = Yolov7Detector(weights='../best.pt').model
'''
if __name__ == '__main__':
    east = EAST().model
    ocr = OCR().model
    simple_ocr = SimpleOCR().model
    # yolov7 = YOLOv7()
