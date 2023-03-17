import os.path

import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.data import AUTOTUNE
from keras import layers, models, Input
from keras.applications.resnet import ResNet50
from keras.losses import MeanSquaredError
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
import numpy as np
import deeplake
import cv2


class EAST():
    model = None
    images_data = None
    boxes_data = None


    def __init__(self, img):
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           height_shift_range=0.05,
                                           width_shift_range=0.05)
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        val_datagen = ImageDataGenerator(rescale=1. / 255)

        # self.save_dataset()
        self.images_data, self.boxes_data = self.load_dataset()
        self.display_data_batch()
        pass

    # Define the input size
    INPUT_SIZE = 512

    # Bilinear resize factor
    RESIZE_FACTOR = 2

    def load_dataset(self):
        # Check if saved before loading
        os.path.isdir('../np_data')
        os.path.isfile('../np_data/images.npy')
        os.path.isfile('../np_data/boxes.npy')

        images_data = np.load('np_data/images.npy')
        print('Loaded images data')
        boxes_data = np.load('np_data/boxes.npy')
        print('Loaded bounding box data')
        return images_data, boxes_data

    def save_dataset(self):
        # Load dataset from deeplake
        dataset = deeplake.load("hub://activeloop/icdar-2013-text-localize-train")
        tf_dataset = dataset.tensorflow()

        # Iterate over the dataset preprocessing it's contents and forming numpy arrays
        iterator = iter(tf_dataset)
        images = []
        boxes = []
        for element in iterator:
            image, box = self.preprocess(element)
            images.append(image)
            boxes.append(box)
        boxes = self.homogenize_array(boxes)  # Inhomogeneous array boxes passed
        images = np.array(images)

        # Saving numpy arrays externally for faster access in subsequent runs
        np.save('np_data/images.npy', images)
        print('Saving numpy array for images of shape:', images.shape)
        np.save('np_data/boxes.npy', boxes)
        print('Saving numpy array for boxes of shape:', boxes.shape)

    def homogenize_array(self, array):
        # Get the maximum number of elements
        max_elements = max([element.shape[0] for element in array])
        # Pad all the individual arrays with zeros
        for i in range(len(array)):
            num_elements = array[i].shape[0]
            if num_elements < max_elements:
                pad_width = ((0, max_elements - num_elements), (0, 0))
                array[i] = np.pad(array[i], pad_width, mode='constant')
        # Convert the boxes array to a numpy array
        result = np.array(array)
        return result

    def display_data_batch(self):
        for i in range(0, 100):
            img = self.images_data[i][0]
            for j in range(0, 19):
                x1, y1, x2, y2 = self.boxes_data[i][j]
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.imshow('sample_{i}', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def preprocess(self, tensor):
        """Preprocess the input image by resizing and normalizing the pixel values."""
        box = tensor['boxes/box'].numpy()
        img = tensor['images'].numpy()

        h, w = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (self.INPUT_SIZE, self.INPUT_SIZE))
        box = box * [self.INPUT_SIZE / w, self.INPUT_SIZE / h, self.INPUT_SIZE / w, self.INPUT_SIZE / h]

        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = img.astype(np.float32) / 255.0
        return img, box

    def resize_bilinear(self, layer):
        return tf.image.resize(layer, size=[K.int_shape(layer)[1] * self.RESIZE_FACTOR,
                                            K.int_shape(layer)[2] * self.RESIZE_FACTOR])

    def make_model(self):
        input_tensor = Input(shape=(self.INPUT_SIZE, self.INPUT_SIZE, 3), name='input_t')
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

        stem_tensor_32 = layers.Lambda(self.resize_bilinear, name='stem_1')(fusion_tensor_16)
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

        stem_tensor_64 = layers.Lambda(self.resize_bilinear, name='stem_2')(activation2_block_1)
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

        stem_tensor_128 = layers.Lambda(self.resize_bilinear, name='stem_3')(activation2_block_2)
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
        rbox_geo_map_relative = layers.Lambda(lambda x: x * self.INPUT_SIZE)(rbox_geo_map)

        rbox_angle_map = layers.Conv2D(1, (1, 1), activation='sigmoid', name='rbox_angle_map')(ultimate_conv)
        rbox_angle_map_radians = layers.Lambda(lambda x: (x - 0.5) * np.pi / 2)(rbox_angle_map)

        rbox_map = K.concatenate([rbox_geo_map_relative, rbox_angle_map_radians], axis=3)

        model = models.Model(inputs=input_tensor, outputs=[score_map, rbox_map])

        self.model = model

    def predict(self, img):
        # Run inference on a single image.
        img = self.preprocess(img)

        # Make predictions
        score_map, rbox_map = self.model.predict(img)

        return score_map, rbox_map

    def plot_prediction(self, img, score_map, rbox_map):
        # Overlay predicted boxes on the original image
        for i in range(score_map.shape[0]):
            for j in range(score_map.shape[1]):
                if np.any(score_map[i, j] > 0.5):  # Only draw boxes with score > 0.5
                    x, y, w, h, angle = rbox_map[i, j][3]
                    box = cv2.boxPoints(((x, y), (w, h), angle))
                    box = np.int0(box)
                    cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

        # Display the image with overlays
        cv2.imshow('Overlay', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
