import argparse
import os
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

import losses
import models
import warnings
import math
import icdar

warnings.filterwarnings('ignore', message='A newer version of deeplake*')
import deeplake

MAIN_DATASET_DIR = '../np_data'


class ICDAR13:
    images_data = None
    score_boxes = None
    rbox_boxes = None

    INPUT_SIZE = models.INPUT_SIZE

    EAST_SIZE = 128

    def load_dataset(self):
        # Check if saved before loading
        os.path.isdir('../np_data')
        os.path.isfile('../np_data/icdar13_images.npy')
        os.path.isfile('../np_data/icdar13_boxes.npy')

        try:
            self.images_data = np.load('../np_data/icdar13_images.npy')
            print('Loaded ICDAR 2013 images data')
            self.score_boxes = np.load('../np_data/icdar13_score_map_boxes.npy')
            self.rbox_boxes = np.load('../np_data/icdar13_rbox_map_boxes.npy')
            print('Loaded ICDAR 2013 bounding box data')
        except FileNotFoundError:
            return None, None, None

        return self.images_data, self.score_boxes, self.rbox_boxes

    def save_dataset(self):
        # Load dataset from deeplake
        dataset = deeplake.load("hub://activeloop/icdar-2013-text-localize-train")
        tf_dataset = dataset.tensorflow()

        # Iterate over the dataset preprocessing its contents and forming numpy arrays
        iterator = iter(tf_dataset)
        images = []
        score_map_compliants = []
        rbox_map_compliants = []
        boxes = []
        for element in iterator:
            image, score_map_compliant, rbox_map_compliant = self.preprocess(element)
            images.append(image)
            score_map_compliants.append(score_map_compliant)
            rbox_map_compliants.append(rbox_map_compliant)
        score_map_compliants = self.homogenize_array(score_map_compliants)
        rbox_map_compliants = self.homogenize_array(rbox_map_compliants)
        images = np.array(images)

        # Saving numpy arrays externally for faster access in subsequent runs
        if not os.path.exists(MAIN_DATASET_DIR):
            os.mkdir(MAIN_DATASET_DIR)

        np.save('../np_data/icdar13_images.npy', images)
        print('Saving numpy array for ICDAR 2013 images of shape:', images.shape)
        np.save('../np_data/icdar13_score_map_boxes.npy', score_map_compliants)
        np.save('../np_data/icdar13_rbox_map_boxes.npy', rbox_map_compliants)
        print('Saving numpy array for ICDAR 2013 boxes [Shape print TODO]')

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
            img = self.images_data[i]
            for j in range(0, 19):
                x1, y1, x2, y2 = self.score_boxes[i][j]
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.imshow('sample_{i}', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def preprocess(self, tensor, ):
        """Preprocess the input image by resizing and normalizing the pixel values."""
        box = tensor['boxes/box'].numpy()
        img = tensor['images'].numpy()

        h, w = img.shape[:2]

        img = cv2.resize(img, (self.INPUT_SIZE, self.INPUT_SIZE))
        score_map_compliant = box * [self.INPUT_SIZE / w, self.INPUT_SIZE / h, self.INPUT_SIZE / w, self.INPUT_SIZE / h]
        rbox_map_compliant = self.process_rbox_map(score_map_compliant)

        img = img.astype(np.float32) / 255.0
        return img, score_map_compliant, rbox_map_compliant

    def process_rbox_map(self, boxes):
        rboxes = np.zeros(boxes.shape)

        for i, box in enumerate(boxes):
            xmin, ymin, xmax, ymax = box

            center_top = (round((xmin + xmax) / 2), round(ymin))
            center_bottom = (round((xmin + xmax) / 2), round(ymax))

            slope = (center_bottom[0] - center_top[0]) / (center_bottom[1] - center_top[1])
            y_intercept = center_top[1] - slope * center_top[0]

            if slope == 0:
                slope = 1

            x_left = (ymin - y_intercept) / slope
            x_right = (ymax - y_intercept) / slope
            y_top = slope * xmin + y_intercept
            y_bottom = slope * xmax + y_intercept

            d1 = math.sqrt((x_left - xmin) ** 2 + (ymin - y_top) ** 2)
            d2 = math.sqrt((xmax - x_right) ** 2 + (ymin - y_top) ** 2)
            d3 = math.sqrt((xmax - x_right) ** 2 + (ymax - y_bottom) ** 2)
            d4 = math.sqrt((x_left - xmin) ** 2 + (ymax - y_bottom) ** 2)

            rboxes[i, 0] = d1
            rboxes[i, 1] = d2
            rboxes[i, 2] = d3
            rboxes[i, 3] = d4

        return rboxes

class ICDAR15:
    images_data = None
    score_maps_data = None
    geo_maps_data = None

    INPUT_SIZE = models.INPUT_SIZE

    EAST_SIZE = 128

    def save_dataset(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--validation_data_path', type=str, default='../train_data')
        parser.add_argument('--input_size', type=int, default=self.INPUT_SIZE)
        parser.add_argument('--geometry', type=str, default='RBOX')
        parser.add_argument('--max_image_large_side', type=int, default=1280)
        parser.add_argument('--max_text_size', type=int, default=800)
        parser.add_argument('--min_text_size', type=int, default=10)
        parser.add_argument('--min_crop_side_ratio', type=float, default=0.1)
        FLAGS = parser.parse_args()
        FLAGS.suppress_warnings_and_error_messages = False
        FLAGS.min_crop_side_ratio = 0.1

        # Saving numpy arrays externally for faster access in subsequent runs
        if not os.path.exists(MAIN_DATASET_DIR):
            os.mkdir(MAIN_DATASET_DIR)

        images, score_maps, geo_maps = icdar.load_data(FLAGS)

        np.save('../np_data/icdar15_images.npy', images)
        print('Saving numpy array for ICDAR 2015 images of shape:', images.shape)

        np.save('../np_data/icdar15_score_maps.npy', score_maps)
        print('Saving numpy array for ICDAR 2015 score maps of shape:', score_maps.shape)

        np.save('../np_data/icdar15_geo_maps.npy', geo_maps)
        print('Saving numpy array for ICDAR 2015 geo maps of shape:', geo_maps.shape)

    def load_dataset(self):
        # Check if saved before loading
        os.path.isdir('../np_data')
        os.path.isfile('../np_data/icdar15_images.npy')
        os.path.isfile('../np_data/icdar15_score_maps.npy')
        os.path.isfile('../np_data/icdar15_geo_maps.npy')

        images = np.load('../np_data/icdar15_images.npy')
        score_maps = np.load('../np_data/icdar15_score_maps.npy')
        geo_maps = np.load('../np_data/icdar15_geo_maps.npy')

        try:
            self.images_data = np.load('../np_data/icdar15_images.npy')
            print('Loaded ICDAR 2015 images data')

            self.score_maps_data = np.load('../np_data/icdar15_score_maps.npy')
            print('Loaded ICDAR 2015 score maps data')

            self.geo_maps_data = np.load('../np_data/icdar15_geo_maps.npy')
            print('Loaded ICDAR 2015 geo maps data')
        except FileNotFoundError:
            return None, None, None

        return self.images_data, self.score_maps_data, self.geo_maps_data
