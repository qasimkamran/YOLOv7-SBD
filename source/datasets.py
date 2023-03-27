import os
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

import losses
import models
import warnings

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
            images_data = np.load('../np_data/icdar13_images.npy')
            print('Loaded ICDAR 2013 images data')
            score_boxes = np.load('../np_data/icdar13_score_map_boxes.npy')
            rbox_boxes = np.load('../np_data/icdar13_rbox_map_boxes.npy')
            print('Loaded ICDAR 2013 bounding box data')
        except FileNotFoundError:
            return None, None, None

        return images_data, score_boxes, rbox_boxes

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
            img = self.images_data[i][0]
            for j in range(0, 19):
                x1, y1, x2, y2 = self.boxes_data[i][j]
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
        pred_width, pred_height = self.EAST_SIZE, self.EAST_SIZE
        true_width, true_height = self.INPUT_SIZE, self.INPUT_SIZE

        rboxes = np.zeros((20, 5))

        for j, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            scale_factor_x = pred_width / true_width
            scale_factor_y = pred_height / true_height
            x1, y1, x2, y2 = losses.rescale_coords(x1, y1, x2, y2, scale_factor_x, scale_factor_y)
            rboxes[j] = losses.bbox_to_rbox([x1, y1, x2, y2])

        overlayed_rboxes = np.ones((pred_width, pred_height))

        for j, rbox in enumerate(rboxes):
            x, y, w, h, angle = rboxes[j]
            rbox = cv2.boxPoints(((x, y), (w, h), angle))
            rbox = np.int0(rbox)
            cv2.drawContours(overlayed_rboxes, [rbox], 0, (0, 0, 0), 1)

        ones_converted = np.zeros((pred_width, pred_height, 5))

        # Get the indices of all 0s in the image
        zero_indices = np.argwhere(overlayed_rboxes == 0)

        # Compute the distance transform of the image
        distance_transform = distance_transform_edt(overlayed_rboxes)

        # Initialize the output arrays
        distances = np.zeros((overlayed_rboxes.shape[0], overlayed_rboxes.shape[1], 4), dtype=np.float32)
        angles = np.zeros_like(overlayed_rboxes, dtype=np.float32)

        # Loop over all pixels in the image
        for j in range(overlayed_rboxes.shape[0]):
            for k in range(overlayed_rboxes.shape[1]):
                if overlayed_rboxes[j, k] == 1:
                    # Compute the distance to the closest 0 in each direction
                    distances[j, k, 0] = distance_transform[j, :k][::-1].argmin() + 1 if k > 0 else 0
                    distances[j, k, 1] = distance_transform[:j, k][::-1].argmin() + 1 if j > 0 else 0
                    distances[j, k, 2] = distance_transform[j, k + 1:].argmin() + 1 if k < overlayed_rboxes.shape[
                        1] - 1 else 0
                    distances[j, k, 3] = distance_transform[j + 1:, k].argmin() + 1 if j < overlayed_rboxes.shape[
                        0] - 1 else 0

                    # Compute the angle to the closest 0
                    x, y = zero_indices[
                        np.argmin(np.sqrt((j - zero_indices[:, 0]) ** 2 + (k - zero_indices[:, 1]) ** 2))]
                    angles[j, k] = np.arctan2(y - k, x - j)

        ones_converted = np.concatenate([distances, angles[..., None]], axis=-1)

        return ones_converted
