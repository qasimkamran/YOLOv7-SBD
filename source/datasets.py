import os
import cv2
import deeplake
import numpy as np
import models

MAIN_DATASET_DIR = '../np_data'


class ICDAR13:

    images_data = None
    boxes_data = None

    INPUT_SIZE = models.INPUT_SIZE

    def load_dataset(self):
        # Check if saved before loading
        os.path.isdir('../np_data')
        os.path.isfile('../np_data/icdar13_images.npy')
        os.path.isfile('../np_data/icdar13_boxes.npy')

        try:
            images_data = np.load('../np_data/icdar13_images.npy')
            print('Loaded ICDAR 2013 images data')
            boxes_data = np.load('../np_data/icdar13_boxes.npy')
            print('Loaded ICDAR 2013 bounding box data')
        except FileNotFoundError:
            return None, None

        return images_data, boxes_data

    def save_dataset(self):
        # Load dataset from deeplake
        dataset = deeplake.load("hub://activeloop/icdar-2013-text-localize-train")
        tf_dataset = dataset.tensorflow()

        # Iterate over the dataset preprocessing its contents and forming numpy arrays
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
        if not os.path.exists(MAIN_DATASET_DIR):
            os.mkdir(MAIN_DATASET_DIR)

        np.save('../np_data/icdar13_images.npy', images)
        print('Saving numpy array for ICDAR 2013 images of shape:', images.shape)
        np.save('../np_data/icdar13_boxes.npy', boxes)
        print('Saving numpy array for ICDAR 2013 boxes of shape:', boxes.shape)

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
        box = box * [self.INPUT_SIZE / w, self.INPUT_SIZE / h, self.INPUT_SIZE / w, self.INPUT_SIZE / h]

        img = img.astype(np.float32) / 255.0
        return img, box