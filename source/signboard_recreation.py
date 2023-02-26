'''
This file includes all the functionality required to recreate content within
a signboard prediction through its bounding box crop.
'''

import cv2
import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

class SignboardCreator:

    labels = []
    prediction = None
    image_crops = []

    def __init__(self, label_path, prediction_path):
        assert os.path.exists(label_path), f'Incorrect file path: {label_path}'
        assert os.path.exists(prediction_path), f'Incorrect file path: {prediction_path}'

        with open(label_path) as file:
            self.labels = self.list_from_labels_string(file.read())
            print('Read prediction labels as list')

        with Image.open(prediction_path) as image:
            self.prediction = np.array(image)
            print('Read prediction image as np array')
        pass

    def show(self, win_name, mat=None):
        if mat is None:
            mat = self.prediction
        cv2.imshow(win_name, mat)
        cv2.waitKey(0)
        cv2.destroyWindow(win_name)

    def is_valid_label_format(self, labels_string):
        # List of each detection's label as a line of text
        labels_string_list = labels_string.strip().split('\n')

        try:
            for label_string in labels_string_list:
                values = list(map(float, label_string.strip().split()))
                if len(values) != 5:
                    return False
                if int(values[0]) != values[0]:
                    return False
                return True
        except:
            return False

    def list_from_labels_string(self, labels_string):
        assert self.is_valid_label_format(labels_string), f'Invalid label format'

        labels_list = []

        # List of each detection's label as a line of text
        labels_string_list = labels_string.strip().split('\n')

        # Split values as floats from each string and append to labels_list
        for label_string in labels_string_list:
            label_values = list(map(float, label_string.strip().split()))
            labels_list.append(label_values)

        return labels_list

    def set_image_crops(self):
        # Loop over the boxes and draw them on the image
        for box in self.labels:
            prediction_copy = np.copy(self.prediction)

            # Extract the bounding box coordinates
            class_id, x_center, y_center, width, height = box
            x1 = int((x_center - width / 2) * prediction_copy.shape[1])
            y1 = int((y_center - height / 2) * prediction_copy.shape[0])
            x2 = int((x_center + width / 2) * prediction_copy.shape[1])
            y2 = int((y_center + height / 2) * prediction_copy.shape[0])

            prediction_copy = prediction_copy[y1:y2, x1:x2]

            assert 0 not in prediction_copy.shape, f'Invalid crop made'

            self.image_crops.append(prediction_copy)

        assert len(self.image_crops) == len(self.labels), f'Image crops length: {len(self.image_crops)} ' \
                                                          f'not equal to ' \
                                                          f'labels length: {len(self.labels)}'

    def plot_image_crops(self):
        assert self.image_crops, f'Image crops not set'

        nimages = len(self.image_crops)

        # Display nimages for rows until they equal or exceed 5 in number
        nrows = nimages
        if nimages >= 5:
            nrows = 5

        ncols = int(np.ceil(nimages / nrows))

        fig, axs = plt.subplots(nrows, ncols, figsize=(20, 20))
        axs = axs.ravel()
        for i, img in enumerate(self.image_crops):
            axs[i].imshow(img)
            axs[i].axis('off')
        plt.show()

    def get_canny_edge_detection(self, mat):
        assert mat is not None, f'Function argument cannot be None'
        grayscale_map = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
        canny_edge_map = cv2.Canny(grayscale_map, 100, 150)  # Apply Canny edge detection
        return canny_edge_map

    def get_laplacian_edge_detection(self, mat):
        assert mat is not None, f'Function argument cannot be None'
        grayscale_map = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
        laplacian_edge_map = cv2.Laplacian(grayscale_map, cv2.CV_64F)  # Apply Laplacian edge detection
        laplacian_edge_map = np.uint8(np.absolute(laplacian_edge_map))  # Convert the edge map to unsigned 8-bit integer format
        return laplacian_edge_map

    def get_threshold_edge_detection(self, edge_map):
        assert edge_map is not None, f'Function argument cannot be None'
        # Apply a binary threshold to convert the edge map to a binary image
        threshold_value = 50  # adjust this value as needed
        _, binary = cv2.threshold(edge_map, threshold_value, 255, cv2.THRESH_BINARY)
        return binary

    def apply_hough_transform(self, mat):
        canny_edge_map = self.get_canny_edge_detection(mat)
        threshold_edge_map = self.get_threshold_edge_detection(canny_edge_map)

        # Perform Hough transform to detect lines
        lines = cv2.HoughLines(threshold_edge_map, rho=1, theta=np.pi / 360, threshold=100)

        # Draw the detected lines on the original image
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a, b = np.cos(theta), np.sin(theta)
                x0, y0 = a * rho, b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                cv2.line(mat, pt1, pt2, (0, 0, 255), 2)