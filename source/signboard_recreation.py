'''
This file includes all the functionality required to recreate content within
a signboard prediction through its bounding box crop.
'''

import cv2
import os
import numpy as np
from PIL import Image

class SignboardCreator:

    labels = None
    prediction = None
    image_crops = None

    def __init__(self, label_path, prediction_path):
        assert os.path.exists(label_path), f'Incorrect file path: {label_path}'
        assert os.path.exists(prediction_path), f'Incorrect file path: {prediction_path}'

        with open(label_path) as file:
            self.labels = file.read()
            print('Read prediction labels as list of strings')

        with Image.open(prediction_path) as image:
            self.prediction = np.array(image)
            print('Read prediction image as np array')
        pass

    def show_prediction(self, win_name):
        print(self.labels)
        cv2.imshow(win_name, self.prediction)
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

    def highlight_crops(self):
        # Loop over the boxes and draw them on the image
        for box in self.labels:
            # Extract the bounding box coordinates
            class_id, x_center, y_center, width, height = box
            x1 = int((x_center - width / 2) * self.prediction.shape[1])
            y1 = int((y_center - height / 2) * self.prediction.shape[0])
            x2 = int((x_center + width / 2) * self.prediction.shape[1])
            y2 = int((y_center + height / 2) * self.prediction.shape[0])

            # Draw the bounding box on the image
            cv2.rectangle(self.prediction, (x1, y1), (x2, y2), (0, 0, 0), 2)

        # Return the image with the bounding boxes drawn
        return self.prediction