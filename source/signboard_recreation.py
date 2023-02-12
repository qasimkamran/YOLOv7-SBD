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

    def __init__(self, label_path, prediction_path):
        assert os.path.exists(label_path), f'Incorrect file path: {label_path}'
        assert os.path.exists(prediction_path), f'Incorrect file path: {prediction_path}'

        with open(label_path) as file:
            self.labels = file.read()
            print('Read prediction labels as string')

        with Image.open(prediction_path) as image:
            self.prediction = np.array(image)
            print('Read prediction image as np array')
        pass

    def show_prediction(self, win_name):
        cv2.imshow(win_name, self.prediction)
        cv2.waitKey(0)
        cv2.destroyWindow(win_name)