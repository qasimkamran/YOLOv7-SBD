'''
This file wraps YOLOv7 scripts in abstracted functions.
The one-stop-shop for all network functionality on this repository.
'''

import yolov7

class SignboardDetector:

    network = None

    def __init__(self, weights=None):
        self.network = yolov7.load(weights)
        pass

    def ensure_data_integrity()
        # Ensuring that dataset is in certain format

    def train_model():
        # Training yolov7

    def test_model():
        # Testing yolov7