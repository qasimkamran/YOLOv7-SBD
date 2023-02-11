'''
This file wraps YOLOv7 scripts in abstracted functions.
The one-stop-shop for all network functionality on this repository.
'''

import yolov7


class SignboardDetector:
    network = None

    def __init__(self, weights=None):
        assert weights is not None, f'Must specify model weights'
        self.network = yolov7.load(weights)
        print('Successfully loaded model!')
        pass

    def ensure_data_integrity(self):
        # Ensuring that the dataset is in valid format
        print('Ensure data integrity')

    def train_model(self):
        # Training yolov7
        print('Train')

    def test_model(self):
        # Testing yolov7
        print('Test')
