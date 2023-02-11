'''
This file wraps YOLOv7 scripts in abstracted functions.
The one-stop-shop for all network functionality on this repository.
'''

from yolov7_package import Yolov7Detector


class SignboardDetector:
    network = None

    def __init__(self, weights=None):
        assert weights is not None, f'Must specify model weights'

        self.network = Yolov7Detector()
        print('Successfully initialised network!')
        pass

    def ensure_data_integrity(self):
        # Ensuring that the dataset is in valid format
        print('Ensure data integrity')

    def train_model(self):
        # Training yolov7
        self.network.train()

    def test_model(self):
        # Testing yolov7
        print('Test')
