'''
This file wraps YOLOv7 scripts in abstracted functions.
The one-stop-shop for all network functionality on this repository.
'''

import os
from yolov7_package import Yolov7Detector


class SignboardDetector:
    network = None

    # Yaml directories
    HYP_DIR = os.path.abspath('hyp')
    DATA_DIR = os.path.abspath('data')
    CFG_DIR = os.path.abspath('cfg')

    # Results directories
    RESULTS_DIR = os.path.abspath('results')

    def __init__(self):
        self.network = Yolov7Detector()
        print('Successfully initialised network!')
        pass

    def ensure_data_integrity(self):
        # Ensuring that the dataset is in valid format
        print('Ensure data integrity')

    def train_model(self, cfg, hyp):
        # Stay the same for any configuration
        data = os.path.join(self.DATA_DIR, 'signboard.yaml')
        save_dir = os.path.join(self.RESULTS_DIR, 'train')

        cfg = os.path.join(self.CFG_DIR, cfg)
        hyp = os.path.join(self.HYP_DIR, hyp)

        self.network.train(save_dir=save_dir,
                           cfg=cfg,
                           hyp=hyp,
                           data=data)

    def make_prediction(self):
        # Testing yolov7
        print('Test')
