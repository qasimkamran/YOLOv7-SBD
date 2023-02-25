import numpy as np

from signboard_detector import SignboardDetector
from signboard_recreation import SignboardCreator

if __name__ == '__main__':
    # detector = SignboardDetector()
    # detector.train_model("training/yolov7.yaml", "hyp.scratch.custom.yaml")

    creator = SignboardCreator(prediction_path='/home/qasimk/YOLOv7-SBD/yolov7/runs/detect/exp6/blank_input.png',
                               label_path='/home/qasimk/YOLOv7-SBD/yolov7/runs/detect/exp6/labels/blank_input.txt')

    creator.set_image_crops()
    for i in range(len(creator.image_crops)):
        creator.image_crops[i] = creator.create_edge_map(creator.image_crops[i])
        creator.show('Crop {0}'.format(i), creator.image_crops[i])