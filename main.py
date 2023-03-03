import cv2
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
        # denoised_map = creator.get_denoised_map(creator.image_crops[i])
        # canny_edge_map = creator.get_canny_edge_map(denoised_map)
        # dilation_map = creator.get_dilation_map(canny_edge_map)
        # creator.image_crops[i] = creator.get_threshold_map(dilation_map)
        creator.new_apply_hough_transform(creator.image_crops[i])
        creator.show('Crop {0}'.format(i), creator.image_crops[i])