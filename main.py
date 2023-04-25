from pt_train import SignboardDetector
from signboard_recreation import SignboardCreator
from text_detector import EAST

if __name__ == '__main__':
    # detector = SignboardDetector()
    # detector.train_model("training/yolov7.yaml", "hyp.scratch.custom.yaml")

    creator = SignboardCreator(prediction_path='/home/qasimk/YOLOv7-SBD/yolov7/runs/detect/exp6/blank_input.png',
                               label_path='/home/qasimk/YOLOv7-SBD/yolov7/runs/detect/exp6/labels/blank_input.txt')
    creator.set_image_crops()

    print(creator.image_crops[1].shape)

    # east = EAST(creator.image_crops[1])
