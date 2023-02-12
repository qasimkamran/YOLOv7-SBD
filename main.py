from signboard_detector import SignboardDetector
from signboard_recreation import SignboardCreator

if __name__ == '__main__':
    # detector = SignboardDetector()
    # detector.train_model("training/yolov7.yaml", "hyp.scratch.custom.yaml")

    creator = SignboardCreator(prediction_path='/home/qasimk/YOLOv7-SBD/yolov7/runs/detect/exp6/blank_input.png',
                               label_path='/home/qasimk/YOLOv7-SBD/yolov7/runs/detect/exp6/labels/blank_input.txt')
    boxes = creator.list_from_labels_string(creator.labels)
    creator.highlight_crops(boxes)
    creator.show_prediction('Prediction')