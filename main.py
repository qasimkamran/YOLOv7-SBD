from signboard_detector import SignboardDetector

if __name__ == '__main__':
    detector = SignboardDetector()
    detector.train_model("training/yolov7.yaml", "hyp.scratch.custom.yaml")