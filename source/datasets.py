import argparse
import glob
import os
import cv2
import numpy as np
from keras.utils import pad_sequences
from matplotlib import pyplot as plt
from scipy.ndimage import distance_transform_edt
from tensorflow.keras.utils import to_categorical
import losses
import tf_models
import warnings
import math
import icdar
import tensorflow as tf

warnings.filterwarnings('ignore', message='A newer version of deeplake*')
import deeplake

MAIN_DATASET_DIR = '../np_data'


class ICDAR13:
    images_data = None
    score_boxes = None
    rbox_boxes = None

    INPUT_SIZE = tf_models.INPUT_SIZE

    EAST_SIZE = 128

    def load_dataset(self):
        # Check if saved before loading
        os.path.isdir('../np_data')
        os.path.isfile('../np_data/icdar13_images.npy')
        os.path.isfile('../np_data/icdar13_boxes.npy')

        try:
            self.images_data = np.load('../np_data/icdar13_images.npy')
            print('Loaded ICDAR 2013 images data')
            self.score_boxes = np.load('../np_data/icdar13_score_map_boxes.npy')
            self.rbox_boxes = np.load('../np_data/icdar13_rbox_map_boxes.npy')
            print('Loaded ICDAR 2013 bounding box data')
        except FileNotFoundError:
            return None, None, None

        return self.images_data, self.score_boxes, self.rbox_boxes

    def save_dataset(self):
        # Load icdar13 from deeplake
        dataset = deeplake.load("hub://activeloop/icdar-2013-text-localize-train")
        tf_dataset = dataset.tensorflow()

        # Iterate over the icdar13 preprocessing its contents and forming numpy arrays
        iterator = iter(tf_dataset)
        images = []
        score_map_compliants = []
        rbox_map_compliants = []
        boxes = []
        for element in iterator:
            image, score_map_compliant, rbox_map_compliant = self.preprocess(element)
            images.append(image)
            score_map_compliants.append(score_map_compliant)
            rbox_map_compliants.append(rbox_map_compliant)
        score_map_compliants = self.homogenize_array(score_map_compliants)
        rbox_map_compliants = self.homogenize_array(rbox_map_compliants)
        images = np.array(images)

        # Saving numpy arrays externally for faster access in subsequent runs
        if not os.path.exists(MAIN_DATASET_DIR):
            os.mkdir(MAIN_DATASET_DIR)

        np.save('../np_data/icdar13_images.npy', images)
        print('Saving numpy array for ICDAR 2013 images of shape:', images.shape)
        np.save('../np_data/icdar13_score_map_boxes.npy', score_map_compliants)
        np.save('../np_data/icdar13_rbox_map_boxes.npy', rbox_map_compliants)
        print('Saving numpy array for ICDAR 2013 boxes [Shape print TODO]')

    @staticmethod
    def homogenize_array(array):
        # Get the maximum number of elements
        max_elements = max([element.shape[0] for element in array])
        # Pad all the individual arrays with zeros
        for i in range(len(array)):
            num_elements = array[i].shape[0]
            if num_elements < max_elements:
                pad_width = ((0, max_elements - num_elements), (0, 0))
                array[i] = np.pad(array[i], pad_width, mode='constant')
        # Convert the boxes array to a numpy array
        result = np.array(array)
        return result

    def display_data_batch(self):
        for i in range(0, 100):
            img = self.images_data[i]
            for j in range(0, 19):
                x1, y1, x2, y2 = self.score_boxes[i][j]
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.imshow('sample_{i}', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def preprocess(self, tensor, ):
        """Preprocess the input image by resizing and normalizing the pixel values."""
        box = tensor['boxes/box'].numpy()
        img = tensor['images'].numpy()

        h, w = img.shape[:2]

        img = cv2.resize(img, (self.INPUT_SIZE, self.INPUT_SIZE))
        score_map_compliant = box * [self.INPUT_SIZE / w, self.INPUT_SIZE / h, self.INPUT_SIZE / w, self.INPUT_SIZE / h]
        rbox_map_compliant = self.process_rbox_map(score_map_compliant)

        img = img.astype(np.float32) / 255.0
        return img, score_map_compliant, rbox_map_compliant

    def process_rbox_map(self, boxes):
        rboxes = np.zeros(boxes.shape)

        for i, box in enumerate(boxes):
            xmin, ymin, xmax, ymax = box

            center_top = (round((xmin + xmax) / 2), round(ymin))
            center_bottom = (round((xmin + xmax) / 2), round(ymax))

            slope = (center_bottom[0] - center_top[0]) / (center_bottom[1] - center_top[1])
            y_intercept = center_top[1] - slope * center_top[0]

            if slope == 0:
                slope = 1

            x_left = (ymin - y_intercept) / slope
            x_right = (ymax - y_intercept) / slope
            y_top = slope * xmin + y_intercept
            y_bottom = slope * xmax + y_intercept

            d1 = math.sqrt((x_left - xmin) ** 2 + (ymin - y_top) ** 2)
            d2 = math.sqrt((xmax - x_right) ** 2 + (ymin - y_top) ** 2)
            d3 = math.sqrt((xmax - x_right) ** 2 + (ymax - y_bottom) ** 2)
            d4 = math.sqrt((x_left - xmin) ** 2 + (ymax - y_bottom) ** 2)

            rboxes[i, 0] = d1
            rboxes[i, 1] = d2
            rboxes[i, 2] = d3
            rboxes[i, 3] = d4

        return rboxes


class ICDAR15:
    images_data = None
    score_maps_data = None
    geo_maps_data = None

    cropped_images_data = None
    transcription_data = None
    index_transcriptions = None
    one_hot_labels_data = None

    EAST_INPUT_SIZE = tf_models.INPUT_SIZE
    EAST_SIZE = 128

    OCR_INPUT_SIZE = (160, 80)
    RECOGNITION_CLASSES = 53

    char_to_index = None

    TRAIN_DIR = '../raw_data/icdar_composite'

    def save_detection_dataset(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--validation_data_path', type=str, default=self.TRAIN_DIR)
        parser.add_argument('--input_size', type=int, default=self.EAST_INPUT_SIZE)
        parser.add_argument('--geometry', type=str, default='RBOX')
        parser.add_argument('--max_image_large_side', type=int, default=1280)
        parser.add_argument('--max_text_size', type=int, default=800)
        parser.add_argument('--min_text_size', type=int, default=10)
        parser.add_argument('--min_crop_side_ratio', type=float, default=0.1)
        FLAGS = parser.parse_args()
        FLAGS.suppress_warnings_and_error_messages = False
        FLAGS.min_crop_side_ratio = 0.1

        # Saving numpy arrays externally for faster access in subsequent runs
        if not os.path.exists(MAIN_DATASET_DIR):
            os.mkdir(MAIN_DATASET_DIR)

        images, score_maps, geo_maps = icdar.load_data(FLAGS)

        np.save('../np_data/icdar15_detection_images.npy', images)
        print('Saving numpy array for ICDAR 2015 images of shape:', images.shape)

        np.save('../np_data/icdar15_detection_score_maps.npy', score_maps)
        print('Saving numpy array for ICDAR 2015 score maps of shape:', score_maps.shape)

        np.save('../np_data/icdar15_detection_geo_maps.npy', geo_maps)
        print('Saving numpy array for ICDAR 2015 geo maps of shape:', geo_maps.shape)

    def load_detection_dataset(self):
        # Check if saved before loading
        os.path.isdir('../np_data')
        os.path.isfile('../np_data/icdar15_detection_images.npy')
        os.path.isfile('../np_data/icdar15_detection_score_maps.npy')
        os.path.isfile('../np_data/icdar15_detection_geo_maps.npy')

        try:
            self.images_data = np.load('../np_data/icdar15_detection_images.npy')
            print('Loaded ICDAR 2015 images data')

            self.score_maps_data = np.load('../np_data/icdar15_detection_score_maps.npy')
            print('Loaded ICDAR 2015 score maps data')

            self.geo_maps_data = np.load('../np_data/icdar15_detection_geo_maps.npy')
            print('Loaded ICDAR 2015 geo maps data')
        except FileNotFoundError:
            return None, None, None

        return self.images_data, self.score_maps_data, self.geo_maps_data

    def save_recognition_dataset(self):
        parser = argparse.ArgumentParser()
        FLAGS = parser.parse_args()
        FLAGS.suppress_warnings_and_error_messages = False
        FLAGS.min_crop_side_ratio = 0.1

        self.cropped_images_data = []
        self.transcription_data = []
        self.one_hot_labels_data = []

        image_paths = glob.glob(os.path.join(self.TRAIN_DIR, '*.jpg'))

        for i in range(900):
            image_path = image_paths[i]
            image = cv2.imread(image_path)
            h, w = image.shape[:2]

            image_fname = os.path.split(image_path)[-1]
            image_fname_noext = os.path.splitext(image_fname)[0]
            label_fname = 'gt_' + image_fname_noext + '.txt'

            label_path = os.path.join(self.TRAIN_DIR, label_fname)
            text_polys, text_tags = icdar.load_annotation(label_path)
            text_polys, text_tags = icdar.check_and_validate_polys(FLAGS, text_polys, text_tags, (h, w))

            for text_poly in text_polys:
                x, y, w, h = cv2.boundingRect(text_poly)
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        fields = line.strip().split(',')
                        label_x = fields[0]
                        label_y = fields[3]
                        label_string = fields[8]
                        if str(x) == label_x and str(y) == label_y and label_string.isalpha():
                            self.transcription_data.append(label_string)
                            cropped_image = image[y:y + h, x:x + w]
                            self.cropped_images_data.append(cropped_image)

        for i, cropped_image in enumerate(self.cropped_images_data):
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            cropped_image = cv2.resize(cropped_image, self.OCR_INPUT_SIZE)
            self.cropped_images_data[i] = np.expand_dims(cropped_image, axis=-1)

        # Create the character set and mapping dictionary
        char_set = sorted(set(''.join(self.transcription_data)))
        self.char_to_index = {c: i+1 for i, c in enumerate(char_set)}

        # Convert transcriptions to lists of indices
        self.index_transcriptions = [[self.char_to_index[c] for c in transcription] for transcription in self.transcription_data]

        # Function to convert a list of indices to a one-hot encoded array
        def one_hot_encode(indices, num_classes):
            one_hot = np.zeros((len(indices), num_classes))
            for i, index in enumerate(indices):
                one_hot[i, index] = 1
            return one_hot

        # Convert the lists of indices to one-hot encoded arrays
        self.one_hot_labels_data = [one_hot_encode(index_transcription, self.RECOGNITION_CLASSES) for index_transcription in
                                    self.index_transcriptions]

        self.cropped_images_data = np.array(self.cropped_images_data, dtype=np.float64)
        self.one_hot_labels_data = np.asarray(self.one_hot_labels_data)
        self.transcription_data = np.asarray(self.transcription_data)

        self.index_transcriptions = pad_sequences(self.index_transcriptions, padding='post', value=0)

        assert len(self.cropped_images_data) == len(self.transcription_data) == len(
            self.one_hot_labels_data), f'Lengths mismatch in lists'

        np.save('../np_data/icdar15_recognition_index_transcriptions.npy', self.index_transcriptions)
        print('Saving numpy array for ICDAR 2015 recognition indexed transcriptions of shape:', self.index_transcriptions.shape)

        np.save('../np_data/icdar15_recognition_images.npy', self.cropped_images_data)
        print('Saving numpy array for ICDAR 2015 recognition images of shape:', self.cropped_images_data.shape)

        np.save('../np_data/icdar15_recognition_transcripts.npy', self.transcription_data)
        print('Saving numpy array for ICDAR 2015 transcripts of length:', len(self.transcription_data))

        np.save('../np_data/icdar15_recognition_one_hot_labels.npy', self.one_hot_labels_data)
        print('Saving numpy array for ICDAR 2015 recognition one hot labels of shape:', self.one_hot_labels_data.shape)

    @staticmethod
    def recognition_dataset_generator(cropped_images, one_hot_labels):
        for i, cropped_image in enumerate(cropped_images):
            sequenced_image = np.zeros((one_hot_labels[i].shape[0], *cropped_image.shape), dtype=np.float64)
            sequenced_image[:] = cropped_image
            yield sequenced_image, one_hot_labels[i]

    def load_recognition_dataset(self):
        # Check if saved before loading
        os.path.isdir('../np_data')
        os.path.isfile('../np_data/icdar15_recognition_images.npy')
        os.path.isfile('../np_data/icdar15_recognition_transcripts.npy')
        os.path.isfile('../np_data/icdar15_recognition_one_hot_labels.npy')
        os.path.isfile('../np_data/icdar15_recognition_index_transcriptions.npy')

        try:
            self.cropped_images_data = np.load('../np_data/icdar15_recognition_images.npy')
            print('Loaded ICDAR 2015 recognition images data')

            self.transcription_data = np.load('../np_data/icdar15_recognition_transcripts.npy')
            print('Loaded ICDAR 2015 recognition transcripts data')

            char_set = sorted(set(''.join(self.transcription_data)))
            self.char_to_index = {c: i + 1 for i, c in enumerate(char_set)}

            self.one_hot_labels_data = np.load('../np_data/icdar15_recognition_one_hot_labels.npy')
            print('Loaded ICDAR 2015 recognition one hot labels data')

            self.index_transcriptions = np.load('../np_data/icdar15_recognition_index_transcriptions.npy')
            print('Loaded ICDAR 2015 recognition index transcriptions data')

        except FileNotFoundError:
            return None, None, None

        return self.cropped_images_data, self.transcription_data, self.index_transcriptions

    def visualise_data_batch(self):
        # create a 5x4 grid of subplots
        fig, axes = plt.subplots(num='Text Recognition Data Batch', nrows=5, ncols=4, figsize=(10, 10))

        # loop through each subplot and display an image with its label
        for i, ax in enumerate(axes.flat):
            # get the image and label for this subplot
            img = self.cropped_images_data[i]
            label = self.transcription_data[i]

            # display the image and label
            ax.imshow(img)
            ax.set_title(str(label))
            ax.axis('off')

        # adjust the layout and display the plot
        plt.tight_layout()
        plt.show()

    def decode_recognition_label(self, one_hot_label):
        index_to_char = {idx: char for char, idx in self.char_to_index.items()}

        def decode_one_hot_label(one_hot_label):
            return np.argmax(one_hot_label, axis=1)

        decoded_indices = decode_one_hot_label(one_hot_label)
        decoded_transcriptions = [''.join([index_to_char[index] for index in decoded_indices if index != 0])]

        return decoded_transcriptions
