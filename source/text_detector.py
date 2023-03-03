import keras
import tensorflow as tf
from keras import layers
import numpy as np
import cv2


class EAST():

    def __init__(self):
        pass

    # Define the image size
    IMAGE_SIZE = (512, 512)

    # Define the RPN parameters
    RPN_KERNEL_SIZE = (3, 3)
    RPN_PADDING = 'same'
    RPN_FILTERS = 128
    RPN_CHANNELS = 512
    RPN_BOXES_PER_LOCATION = 9
    RPN_NUM_ANCHORS = RPN_BOXES_PER_LOCATION * 2

    # Define the binary classifier parameters
    BINARY_KERNEL_SIZE = (1, 1)
    BINARY_PADDING = 'valid'
    BINARY_FILTERS = 64

    # Define the non-maximum suppression (NMS) parameters
    NMS_THRESH = 0.4
    NMS_MAX_BOXES = 100

    # Anchor box settings
    RATIOS = [0.5, 1, 2]
    SCALES = [128, 256, 512]
    FEATURE_STRIDES = [4, 8, 16, 32, 64]

    # Proposal generation settings
    RPN_PRE_NMS_TOP_N = 1000
    RPN_POST_NMS_TOP_N = 300
    RPN_NMS_THRESH = 0.7
    RPN_BOX_SCORE_THRESH = 0.5

    # Box transformation settings
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2], dtype=np.float32)

    def preprocess(self, img):
        """Preprocess the input image by resizing and normalizing the pixel values."""
        img = cv2.resize(img, self.IMAGE_SIZE)
        img = img.astype(np.float32) / 255.0
        return img

    def define_model(self):
        model = keras.Sequential()


    def fcn(self, input):
        """Define the FCN layers."""
        conv1 = layers.Conv2D(input, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
        pool1 = layers.MaxPool2D(conv1, pool_size=(2, 2), strides=(2, 2))
        conv2 = layers.Conv2D(pool1, filters=64, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
        pool2 = layers.MaxPool2D(conv2, pool_size=(2, 2), strides=(2, 2))
        conv3 = layers.Conv2D(pool2, filters=128, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
        pool3 = layers.MaxPool2D(conv3, pool_size=(2, 2), strides=(2, 2))
        conv4 = layers.Conv2D(pool3, filters=256, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
        pool4 = layers.MaxPool2D(conv4, pool_size=(2, 2), strides=(2, 2))
        conv5 = layers.Conv2D(pool4, filters=512, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
        output = layers.Conv2D(conv5, filters=512, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
        return output

    def rpn(self, input):
        """Define the RPN layers."""
        conv1 = layers.Conv2D(input, filters=self.RPN_FILTERS, kernel_size=self.RPN_KERNEL_SIZE, padding=self.RPN_PADDING,
                                 activation=tf.nn.relu)
        conv2 = layers.Conv2D(conv1, filters=self.RPN_FILTERS, kernel_size=self.RPN_KERNEL_SIZE, padding=self.RPN_PADDING,
                                 activation=tf.nn.relu)
        conv3 = layers.Conv2D(conv2, filters=self.RPN_FILTERS, kernel_size=self.RPN_KERNEL_SIZE, padding=self.RPN_PADDING,
                                 activation=tf.nn.relu)
        conv4 = layers.Conv2D(conv3, filters=self.RPN_FILTERS, kernel_size=self.RPN_KERNEL_SIZE, padding=self.RPN_PADDING,
                                 activation=tf.nn.relu)

        # Compute the RPN box scores and reshape the tensor to have 2 channels for each anchor
        rpn_box_scores = layers.Conv2D(conv4, filters=self.RPN_CHANNELS * self.RPN_BOXES_PER_LOCATION, kernel_size=(1, 1),
                                          padding=self.BINARY_PADDING)
        rpn_box_scores = tf.reshape(rpn_box_scores, [-1, tf.shape(rpn_box_scores)[1], tf.shape(rpn_box_scores)[2],
                                                     self.RPN_BOXES_PER_LOCATION, self.RPN_CHANNELS])
        rpn_box_scores = tf.transpose(rpn_box_scores, [0, 1, 2, 4, 3])

        # Compute the RPN box coordinates and reshape the tensor to have 4 channels for each anchor
        rpn_box_coords = layers.Conv2D(conv4, filters=self.RPN_CHANNELS * 4, kernel_size=(1, 1), padding=self.BINARY_PADDING)
        rpn_box_coords = tf.reshape(rpn_box_coords, [-1, tf.shape(rpn_box_coords)[1], tf.shape(rpn_box_coords)[2],
                                                     self.RPN_BOXES_PER_LOCATION, 4])
        rpn_box_coords = tf.transpose(rpn_box_coords, [0, 1, 2, 4, 3])

        # Compute the anchor boxes and reshape the tensor to have 4 channels for each anchor
        anchor_boxes = tf.constant(self.generate_anchor_boxes(), dtype=tf.float32)
        anchor_boxes = tf.reshape(anchor_boxes, [1, 1, 1, self.RPN_NUM_ANCHORS, 4])
        anchor_boxes = tf.tile(anchor_boxes, [tf.shape(input)[0], tf.shape(input)[1], tf.shape(input)[2], 1, 1])

        # Compute the proposal boxes and their scores
        rpn_proposal_boxes, rpn_proposal_scores = self.proposal_layer(rpn_box_coords, rpn_box_scores, anchor_boxes)

        return rpn_proposal_boxes, rpn_proposal_scores

    def generate_anchor_boxes(self):
        """Generate a set of anchor boxes for each pixel in the feature map."""
        anchor_boxes = []
        for scale in self.RPN_ANCHOR_SCALES:
            for aspect_ratio in self.RPN_ASPECT_RATIOS:
                width = scale * np.sqrt(aspect_ratio)
                height = scale / np.sqrt(aspect_ratio)
                x1 = -width / 2
                y1 = -height / 2
                x2 = width / 2
                y2 = height / 2
                anchor_boxes.append([x1, y1, x2, y2])
        return np.array(anchor_boxes)

    def proposal_layer(self, rpn_box_coords, rpn_box_scores, anchor_boxes):
        """Compute the proposal boxes and their scores."""
        # Compute the box coordinates and clip them to the image boundaries
        proposals = box_utils.bbox_transform(anchor_boxes, rpn_box_coords)
        proposals = box_utils.clip_boxes(proposals, tf.shape(rpn_box_scores)[1:3] * self.FEATURE_STRIDES)

        # Flatten the proposal boxes and their scores
        proposals = tf.reshape(proposals, [-1, 4])
        rpn_box_scores = tf.reshape(rpn_box_scores, [-1, self.RPN_BOXES_PER_LOCATION])

        # Filter out the proposals with a low score
        keep_indices = tf.where(tf.greater(rpn_box_scores[:, 1], self.RPN_BOX_SCORE_THRESH))[:, 0]
        proposals = tf.gather(proposals, keep_indices)
        rpn_box_scores = tf.gather(rpn_box_scores, keep_indices)

        # Sort the proposals by their score and select the top N
        sorted_indices = tf.argsort(rpn_box_scores[:, 1], direction="DESCENDING")
        top_indices = tf.slice(sorted_indices, [0], [self.RPN_PRE_NMS_TOP_N])
        top_proposals = tf.gather(proposals, top_indices)
        top_scores = tf.gather(rpn_box_scores, top_indices)

        # Apply non-maximum suppression to the top proposals
        nms_indices = tf.image.non_max_suppression(top_proposals, top_scores[:, 1], max_output_size=self.RPN_POST_NMS_TOP_N,
                                                   iou_threshold=self.RPN_NMS_THRESH)
        rpn_proposal_boxes = tf.gather(top_proposals, nms_indices)
        rpn_proposal_scores = tf.gather(top_scores, nms_indices)

        return rpn_proposal_boxes, rpn_proposal_scores


