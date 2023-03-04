import keras
import tensorflow as tf
from keras import layers, models
from keras import backend as K
import numpy as np
import cv2
import bbox


class EAST():
    model = None

    def __init__(self, img):
        self.model = models.Sequential()
        self.predict(img)
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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.IMAGE_SIZE)
        img = img.astype(np.float32) / 255.0
        return img

    def fcn(self, input):
        input_tensor = tf.keras.Input(shape=input.shape)

        """Define the FCN layers."""
        conv1 = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        pool1 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv1)
        conv2 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(pool1)
        pool2 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv2)
        conv3 = layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(pool2)
        pool3 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv3)
        conv4 = layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(pool3)
        pool4 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv4)
        conv5 = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(pool4)
        output = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(conv5)
        return output

    def rpn(self, input_tensor):
        """Define the RPN layers."""
        conv1 = layers.Conv2D(self.RPN_FILTERS, self.RPN_KERNEL_SIZE, padding=self.RPN_PADDING,
                              activation='relu')(input_tensor)
        conv2 = layers.Conv2D(self.RPN_FILTERS, self.RPN_KERNEL_SIZE, padding=self.RPN_PADDING,
                              activation='relu')(conv1)
        conv3 = layers.Conv2D(self.RPN_FILTERS, self.RPN_KERNEL_SIZE, padding=self.RPN_PADDING,
                              activation='relu')(conv2)
        conv4 = layers.Conv2D(self.RPN_FILTERS, self.RPN_KERNEL_SIZE, padding=self.RPN_PADDING,
                              activation='relu')(conv3)

        # Compute the RPN box scores and reshape the tensor to have 2 channels for each anchor
        rpn_box_scores = layers.Conv2D(2 * self.RPN_BOXES_PER_LOCATION, (1, 1),
                                       padding=self.BINARY_PADDING,
                                       name='rpn_box_scores')(conv4)
        rpn_box_scores = layers.Reshape((-1, K.int_shape(rpn_box_scores)[2],
                                         self.RPN_BOXES_PER_LOCATION, 2))(rpn_box_scores)
        rpn_box_scores = layers.Permute((1, 2, 4, 3))(rpn_box_scores)

        # Compute the RPN box coordinates and reshape the tensor to have 4 channels for each anchor
        rpn_box_coords = layers.Conv2D(4 * self.RPN_BOXES_PER_LOCATION, (1, 1), padding=self.BINARY_PADDING,
                                       name='rpn_box_coords')(conv4)
        rpn_box_coords = layers.Reshape((-1, K.int_shape(rpn_box_coords)[2],
                                         self.RPN_BOXES_PER_LOCATION, 4))(rpn_box_coords)
        rpn_box_coords = layers.Permute((1, 2, 4, 3))(rpn_box_coords)

        # Compute the anchor boxes and reshape the tensor to have 4 channels for each anchor
        # anchor_boxes = layers.Lambda(lambda x: K.constant(self.generate_anchor_boxes(), dtype=tf.float32))(input_tensor)
        # anchor_boxes = layers.Reshape((1, 1, self.RPN_NUM_ANCHORS, 4))(anchor_boxes)
        # anchor_boxes = layers.Lambda(
        #    lambda x: K.tile(x, (K.shape(input_tensor)[0], K.shape(input_tensor)[1], K.shape(input_tensor)[2], 1, 1)))(anchor_boxes)

        feature_map_height = K.int_shape(input_tensor)[1]
        feature_map_width = K.int_shape(input_tensor)[2]

        anchor_boxes = self.generate_anchor_boxes()
        anchor_boxes_tensor = K.variable(anchor_boxes, dtype='float32')
        tile_multiples = (self.RPN_NUM_ANCHORS, 1, 1)
        # anchor_boxes = anchor_boxes.reshape(feature_map_height * feature_map_width * self.RPN_NUM_ANCHORS, 4)
        anchor_boxes_tensor = K.variable(anchor_boxes, dtype='float32')
        anchor_boxes_tensor = K.reshape(anchor_boxes, (feature_map_height * feature_map_width * self.RPN_NUM_ANCHORS, 4))
        anchor_boxes_tensor = K.tile(anchor_boxes_tensor, (K.shape(input_tensor)[0], 1, 1, 1, 1))

        # Compute the proposal boxes and their scores
        rpn_proposal_boxes, rpn_proposal_scores = self.proposal_layer(rpn_box_coords, rpn_box_scores, anchor_boxes_tensor)

        return rpn_proposal_boxes, rpn_proposal_scores

    def generate_anchor_boxes(self):
        """Generate a set of anchor boxes for each pixel in the feature map."""
        anchor_boxes = []
        for scale in self.SCALES:
            for aspect_ratio in self.RATIOS:
                width = scale * np.sqrt(aspect_ratio)
                height = scale / np.sqrt(aspect_ratio)
                x1 = y1 = -width / 2
                x2 = y2 = width / 2
                anchor_boxes.append([x1, y1, x2, y2])
        return np.array(anchor_boxes)

    def proposal_layer(self, rpn_box_coords, rpn_box_scores, anchor_boxes):
        """Compute the proposal boxes and their scores."""
        # Compute the box coordinates and clip them to the image boundaries
        proposals = self.bbox_transform(anchor_boxes, rpn_box_coords)
        proposals = self.clip_boxes(proposals, tf.shape(rpn_box_scores)[1:3] * self.FEATURE_STRIDES)

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

    def bbox_transform(self, anchor_boxes, rpn_box_coords):
        """Transforms anchor boxes to proposal boxes using RPN box coordinates.

        Args:
            anchor_boxes: numpy array of shape (num_anchors, 4) containing the coordinates
                of the anchor boxes in the format (x1, y1, x2, y2).
            rpn_box_coords: numpy array of shape (num_anchors, 4) containing the coordinates
                of the RPN box coordinates in the format (dx, dy, dw, dh).

        Returns:
            proposals: numpy array of shape (num_anchors, 4) containing the coordinates
                of the proposal boxes in the format (x1, y1, x2, y2).
        """
        x_center = (anchor_boxes[:, 2] + anchor_boxes[:, 0]) / 2
        y_center = (anchor_boxes[:, 3] + anchor_boxes[:, 1]) / 2
        width = anchor_boxes[:, 2] - anchor_boxes[:, 0]
        height = anchor_boxes[:, 3] - anchor_boxes[:, 1]

        # Calculate the proposals' coordinates
        proposals_x_center = x_center + rpn_box_coords[:, 0] * width
        proposals_y_center = y_center + rpn_box_coords[:, 1] * height
        proposals_width = width * np.exp(rpn_box_coords[:, 2])
        proposals_height = height * np.exp(rpn_box_coords[:, 3])

        # Calculate the proposals' coordinates
        proposals_x1 = proposals_x_center - proposals_width / 2
        proposals_y1 = proposals_y_center - proposals_height / 2
        proposals_x2 = proposals_x_center + proposals_width / 2
        proposals_y2 = proposals_y_center + proposals_height / 2

        proposals = np.stack([proposals_x1, proposals_y1, proposals_x2, proposals_y2], axis=1)

        return proposals

    def clip_boxes(self, boxes, image_shape):
        """
        Clips boxes to image boundaries.
        Args:
            boxes: numpy array of shape [N, 4] containing the coordinates of N boxes
            image_shape: tuple or list of length 2 or 3, containing the height and width of the image
        Returns:
            A numpy array of the same shape as boxes, with the coordinates clipped to the image boundaries
        """
        boxes = np.asarray(boxes)
        image_shape = np.asarray(image_shape)

        # Compute the minimum and maximum values for each dimension
        ymin = np.maximum(boxes[:, 0], 0)
        xmin = np.maximum(boxes[:, 1], 0)
        ymax = np.minimum(boxes[:, 2], image_shape[0])
        xmax = np.minimum(boxes[:, 3], image_shape[1])

        # Combine the values to form the new boxes
        clipped_boxes = np.stack([ymin, xmin, ymax, xmax], axis=-1)

        return clipped_boxes

    def predict(self, img):
        # Run inference on a single image.
        img = self.preprocess(img)
        fcn_output = self.fcn(img)
        boxes, scores = self.rpn(fcn_output)

        # Loop over the boxes and scores
        for box, score in zip(boxes, scores):
            # Extract the coordinates of the box
            x1, y1, x2, y2 = box

            # Draw the box on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add the score to the box
            cv2.putText(img, str(score), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Show the image
        cv2.imshow('Prediction', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()