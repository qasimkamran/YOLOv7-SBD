import tensorflow as tf
import numpy as np
import math
import cv2
from fontTools.misc.arrayTools import pointInRect

import nets
from scipy.ndimage import distance_transform_edt
from skimage import measure


def rescale_coords(x1, y1, x2, y2, scale_factor_x, scale_factor_y):
    x1 = int(round(x1 * scale_factor_x))
    y1 = int(round(y1 * scale_factor_y))
    x2 = int(round(x2 * scale_factor_x))
    y2 = int(round(y2 * scale_factor_y))
    return x1, y1, x2, y2


def rescale_coords_tf(x1, y1, x2, y2, scale_factor_x, scale_factor_y):
    x1 = tf.cast(tf.round(x1 * scale_factor_x), dtype=tf.int32)
    y1 = tf.cast(tf.round(y1 * scale_factor_y), dtype=tf.int32)
    x2 = tf.cast(tf.round(x2 * scale_factor_x), dtype=tf.int32)
    y2 = tf.cast(tf.round(y2 * scale_factor_y), dtype=tf.int32)
    return x1, y1, x2, y2


def fit_coords_in_bounds(coords, bounds):
    assert len(coords) == len(bounds), f'Length mismatch is coords and bounds input arrays'
    decrement = False
    for i in range(len(coords)):
        if coords[i] == bounds[i]:
            decrement = True
            break
    for i in range(len(coords)):
        if coords[i] != 0:
            coords[i] -= 1
    return coords


def gt_boxes_to_score_map(boxes, pred_size, true_size):
    pred_width, pred_height = pred_size
    true_width, true_height = true_size
    converted = np.zeros((boxes.shape[0], pred_width, pred_height))
    for i in range(boxes.shape[0]):
        for j in range(boxes.shape[1]):
            if np.all(boxes[i, j] == 0):
                break
            x1, y1, x2, y2 = boxes[i, j]
            scale_factor_x = pred_width / true_width
            scale_factor_y = pred_height / true_height
            x1, y1, x2, y2 = rescale_coords(x1, y1, x2, y2, scale_factor_x, scale_factor_y)
            converted[i, x1:x2, y1:y2] = 1.0
    return converted


def gt_boxes_to_score_map_tf(boxes, pred_size, true_size):
    pred_width, pred_height = pred_size
    true_width, true_height = true_size
    input = tf.zeros((tf.shape(boxes)[0], pred_width, pred_height))
    converted = tf.zeros((tf.shape(boxes)[0], pred_width, pred_height))
    for i in range(tf.shape(boxes)[0]):
        for j in range(tf.shape(boxes)[1]):
            if tf.reduce_all(tf.equal(boxes[i, j], 0)):
                break
            x1, y1, x2, y2 = boxes[i, j]
            scale_factor_x = tf.cast(pred_width / true_width, dtype=tf.float32)
            scale_factor_y = tf.cast(pred_height / true_height, dtype=tf.float32)
            x1, y1, x2, y2 = rescale_coords_tf(x1, y1, x2, y2, scale_factor_x, scale_factor_y)
            rows = tf.range(x1, x2, dtype=tf.int32)
            cols = tf.range(y1, y2, dtype=tf.int32)
            indices = tf.stack(tf.meshgrid(rows, cols, indexing='ij'), axis=-1)
            indices = tf.reshape(indices, shape=(-1, 2))
            values = tf.ones((x2 - x1) * (y2 - y1), dtype=tf.float32)
            updated_image = tf.tensor_scatter_nd_update(input[i], indices, values)
            converted = tf.tensor_scatter_nd_update(converted, [[i]], tf.expand_dims(updated_image, axis=0))
    return converted


def bbox_to_rbox(bbox):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    theta = math.atan2(y2 - y1, x2 - x1)
    rbox = [cx, cy, w, h, theta]
    return rbox


def bbox_to_rbox_tf(bbox):
    x1, y1, x2, y2 = tf.unstack(bbox, axis=0)
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    theta = tf.math.atan2(y2 - y1, x2 - x1)
    rbox = tf.stack([cx, cy, w, h, theta])
    return rbox


def score_loss(y_true, y_pred):
    y_true_np = y_true.numpy()
    y_pred_np = y_pred.numpy()

    '''
    y_true_score = gt_boxes_to_score_map_tf(y_true, [y_pred_np.shape[1], y_pred_np.shape[1]], [models.INPUT_SIZE, models.INPUT_SIZE])
    y_true_score_tensor = tf.convert_to_tensor(y_true_score)
    y_true_score_tensor = tf.cast(y_true_score_tensor, tf.float32)
    
    img = np.zeros((128, 128, 3))

    # Overlay predicted boxes on the original image
    for i in range(y_pred_np.shape[0]):
        for j in range(y_pred_np.shape[1]):
            for k in range(y_pred_np.shape[2]):
                if np.any(y_true_np[i, j, k] > 0.5):  # Only draw boxes with score > 0.5
                    cv2.circle(img, (j, k), 1, (255, 255, 255), -1)
    # Display the image with overlays
    cv2.imshow('Overlay', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    loss = - (y_true * tf.math.log(y_pred + 1e-10) + (1 - y_true) * tf.math.log(
        1 - y_pred + 1e-10))

    loss = tf.reduce_mean(loss, axis=[0, 1, 2])  # average over batch size and other dimensions

    return loss


def rbox_loss(y_true, y_pred):
    y_true_np = y_true.numpy()
    y_pred_np = y_pred.numpy()

    d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(y_pred, [1, 1, 1, 1, 1], axis=-1)

    d1_true, d2_true, d3_true, d4_true, theta_true = tf.split(y_true, [1, 1, 1, 1, 1], axis=-1)
    '''
    img = np.zeros((128, 128, 3))

    for i in range(y_true_np.shape[0]):
        for j in range(y_true_np.shape[1]):
            for k in range(y_true_np.shape[2]):
                # print(y_true_np[i, j, k])
                if y_true_np[i, j, k].all() == 0:
                    continue
                x, y, w, h, angle = y_true_np[i, j, k]
                box = cv2.boxPoints(((x, y), (w, h), angle))
                box = np.int0(box)
                cv2.drawContours(img, [box], 0, (0, 255, 0), 1)
    # Display the image with overlays
    cv2.imshow('Overlay', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    area_true = (d1_true + d3_true) * (d2_true + d4_true)
    area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)

    w_union = tf.minimum(d2_true, d2_pred) + tf.minimum(d4_true, d4_pred)
    h_union = tf.minimum(d1_true, d1_pred) + tf.minimum(d3_true, d3_pred)

    intersection_area = w_union * h_union
    union_area = area_true + area_pred - intersection_area

    AABB_loss = -tf.math.log((intersection_area + 1.0) / (union_area + 1.0))
    theta_loss = 1 - tf.cos(theta_pred - theta_true)
    loss = AABB_loss + 20 * theta_loss
    loss = tf.reduce_mean(loss)

    return loss
