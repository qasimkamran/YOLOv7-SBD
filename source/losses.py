import tensorflow as tf
import numpy as np
import math

import models


def rescale_coords(x1, y1, x2, y2, scale_factor_x, scale_factor_y):
    x1 = int(round(x1 * scale_factor_x))
    y1 = int(round(y1 * scale_factor_y))
    x2 = int(round(x2 * scale_factor_x))
    y2 = int(round(y2 * scale_factor_y))
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
            converted[i, y1:y2, x1:x2] = 1.0
    return converted


def gt_boxes_to_rbox_map(boxes, pred_size, true_size):
    pred_width, pred_height = pred_size
    true_width, true_height = true_size
    converted = np.zeros((boxes.shape[0], pred_width, pred_height, 5))
    for i in range(boxes.shape[0]):
        for j in range(20):
            if np.all(boxes[i, j] == 0):
                break
            x1, y1, x2, y2 = boxes[i, j]
            scale_factor_x = pred_width / true_width
            scale_factor_y = pred_height / true_height
            x1, y1, x2, y2 = rescale_coords(x1, y1, x2, y2, scale_factor_x, scale_factor_y)
            cx, cy, w, h, theta = bbox_to_rbox(boxes[i, j])
            x1, y1, x2, y2 = fit_coords_in_bounds([x1, y1, x2, y2],
                                                  [pred_width, pred_height, pred_width, pred_height])
            converted[i, y1, x1] = [cx, cy, w, h, theta]
            converted[i, y2, x2] = [cx, cy, w, h, theta]
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


def score_loss(y_true, y_pred):
    y_true_np = y_true.numpy()
    y_pred_np = y_pred.numpy()

    y_true_score = gt_boxes_to_score_map(y_true_np, [y_pred_np.shape[1], y_pred_np.shape[1]], [models.INPUT_SIZE, models.INPUT_SIZE])
    y_true_score_tensor = tf.convert_to_tensor(y_true_score)
    y_true_score_tensor = tf.cast(y_true_score_tensor, tf.float32)

    loss = - (y_true_score_tensor * tf.math.log(y_pred + 1e-10) + (1 - y_true_score_tensor) * tf.math.log(1 - y_pred + 1e-10))
    loss = tf.reduce_mean(loss, axis=[0, 1, 2])  # average over batch size and other dimensions

    return loss


def rbox_loss(y_true, y_pred):
    y_true_np = y_true.numpy()
    y_pred_np = y_pred.numpy()

    y_true_rbox = gt_boxes_to_rbox_map(y_true_np, [y_pred_np.shape[1], y_pred_np.shape[1]], [models.INPUT_SIZE, models.INPUT_SIZE])
    y_true_rbox_tensor = tf.convert_to_tensor(y_true_rbox)
    y_true_rbox_tensor = tf.cast(y_true_rbox_tensor, tf.float32)

    y_true_cx, y_true_cy, y_true_h, y_true_w, y_true_angle = tf.split(y_true_rbox_tensor, [1, 1, 1, 1, 1], axis=-1)
    y_true_cx, y_true_cy, y_pred_h, y_pred_w, y_pred_angle = tf.split(y_pred, [1, 1, 1, 1, 1], axis=-1)

    # Compute the sin and cos of the angle parameters
    y_true_sin = tf.sin(y_true_angle)
    y_true_cos = tf.cos(y_true_angle)
    y_pred_sin = tf.sin(y_pred_angle)
    y_pred_cos = tf.cos(y_pred_angle)

    # Compute the smooth L1 loss for the angle parameter
    angle_loss = tf.reduce_mean(tf.compat.v1.losses.huber_loss(y_true_sin, y_pred_sin) +
                                tf.compat.v1.losses.huber_loss(y_true_cos, y_pred_cos))

    # Compute the smooth L1 loss for the height and width parameters
    hw_loss = tf.reduce_mean(tf.compat.v1.losses.huber_loss(y_true_h, y_pred_h) +
                             tf.compat.v1.losses.huber_loss(y_true_w, y_pred_w))

    # Compute the smooth L1 loss for the geometric parameter
    y_true_geo = tf.concat([y_true_sin, y_true_cos, y_true_h, y_true_w], axis=-1)
    y_pred_geo = tf.concat([y_pred_sin, y_pred_cos, y_pred_h, y_pred_w], axis=-1)
    geo_loss = tf.reduce_mean(tf.compat.v1.losses.huber_loss(y_true_geo, y_pred_geo))

    # Compute the total RBOX map loss
    rbox_map_loss = angle_loss + hw_loss + geo_loss

    return rbox_map_loss
