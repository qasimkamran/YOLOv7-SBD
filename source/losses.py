import tensorflow as tf
import numpy as np
import math
import cv2
from fontTools.misc.arrayTools import pointInRect

import models
from scipy.ndimage import distance_transform_edt
from skimage import measure


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
            converted[i, x1:x2, y1:y2] = 1.0
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
            cx, cy, w, h, theta = bbox_to_rbox([x1, y1, x2, y2])
            x1, y1, x2, y2 = fit_coords_in_bounds([x1, y1, x2, y2],
                                                  [pred_width, pred_height, pred_width, pred_height])
            converted[i, x1, y1] = [cx, cy, w, h, theta]
            converted[i, x2, y2] = [cx, cy, w, h, theta]
    return converted


def tmp(boxes, pred_size, true_size):
    batch_size = boxes.shape[0]
    pred_width, pred_height = pred_size
    true_width, true_height = true_size

    rboxes = np.zeros((batch_size, 20, 5))

    for i in range(batch_size):
        for j, box in enumerate(boxes[i]):
            x1, y1, x2, y2 = box
            scale_factor_x = pred_width / true_width
            scale_factor_y = pred_height / true_height
            x1, y1, x2, y2 = rescale_coords(x1, y1, x2, y2, scale_factor_x, scale_factor_y)
            rboxes[i, j] = bbox_to_rbox([x1, y1, x2, y2])

    overlayed_rboxes = np.ones((batch_size, pred_width, pred_height))

    for i in range(batch_size):
        for j, rbox in enumerate(rboxes[i]):
            x, y, w, h, angle = rboxes[i, j]
            rbox = cv2.boxPoints(((x, y), (w, h), angle))
            rbox = np.int0(rbox)
            cv2.drawContours(overlayed_rboxes[i], [rbox], 0, (0, 0, 0), 1)

    ones_converted = np.zeros((batch_size, pred_width, pred_height, 5))

    for i, box in enumerate(overlayed_rboxes):
        # Get the indices of all 0s in the image
        zero_indices = np.argwhere(box == 0)

        # Compute the distance transform of the image
        distance_transform = distance_transform_edt(box)

        # Initialize the output arrays
        distances = np.zeros((box.shape[0], box.shape[1], 4), dtype=np.float32)
        angles = np.zeros_like(box, dtype=np.float32)

        # Loop over all pixels in the image
        for j in range(box.shape[0]):
            for k in range(box.shape[1]):
                if box[j, k] == 1:
                    # Compute the distance to the closest 0 in each direction
                    distances[j, k, 0] = distance_transform[j, :k][::-1].argmin() + 1 if k > 0 else 0
                    distances[j, k, 1] = distance_transform[:j, k][::-1].argmin() + 1 if j > 0 else 0
                    distances[j, k, 2] = distance_transform[j, k + 1:].argmin() + 1 if k < box.shape[1] - 1 else 0
                    distances[j, k, 3] = distance_transform[j + 1:, k].argmin() + 1 if j < box.shape[0] - 1 else 0

                    # Compute the angle to the closest 0
                    x, y = zero_indices[
                        np.argmin(np.sqrt((j - zero_indices[:, 0]) ** 2 + (k - zero_indices[:, 1]) ** 2))]
                    angles[j, k] = np.arctan2(y - k, x - j)

        ones_converted[i] = np.concatenate([distances, angles[..., None]], axis=-1)

    return ones_converted


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

    y_true_score = gt_boxes_to_score_map(y_true_np, [y_pred_np.shape[1], y_pred_np.shape[1]],
                                         [models.INPUT_SIZE, models.INPUT_SIZE])
    y_true_score_tensor = tf.convert_to_tensor(y_true_score)
    y_true_score_tensor = tf.cast(y_true_score_tensor, tf.float32)
    '''
    img = np.zeros((128, 128, 3))

    # Overlay predicted boxes on the original image
    for i in range(y_pred_np.shape[0]):
        for j in range(y_pred_np.shape[1]):
            for k in range(y_pred_np.shape[2]):
                if np.any(y_pred_np[i, j, k] > 0.5):  # Only draw boxes with score > 0.5
                    cv2.circle(img, (j, k), 1, (0, 0, 255), -1)
    # Display the image with overlays
    cv2.imshow('Overlay', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    loss = - (y_true_score_tensor * tf.math.log(y_pred + 1e-10) + (1 - y_true_score_tensor) * tf.math.log(
        1 - y_pred + 1e-10))
    loss = tf.reduce_mean(loss, axis=[0, 1, 2])  # average over batch size and other dimensions

    return loss


def rbox_loss(y_true, y_pred):
    y_true_np = y_true.numpy()
    y_pred_np = y_pred.numpy()

    y_true_rbox_tensor = tf.convert_to_tensor(y_true_np)
    y_true_rbox_tensor = tf.cast(y_true_rbox_tensor, tf.float32)
    '''
    img = np.zeros((128, 128, 3))

    for i in range(y_true_rbox.shape[0]):
        for j in range(y_true_rbox.shape[1]):
            for k in range(y_true_rbox.shape[2]):
                print(y_true_rbox[i, j, k])
                x, y, w, h, angle = y_true_rbox[i, j, k]
                box = cv2.boxPoints(((x, y), (w, h), angle))
                box = np.int0(box)
                cv2.drawContours(img, [box], 0, (0, 255, 0), 1)
    # Display the image with overlays
    cv2.imshow('Overlay', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
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
