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


def rbox_to_bbox(rbox):
    cx, cy, w, h, angle = rbox
    angle = math.radians(angle)
    c, s = np.cos(angle), np.sin(angle)
    # Calculate the four corners of the rbox
    corners = [(cx + lx * c - ly * s, cy + lx * s + ly * c) for lx, ly in [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]]
    # Convert to xmin, ymin, xmax, ymax format
    corners = np.array(corners)
    x1 = np.min(corners[:, 0])
    y1 = np.min(corners[:, 1])
    x2 = np.max(corners[:, 0])
    y2 = np.max(corners[:, 1])
    return np.array([x1, y1, x2, y2])


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


def intersection_over_union(bbox1, bbox2):
    # Calculate the coordinates of the intersection rectangle
    xmin = max(bbox1[0], bbox2[0])
    ymin = max(bbox1[1], bbox2[1])
    xmax = min(bbox1[2], bbox2[2])
    ymax = min(bbox1[3], bbox2[3])
    # If the intersection is empty, return 0
    if xmin >= xmax or ymin >= ymax:
        return 0.0
    # Calculate the areas of the two rectangles and the intersection
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    intersection_area = (xmax - xmin) * (ymax - ymin)
    union_area = area1 + area2 - intersection_area
    # Calculate the IoU
    iou = -math.log(intersection_area + 1.0 / union_area + 1.0)
    return iou


def huber_losses(y_true, y_pred, batch_size, delta=1.0):
    huber_losses = []
    huber_loss_fn = tf.keras.losses.Huber()
    for batch_idx in range(batch_size):
        pixel_batch = y_pred[:, batch_idx]
        gt_batch = y_true[:, batch_idx]
        losses = []
        for d, dimension in enumerate(tf.unstack(pixel_batch)):
            dimension_losses = []
            for pixel in tf.unstack(dimension):
                pixel_losses = []
                for gt_coord in gt_batch[d]:
                    loss = huber_loss_fn(gt_coord, dimension)
                    pixel_losses.append(loss)
                min_loss = tf.math.reduce_min(pixel_losses)
                dimension_losses.append(min_loss)
            huber_losses.append(dimension_losses)
    return tf.stack(huber_losses)  # Shape: (batch_size, 128, 128)


def single_dim_loss(dim_true, dim_pred):
    # compute the difference between each pixel's 1 value and each pixel's 20 values
    diff = tf.abs(dim_pred - dim_true)

    # take the minimum difference for each pixel
    min_diff = tf.reduce_min(diff, axis=-1)

    # compute the mean loss across all pixels
    loss = tf.reduce_mean(min_diff, axis=[0, 1, 2])

    return loss


def single_angle_loss(theta_true, theta_pred):
    # compute the difference between each pixel's 1 value and each pixel's 20 values
    diff = 1 - tf.cos(theta_pred - theta_true)

    # take the minimum difference for each pixel
    min_diff = tf.reduce_min(diff, axis=-1)

    # compute the mean loss across all pixels
    loss = tf.reduce_mean(min_diff, axis=[0, 1, 2])

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

    return tf.reduce_mean(loss)

    raise Exception

    d1_true_reshaped = tf.reshape(d1_true, [d1_true.shape[0], d1_true.shape[1]])
    d1_true_tiled = tf.tile(tf.expand_dims(d1_true_reshaped, axis=1), [1, y_pred.shape[1] * y_pred.shape[2], 1])
    d1_true_output = tf.reshape(d1_true_tiled, [d1_true.shape[0], y_pred.shape[1], y_pred.shape[2], d1_true.shape[1]])

    d2_true_reshaped = tf.reshape(d2_true, [d2_true.shape[0], d2_true.shape[1]])
    d2_true_tiled = tf.tile(tf.expand_dims(d2_true_reshaped, axis=1), [1, y_pred.shape[1] * y_pred.shape[2], 1])
    d2_true_output = tf.reshape(d2_true_tiled, [d2_true.shape[0], y_pred.shape[1], y_pred.shape[2], d2_true.shape[1]])

    d3_true_reshaped = tf.reshape(d3_true, [d3_true.shape[0], d3_true.shape[1]])
    d3_true_tiled = tf.tile(tf.expand_dims(d3_true_reshaped, axis=1), [1, y_pred.shape[1] * y_pred.shape[2], 1])
    d3_true_output = tf.reshape(d3_true_tiled, [d3_true.shape[0], y_pred.shape[1], y_pred.shape[2], d3_true.shape[1]])

    d4_true_reshaped = tf.reshape(d4_true, [d4_true.shape[0], d4_true.shape[1]])
    d4_true_tiled = tf.tile(tf.expand_dims(d4_true_reshaped, axis=1), [1, y_pred.shape[1] * y_pred.shape[2], 1])
    d4_true_output = tf.reshape(d4_true_tiled, [d4_true.shape[0], y_pred.shape[1], y_pred.shape[2], d4_true.shape[1]])
    '''
    theta_true_reshaped = tf.reshape(theta_true, [theta_true.shape[0], theta_true.shape[1]])
    theta_true_tiled = tf.tile(tf.expand_dims(theta_true_reshaped, axis=1), [1, theta_pred.shape[1] * theta_pred.shape[2], 1])
    theta_true_output = tf.reshape(theta_true_tiled, [theta_true.shape[0], theta_pred.shape[1], theta_pred.shape[2], theta_true.shape[1]])
    theta_loss = single_angle_loss(theta_true_output, theta_pred)

    inter_xmin = tf.maximum(x1_pred, x1_true_output)
    inter_xmin = tf.reduce_min(inter_xmin, axis=-1)
    inter_ymin = tf.maximum(y1_pred, y1_true_output)
    inter_ymin = tf.reduce_min(inter_ymin, axis=-1)
    inter_xmax = tf.minimum(x2_pred, x2_true_output)
    inter_xmax = tf.reduce_min(inter_xmax, axis=-1)
    inter_ymax = tf.minimum(y2_pred, y2_true_output)
    inter_ymax = tf.reduce_min(inter_ymax, axis=-1)

    inter_area = tf.abs((inter_xmax - inter_xmin) * (inter_ymax - inter_ymin))
    pred_area = tf.abs((d1_pred + d3_pred) * (d2_pred + d4_pred))

    # Calculate center points of the top and bottom edges of the bounding box
    center_top = tf.stack([(x1_true_output + x2_true_output) / 2, y1_true_output], axis=-1)
    center_bottom = tf.stack([(x1_true_output + x2_true_output) / 2, y2_true_output], axis=-1)

    center_top = tf.where(tf.not_equal(center_top, 0), center_top, tf.ones_like(center_top) * 1e9)
    center_bottom = tf.where(tf.not_equal(center_bottom, 0), center_bottom, tf.ones_like(center_bottom) * 1e9)

    # Calculate slope and y-intercept of the text orientation line
    slope = (center_bottom[-2] - center_top[-2]) / (center_bottom[-1] - center_top[-1])
    print(slope)

    y_intercept = center_top[:, 1] - slope * center_top[:, 0]

    # Calculate intersection points between text orientation line and each edge of the bounding box
    x_left = (y1_true_output - y_intercept) / slope
    x_right = (y2_true_output - y_intercept) / slope
    y_top = slope * x1_true_output + y_intercept
    y_bottom = slope * x2_true_output + y_intercept

    # Calculate distances d1_true_output, d2_true_output, d3_true_output, and d4_true_output using the distance formula
    d1_true = tf.sqrt(tf.pow(x_left - x1_true_output, 2) + tf.pow(y1_true_output - y_top, 2))
    d2_true = tf.sqrt(tf.pow(x2_true_output - x_right, 2) + tf.pow(y1_true_output - y_top, 2))
    d3_true = tf.sqrt(tf.pow(x2_true_output - x_right, 2) + tf.pow(y2_true_output - y_bottom, 2))
    d4_true = tf.sqrt(tf.pow(x_left - x1_true_output, 2) + tf.pow(y2_true_output - y_bottom, 2))
    
    print(d1_true)
    '''
    area_pred = tf.abs((d1_pred + d3_pred) * (d2_pred + d4_pred))

    area_true = tf.abs((d1_true_output + d3_true_output) * (d2_true_output + d4_true_output))
    area_non_zero = tf.where(tf.not_equal(area_true, 0), area_true, tf.ones_like(area_true) * 1e9)
    sub_area = tf.abs(area_non_zero - area_pred)
    area_min = tf.reduce_min(sub_area, axis=-1)

    d1_min = tf.minimum(d1_true_output, d1_pred)
    d1_min_non_zero = tf.where(tf.not_equal(d1_min, 0), d1_min, tf.ones_like(d1_min) * 1e9)
    d1_min = tf.reduce_min(d1_min_non_zero, axis=-1)

    d2_min = tf.minimum(d2_true_output, d2_pred)
    d2_min_non_zero = tf.where(tf.not_equal(d2_min, 0), d2_min, tf.ones_like(d2_min) * 1e9)
    d2_min = tf.reduce_min(d2_min_non_zero, axis=-1)

    d3_min = tf.minimum(d3_true_output, d3_pred)
    d3_min_non_zero = tf.where(tf.not_equal(d3_min, 0), d3_min, tf.ones_like(d3_min) * 1e9)
    d3_min = tf.reduce_min(d3_min_non_zero, axis=-1)

    d4_min = tf.minimum(d4_true_output, d1_pred)
    d4_min_non_zero = tf.where(tf.not_equal(d4_min, 0), d4_min, tf.ones_like(d4_min) * 1e9)
    d4_min = tf.reduce_min(d4_min_non_zero, axis=-1)

    w_union = d2_min + d4_min
    h_union = d1_min + d3_min

    intersection_area = w_union * h_union

    squeezed_area_pred = tf.squeeze(area_pred, axis=-1)
    union_area = squeezed_area_pred + area_min - intersection_area

    iou = -tf.math.log(intersection_area / union_area)

    iou = tf.where(tf.math.is_inf(iou), 0.0, iou)

    return iou

    return loss

    raise Exception

    area_true = tf.abs((d3_true_output - d1_true_output) * (d4_true_output - d2_true_output))

    area_true = tf.where(tf.not_equal(area_true, 0), area_true, tf.ones_like(area_true) * 1e9)

    union_area = tf.abs(area_pred + area_true)

    intersection_area = tf.abs(area_true - area_pred)
    intersection_area = tf.where(tf.less_equal(intersection_area, y_pred.shape[1] * y_pred.shape[2]), intersection_area, tf.ones_like(intersection_area) * 0)

    iou = intersection_area / union_area
    iou = tf.reduce_max(iou, axis=-1)

    iou_loss = tf.reduce_mean(iou)

    loss = iou_loss + theta_loss

    return loss

    union_area = tf.abs(tf.reduce_min(area_pred + area_true, axis=-1) - inter_area)
    iou = inter_area / union_area

    iou_loss = tf.reduce_mean(iou)

    loss = iou_loss + theta_loss

    raise Exception

    non_zero_iou = tf.where(tf.not_equal(iou, 0), iou, tf.ones_like(iou) * 1e9)

    min_iou = tf.reduce_min(non_zero_iou, axis=3)

    non_zero_min_iou = tf.where(tf.not_equal(min_iou, 1e9), min_iou, tf.ones_like(min_iou) * 0)

    # loss = tf.reduce_mean(min_iou, axis=[0, 1, 2])

    print(non_zero_min_iou)

    raise Exception

    loss = rbox_loss + theta_loss

    return loss

    raise Exception

    return stacked_losses

    raise exception

    per_pixel_losses = np.zeros((y_pred_np.shape[0], y_pred_np.shape[1], y_pred_np.shape[2], 2))

    cumulative_loss = 0
    for i in range(y_pred_np.shape[0]):
        for x in range(y_pred_np.shape[1]):
            for y in range(y_pred_np.shape[2]):
                pred_bbox = rbox_to_bbox(y_pred_np[i, x, y])
                dim_diff = np.zeros(pred_bbox.shape)  # Default value
                final_mean = 0
                saved_index = 0
                for j, true_bbox in enumerate(y_true_np[i]):
                    if true_bbox.all() == 0:
                        continue
                    curr_diff = pred_bbox - true_bbox
                    curr_diff_mean = abs(np.mean(curr_diff))
                    dim_diff_mean = abs(np.mean(dim_diff))
                    if dim_diff_mean > curr_diff_mean or dim_diff_mean == 0:
                        dim_diff = curr_diff
                        final_mean = curr_diff_mean
                        saved_index = j
                # iou_loss = intersection_over_union(pred_bbox, y_true_np[i, saved_index])
                dim_loss = dim_diff
                # pixel_iou_loss = iou_loss
                true_rbox = bbox_to_rbox(y_true_np[i, saved_index])
                pixel_theta_loss = 1 - math.cos(y_pred_np[i, x, y, -1] - true_rbox[-1])
                # pixel_loss = dim_diff + (10 * pixel_theta_loss)
                per_pixel_losses[i, x, y] = [final_mean, pixel_theta_loss]  # dim_loss[0], dim_loss[1], dim_loss[2], dim_loss[3]
                # cumulative_loss += pixel_loss

    per_pixel_losses = tf.convert_to_tensor(per_pixel_losses)

    loss = tf.reduce_mean(per_pixel_losses, axis=[-1])

    # cumulative_loss = tf.abs(tf.reduce_mean(cumulative_loss / (y_pred_np.shape[1] * y_pred_np.shape[2])))

    return per_pixel_losses

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
