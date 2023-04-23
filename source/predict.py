import argparse
import os

import cv2
import lanms
import numpy as np

import datasets
import tf_models
import matplotlib.pyplot as plt
import time
import icdar


def plot_feature_maps(model, layer_name, image):
    # Extract the feature maps
    feature_maps_func = K.function([model.layers[0].input], [model.get_layer(layer_name).output])
    feature_maps = feature_maps_func([image])[0]

    # Plot the feature maps
    fig, axs = plt.subplots(nrows=feature_maps.shape[-1], figsize=(8, 8))
    for i in range(feature_maps.shape[-1]):
        axs[i].imshow(feature_maps[0, :, :, i], cmap='gray')
        axs[i].axis('off')
    plt.suptitle(layer_name)
    plt.show()


def display_feature_maps(model, layer_names, image):
    # Display the feature maps for the specified layer names
    for layer_name in layer_names:
        plot_feature_maps(model, layer_name, image)


def predict_east(img):
    EAST_model = tf_models.EAST().model

    EAST_model.load_weights('east_saved/saved_model.h5')

    score_map, geo_map = EAST_model.predict(img)

    score_map_thresh = 0.8
    nms_thresh = 0.2
    box_thresh = 0.3

    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]

    # filter the score map and sort the text boxes via the y-axis
    xy_text = np.argwhere(score_map > score_map_thresh)
    xy_text = xy_text[np.argsort(xy_text[:, 0])]

    # restore
    text_box_restored = icdar.restore_rectangle(xy_text[:, ::-1] * 4, geo_map[xy_text[:, 0], xy_text[:, 1], :])  # N*4*2
    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]

    # nms part
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thresh)
    print('{} text boxes after nms'.format(boxes.shape[0]))

    if boxes.shape[0] == 0:
        return None

    # here we filter some low score boxes by the average score map, this is different from the original paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]
    print('{} text boxes after filtering'.format(boxes.shape[0]))

    if boxes is not None:
        boxes = boxes[:, :8].reshape((-1, 4, 2))

    return boxes


def plot_east_prediction(img, boxes):
    # save to file
    if boxes is not None:
        for box in boxes:
            # to avoid submitting errors
            box = icdar.sort_poly(box.astype(np.int32))
            cv2.polylines(img, [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=5)
    cv2.imshow('Prediction', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize_image(im, max_side_len=1280):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def test_predict(img, score_map, geo_map):
    score_map_thresh = 0.8
    nms_thresh = 0.2
    box_thresh = 0.0

    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]

    # filter the score map and sort the text boxes via the y-axis
    xy_text = np.argwhere(score_map > score_map_thresh)
    xy_text = xy_text[np.argsort(xy_text[:, 0])]

    # restore
    text_box_restored = icdar.restore_rectangle(xy_text[:, ::-2] * 4, geo_map[xy_text[:, 0], xy_text[:, 1], :])  # N*4*2
    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    reshaped_score_map = score_map[xy_text[:, 0], xy_text[:, 1]].reshape((-1))
    boxes[:, 8] = reshaped_score_map

    # nms part
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thresh)
    print('{} text boxes after nms'.format(boxes.shape[0]))

    if boxes.shape[0] == 0:
        return None

    if boxes is not None:
        boxes = boxes[:, :8].reshape((-1, 4, 2))
        boxes[:, :, 0] /= 0.3
        boxes[:, :, 0] += 210

    if boxes is not None:
        for box in boxes:
            # to avoid submitting errors
            box = icdar.sort_poly(box.astype(np.int32))
            print(box.astype(np.int32).reshape((-1, 1, 2)))
            cv2.polylines(img, [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 255), thickness=1)
    cv2.imshow('Prediction', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    img = cv2.imread('../misc_images/pump')
    img = cv2.imread('../train_data/img_100_2013.jpg')
    img = cv2.resize(img, (512, 512))
    img_compatible = np.expand_dims(img, axis=0)
    boxes = predict_east(img_compatible)
    # img = cv2.resize(img, (128, 128))
    plot_east_prediction(img, boxes)
