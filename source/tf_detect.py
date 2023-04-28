import cv2
import lanms
import matplotlib.pyplot as plt
import numpy as np
from imutils.object_detection import non_max_suppression
import datasets
import icdar
import tf_models


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

    EAST_model.load_weights('new_east_saved/saved_model.h5')

    score_map, geo_map = EAST_model.predict(img)

    return img, score_map, geo_map

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
    nms_thresh = 0.8
    box_thresh = 0.1

    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]

    # filter the score map and sort the text boxes via the y-axis
    xy_text = np.argwhere(score_map > score_map_thresh)
    xy_text = xy_text[np.argsort(xy_text[:, 0])]

    print(geo_map[xy_text[:, 0], xy_text[:, 1], :].shape)
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
        boxes[:, :, 0] += 128
        boxes[:, :, 0] += 128

    if boxes is not None:
        for box in boxes:
            # to avoid submitting errors
            box = icdar.sort_poly(box.astype(np.int32))
            print(box.astype(np.int32).reshape((-1, 1, 2)))
            cv2.polylines(img, [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 255), thickness=1)
    print(img.shape)
    cv2.imshow('Prediction', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def predict_ocr(img):
    OCR_model = tf_models.SimpleOCR().model
    ICDAR_data = datasets.ICDAR15()
    images, transcripts, labels = ICDAR_data.load_recognition_dataset()

    OCR_model.load_weights('ocr_saved/saved_model.h5')

    one_hot_label = OCR_model.predict(img)

    for i in enumerate(one_hot_label[0]):
        for j in enumerate(one_hot_label[0, i[0]]):
            print(one_hot_label[0, i[0], j[0]])
            one_hot_label[0, i[0], j[0]] = int(one_hot_label[0, i[0], j[0]])

    transcript = ICDAR_data.decode_recognition_label(one_hot_label[0])

    print(transcript)


def show_score(img, score_map):
    # Overlay predicted boxes on the original image
    for j in range(score_map.shape[0]):
        for k in range(score_map.shape[1]):
            if np.any(score_map[j, k] > 0.5):  # Only draw boxes with score > 0.5
                cv2.circle(img, (k, j), 1, (0, 255, 255), -1)
    # Display the image with overlays
    cv2.imshow('Overlay', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def predict_east_cv2(img):
    net = cv2.dnn.readNet('new_east_saved/frozen_east_text_detection.pb')

    orig = img
    (H, W) = img.shape[:2]

    (newW, newH) = (640, 320)
    rW = W / float(newW)
    rH = H / float(newH)

    image = cv2.resize(img, (newW, newH))
    (H, W) = image.shape[:2]

    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    blob = cv2.dnn.blobFromImage(img, 1.0, (W, H),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)

    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):

        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < 0.5:
                continue

            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    boxes = non_max_suppression(np.array(rects), probs=confidences)

    for (startX, startY, endX, endY) in boxes:
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        # draw the bounding box on the image
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 3)
    cv2.imshow("Text Detection", orig)
    k = cv2.waitKey(0) & 0xff
    cv2.destroyAllWindows()


if __name__ == '__main__':
    img = cv2.imread('icdar_composite/img_101_2013.jpg')
    predict_east_cv2(img)
