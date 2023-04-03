import argparse
import datasets
import losses
import models
import tensorflow as tf
import icdar
import numpy as np


def train_east():
    # ICDAR13_data = datasets.ICDAR13()
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation_data_path', type=str, default='../train_data')
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--geometry', type=str, default='RBOX')
    parser.add_argument('--max_image_large_side', type=int, default=1280)
    parser.add_argument('--max_text_size', type=int, default=800)
    parser.add_argument('--min_text_size', type=int, default=10)
    parser.add_argument('--min_crop_side_ratio', type=float, default=0.1)
    FLAGS = parser.parse_args()
    FLAGS.suppress_warnings_and_error_messages = False
    FLAGS.min_crop_side_ratio = 0.1

    images, score_maps, geo_maps = icdar.load_data(FLAGS)

    np.save('../np_data/icdar15_images.npy', images)
    np.save('../np_data/icdar15_score_maps.npy', score_maps)
    np.save('../np_data/icdar15_geo_maps.npy', geo_maps)
    '''

    '''
    images_data, score_boxes, rbox_boxes = ICDAR13_data.load_dataset()
    if images_data is None or score_boxes is None or rbox_boxes is None:
        ICDAR13_data.save_dataset()
        images_data, score_boxes, rbox_boxes = ICDAR13_data.load_dataset()

    train_x = images_data[0:160]
    train_y = score_boxes[0:160]
    train_y2 = rbox_boxes[0:160]

    val_x = images_data[161:]
    val_y = score_boxes[161:]
    val_y2 = rbox_boxes[161:]
    '''

    '''
    img = images_data[0]
    img = np.expand_dims(img, axis=0)

    box = boxes_data[0]
    box = np.expand_dims(box, axis=0)

    for i in range(20):
        x1, y1, x2, y2 = boxes_data[0][i]
        cv2.rectangle(images_data[0], (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.imshow('Sample', images_data[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    # EAST_model.load_weights('east_saved/saved_model.h5')

    EAST_model = models.EAST().model
    ICDAR15_data = datasets.ICDAR15()

    images_data, score_maps_data, rbox_maps_data = ICDAR15_data.load_dataset()
    if images_data is None or score_maps_data is None or rbox_maps_data is None:
        ICDAR15_data.save_dataset()
        images_data, score_maps_data, rbox_maps_data = ICDAR15_data.load_dataset()

    train_x = images_data[0:300]
    train_y1 = score_maps_data[0:300]
    train_y2 = rbox_maps_data[0:300]

    val_x = images_data[300:600]
    val_y1 = score_maps_data[300:600]
    val_y2 = rbox_maps_data[300:600]

    EAST_model.compile(optimizer=tf.optimizers.Adam(),
                       loss=[losses.score_loss, losses.rbox_loss],
                       run_eagerly=True)

    EAST_model.fit(train_x, [train_y1, train_y2],
                   validation_data=(val_x, [val_y1, val_y2]),
                   batch_size=1,
                   epochs=20)

    EAST_model.save('east_saved')

    EAST_model.save_weights('east_saved/saved_model.h5')


if __name__ == '__main__':
    train_east()
