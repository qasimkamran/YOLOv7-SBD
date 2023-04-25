import argparse

from keras.callbacks import TensorBoard

import datasets
import losses
import tf_models
import tensorflow as tf
import numpy as np
import datetime


def train_east():
    # EAST_model.load_weights('east_saved/saved_model.h5')

    EAST_model = tf_models.EAST().model
    ICDAR15_data = datasets.ICDAR15()

    images_data, score_maps_data, geo_maps_data = ICDAR15_data.load_detection_dataset()
    if images_data is None or score_maps_data is None or geo_maps_data is None:
        ICDAR15_data.save_detection_dataset()
        images_data, score_maps_data, geo_maps_data = ICDAR15_data.load_detection_dataset()

    train_x = images_data[0:770]
    train_y1 = score_maps_data[0:770]
    train_y2 = geo_maps_data[0:770]

    val_x = images_data[770:]
    val_y1 = score_maps_data[770:]
    val_y2 = geo_maps_data[770:]

    log_dir = 'east_saved/log'

    # Create the TensorBoard callback
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=1)

    EAST_model.compile(optimizer=tf.optimizers.Adam(),
                       loss=[losses.score_loss, losses.rbox_loss],
                       run_eagerly=True)

    EAST_model.fit(train_x, [train_y1, train_y2],
                   validation_data=(val_x, [val_y1, val_y2]),
                   validation_batch_size=3,
                   batch_size=3,
                   epochs=20,
                   callbacks=[tensorboard_callback])

    EAST_model.save('new_east_saved')

    EAST_model.save_weights('new_east_saved/saved_model.h5')


def train_ocr():
    OCR_model = tf_models.SimpleOCR().model
    ICDAR15_data = datasets.ICDAR15()

    cropped_images_data, transcription_data, one_hot_labels_data = ICDAR15_data.load_recognition_dataset()
    if cropped_images_data is None or transcription_data is None or one_hot_labels_data is None:
        ICDAR15_data.save_recognition_dataset()
        cropped_images_data, transcription_data, one_hot_labels_data = ICDAR15_data.load_recognition_dataset()

    train_x = cropped_images_data[0:1334]
    train_y = one_hot_labels_data[0:1334]

    val_x = cropped_images_data[1335:1906]
    val_y = one_hot_labels_data[1335:1906]

    # Compile the model
    OCR_model.compile(optimizer=tf.keras.optimizers.RMSprop(0.0001),
                      loss=tf.keras.losses.CategoricalCrossentropy())

    # Train the model
    OCR_model.fit(train_x, train_y,
                  validation_data=(val_x, val_y),
                  batch_size=8,
                  epochs=10)

    OCR_model.save('ocr_saved')

    OCR_model.save_weights('ocr_saved/saved_model.h5')


if __name__ == '__main__':
    train_east()
    # train_ocr()
