import argparse
import datasets
import losses
import nets
import tensorflow as tf
import numpy as np


def train_east():
    # EAST_model.load_weights('east_saved/saved_model.h5')

    EAST_model = nets.EAST().model
    ICDAR15_data = datasets.ICDAR15()

    images_data, score_maps_data, rbox_maps_data = ICDAR15_data.load_detection_dataset()
    if images_data is None or score_maps_data is None or rbox_maps_data is None:
        ICDAR15_data.save_detection_dataset()
        images_data, score_maps_data, rbox_maps_data = ICDAR15_data.load_detection_dataset()

    train_x = images_data[0:1050]
    train_y1 = score_maps_data[0:1050]
    train_y2 = rbox_maps_data[0:1050]

    val_x = images_data[1051:1500]
    val_y1 = score_maps_data[1051:1500]
    val_y2 = rbox_maps_data[1051:1500]

    EAST_model.compile(optimizer=tf.optimizers.Adam(),
                       loss=[losses.score_loss, losses.rbox_loss],
                       run_eagerly=True)

    EAST_model.fit(train_x, [train_y1, train_y2],
                   validation_data=(val_x, [val_y1, val_y2]),
                   batch_size=3,
                   epochs=15)

    EAST_model.save('east_saved')

    EAST_model.save_weights('east_saved/saved_model.h5')


def train_ocr():
    OCR_model = nets.SimpleOCR().model
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
    OCR_model.compile(optimizer=tf.keras.optimizers.Adam(0.00001),
                      loss=tf.keras.losses.CategoricalCrossentropy())

    # Train the model
    OCR_model.fit(train_x, train_y,
                  validation_data=(val_x, val_y),
                  batch_size=3,
                  epochs=15)


if __name__ == '__main__':
    # train_east()
    train_ocr()
