import datasets
import losses
import models
import tensorflow as tf


def train_east():
    EAST_model = models.EAST().model
    ICDAR13_data = datasets.ICDAR13()

    images_data, boxes_data = ICDAR13_data.load_dataset()
    if images_data is None or boxes_data is None:
        ICDAR13_data.save_dataset()
        images_data, boxes_data = ICDAR13_data.load_dataset()

    EAST_model.compile(optimizer=tf.optimizers.RMSprop(),
                       loss=[losses.score_loss, losses.rbox_loss],
                       run_eagerly=True)

    EAST_model.fit(images_data, boxes_data, batch_size=2)

    EAST_model.save('east_saved')

    EAST_model.save_weights('east_saved')


if __name__ == '__main__':
    train_east()
