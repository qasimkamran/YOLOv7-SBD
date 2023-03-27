import datasets
import losses
import models
import tensorflow as tf


def train_east():
    EAST_model = models.EAST().model
    ICDAR13_data = datasets.ICDAR13()

    images_data, score_boxes, rbox_boxes = ICDAR13_data.load_dataset()
    if images_data is None or score_boxes is None or rbox_boxes is None:
        ICDAR13_data.save_dataset()
        images_data, score_boxes, rbox_boxes = ICDAR13_data.load_dataset()

    train_x = images_data[0:160]
    train_y1 = score_boxes[0:160]
    train_y2 = rbox_boxes[0:160]

    val_x = images_data[161:]
    val_y1 = score_boxes[161:]
    val_y2 = rbox_boxes[161:]

    EAST_model.compile(optimizer=tf.optimizers.Adam(),
                       loss=[losses.score_loss, losses.rbox_loss],
                       run_eagerly=True)
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
    EAST_model.load_weights('east_saved/saved_model.h5')

    EAST_model.fit(train_x, [train_y1, train_y2],
                   validation_data=(val_x, [val_y1, val_y2]),
                   batch_size=3,
                   epochs=20)

    EAST_model.save('east_saved')

    EAST_model.save_weights('east_saved/saved_model.h5')


if __name__ == '__main__':
    train_east()
