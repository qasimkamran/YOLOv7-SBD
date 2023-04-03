import cv2
import numpy as np
import models
import matplotlib.pyplot as plt


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
    EAST_model = models.EAST().model

    EAST_model.load_weights('east_saved/saved_model.h5')

    return EAST_model.predict(img)


def plot_east_prediction(img, score_map, rbox_map):
    # Overlay predicted boxes on the original image
    for i in range(rbox_map.shape[0]):
        for j in range(rbox_map.shape[1]):
            for k in range(rbox_map.shape[2]):
                if np.any(score_map[i, j, k] > 0.01):  # Only draw boxes with score > 0.5
                    print(score_map[i, j, k])
                    cv2.circle(img, (j, k), 1, (0, 0, 255), -1)
                    '''
                    x, y, h, w, angle = rbox_map[i, j, k]
                    box = cv2.boxPoints(((x, y), (h, w), angle))
                    box = np.int0(box)
                    cv2.drawContours(img, [box], 0, (0, 255, 0), 1)
                    '''
    # Display the image with overlays
    cv2.imshow('Overlay', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    img = cv2.imread('../misc_images/pump')
    img = np.load('east_saved/train_image.npy')
    img = cv2.resize(img, (512, 512))
    img_compatible = np.expand_dims(img, axis=0)
    score_map, rbox_map = predict_east(img_compatible)
    img = cv2.resize(img, (128, 128))
    plot_east_prediction(img, score_map, rbox_map)
