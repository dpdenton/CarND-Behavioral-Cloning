import numpy as np
import ntpath
import csv
import cv2

import sklearn

def _get_image(track_data, source_path):

    filename = ntpath.basename(source_path)
    image_path = "data/{}/IMG/{}".format(track_data, filename)
    return cv2.imread(image_path)


def get_data(samples, flip=True, use_all=True):

    images = []
    measurements = []

    sklearn.utils.shuffle(samples)

    for line in samples:

        image_set, measurement_set = [], []

        image = _get_image(line[-1], line[0])
        image_set.append(image)

        measurement = float(line[3])
        measurement_set.append(measurement)

        # use left and right camera images
        if use_all:
            # create adjusted steering measurements for the side camera images
            correction = 0.1  # this is a parameter to tune
            measurement_left = measurement + correction
            measurement_right = measurement - correction

            image_left = _get_image(line[-1], line[1])
            image_right = _get_image(line[-1], line[2])

            # add images and angles to data set
            image_set.extend([image_left, image_right])
            measurement_set.extend([measurement_left, measurement_right])

        # flip all images
        if flip:
            for idx in range(len(image_set)):
                image_flipped = np.fliplr(image_set[idx])
                image_set.append(image_flipped)
                measurement_flipped = -measurement_set[idx]
                measurement_set.append(measurement_flipped)

        images.extend(image_set)
        measurements.extend(measurement_set)

    return np.array(images), np.array(measurements)


def load_samples(*args):

    samples = []

    for arg in args:
        with open('data/{}/driving_log.csv'.format(arg)) as csvfile:
            reader = csv.reader(csvfile)
            # add path arg to line so we can retrive images later
            # samples += [line for line in reader]
            for line in reader:
                line.append(arg)
                samples.append(line)

    return samples


def generator(samples, batch_size=32, flip=True, use_all=True):

    nb_samples = len(samples)
    sklearn.utils.shuffle(samples)

    while True:

        for offset in range(0, nb_samples, batch_size):

            batch_samples = samples[offset:offset + batch_size]
            images, measurements = [], []

            for line in batch_samples:
                image_set, measurement_set = [], []

                image = _get_image(line[-1], line[0])
                image_set.append(image)

                measurement = float(line[3])
                measurement_set.append(measurement)

                # use left and right camera images
                if use_all:
                    # create adjusted steering measurements for the side camera images
                    correction = 0.25 # this is a parameter to tune
                    measurement_left = measurement + correction
                    measurement_right = measurement - correction

                    image_left = _get_image(line[-1], line[1])
                    image_right = _get_image(line[-1], line[2])

                    # add images and angles to data set
                    image_set.extend([image_left, image_right])
                    measurement_set.extend([measurement_left, measurement_right])

                # flip all images
                if flip:
                    for idx in range(len(image_set)):
                        image_flipped = np.fliplr(image_set[idx])
                        image_set.append(image_flipped)
                        measurement_flipped = -measurement_set[idx]
                        measurement_set.append(measurement_flipped)

                images.extend(image_set)
                measurements.extend(measurement_set)

            yield (np.array(images), np.array(measurements))

if __name__ == "__main__":

    import sys
    import os
    import argparse
    import matplotlib.pyplot as plt

    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Dropout, Activation, Cropping2D

    from drive import load_model

    k_model = load_model('model.h5')

    k_model.summary()

    sys.exit()

    SAVED_MODELS_DIR = 'saved_models'

    parser = argparse.ArgumentParser(description='Remote Driving Model')

    parser.add_argument(
        '--data',
        type=str,
        help='Comma sepearted list of training data labels. Options are "simple", "simple_3lap"'
    )

    parser.add_argument(
        '--model_name',
        type=str,
        default='test_model',
        help='Name of trained model.'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=1,
        help='Number of epochs.'
    )

    args = parser.parse_args()

    model_name = args.model_name
    model_path = "{}/{}".format(SAVED_MODELS_DIR, model_name)
    training_data_labels = args.data.split(',')
    samples = load_samples(*training_data_labels)
    nb_samples = len(samples)
    nb_epochs = args.epochs

    X_train, y_train = get_data(samples)

    # # separate data
    # train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    #
    # # compile and train the model using the generator function
    # train_generator = generator(train_samples, batch_size=32, flip=False, use_all=False)
    # validation_generator = generator(validation_samples, batch_size=32, flip=False, use_all=False)

    model = Sequential()
    # pre-processing layer
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((60, 20), (0, 0)), input_shape=(160, 320, 3)))

    # convolution
    model.add(Conv2D(6, 5, 5, activation="relu"))
    model.add(MaxPooling2D())

    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(1))

    model.compile(loss="mse", optimizer="adam")
    # history = model.fit_generator(
    #     generator=train_generator,
    #     steps_per_epoch=32,
    #     validation_data=validation_generator,
    #     nb_epoch=3,
    #     validation_steps=len(validation_samples),
    # )
    history = model.fit(x=X_train, y=y_train, validation_split=0.2, shuffle=True, epochs=nb_epochs, verbose=1)

    ### plot the training and validation loss for each epoch
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model mean squared error loss for {}'.format(model_name))
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')

    plt.gca().set_position((.1, .3, .8, .6))  # to make a bit of room for extra text

    plt.figtext(.02, .02,
        "Train Loss: {train_loss}\n"
        "Valid Loss: {valid_loss}\n"
        "Training data used: {training_labels}\n"
        "Epochs: {nb_epochs}\n"
        "Samples: {nb_samples}\n".format(
            train_loss=history.history['loss'],
            valid_loss=history.history['val_loss'],
            training_labels=training_data_labels,
            nb_epochs=nb_epochs,
            nb_samples=nb_samples,
        )
    )

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    orig_stdout = sys.stdout
    with open('{}/architecture.txt'.format(model_path), 'w+') as f:
        sys.stdout = f
        print(model.summary())
        sys.stdout = orig_stdout

    with open('{}/notes.txt'.format(model_path), 'a') as f:
        pass

    plt.savefig('{}/summary.pdf'.format(model_path))

    model.save('{}/{}.h5'.format(model_path, model_name))
    print("Saved {}.h5 at {}".format(model_name, model_path))