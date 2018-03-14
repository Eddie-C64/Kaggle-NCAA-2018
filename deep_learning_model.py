import argparse
import sys

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape
from keras.layers.convolutional import Conv1D, Conv2D

from ncaa_predict.data_loader import load_data_multiyear, N_FEATURES, N_Players

from ncaa_predict.util import list_arg

DEFAULT_BATCH_SIZE = 1000
DEFAULT_STEPS = sys.maxsize


def run_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", "-b", default=DEFAULT_BATCH_SIZE, type=int,
        help="The training batch size. Smaller numbers will train faster but "
             "may not converge. (default: %(default)s)")
    parser.add_argument(
        "--model-out", "-o", default=None,
        help="File to save the model to. This will be an entire Keras model, "
             "which can be loaded and used without needing to keep track of the "
             "architecture. Warning: Keras will overwrite existing models. "
             "(default: don't save)")
    parser.add_argument(
        "--steps", "-s", default=DEFAULT_STEPS, type=int,
        help="The maximum number of training steps. Note that you can stop "
             "training at any time and save the output with ctrl+c. (default: "
             "%(default)s)")
    parser.add_argument(
        "--train-years", "-y", default=list(range(2002, 2017)),
        type=list_arg(type=int, container=frozenset),
        help="A comma-separated list of years to train on.")

    args = parser.parse_args()

    model = Sequential([
        Conv2D(10, (2, N_Players), strides=(1, N_Players), activation='relu',
               input_shape=(2, N_Players, N_FEATURES)),

        Flatten(),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(16, activation='relu'),
        Dense(2, activation='softmax')
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adagrad',
        metrics=['accuracy']
    )

    features, labels = load_data_multiyear(args.train_years)

    try:
        model.fit()

    except KeyboardInterrupt:
        print("Stopped training due to keyboard interrupt")
    if args.model_out is not None:
        model.save(args.model_out)