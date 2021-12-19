#!/usr/bin/env python3
#
# Copyright(C) 2021 Francesco Murdaca
#
# This program is free software: you can redistribute it and / or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""Training model code."""

from datetime import datetime
import random
import keras
import tensorflow as tf
from tqdm.keras import TqdmCallback

from src.configuration import Configuration

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


class Training:
    """Base class for training methods."""

    NUMBER_OF_CLASSES = 25

    def __init__(self) -> None:
        """init."""
        model = Sequential()
        model.add(
            Conv2D(
                filters=16,
                kernel_size=(5, 5),
                activation="relu",
                input_shape=Configuration.INPUT_IMAGE_SIZE,
            )
        )
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(filters=32, kernel_size=(5, 5), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(self.NUMBER_OF_CLASSES, activation="sigmoid"))

        # MNIST dataset parameters.
        self.cnn_model = model

    def show_model(self):
        """Show moodel summary."""
        self.cnn_model.summary()

    def train_model(
        self, x_train, y_train, x_test, y_test, epochs, batch_size, verbose
    ):
        """Train model."""
        logdir = (
            str(Configuration.TB_LOGS) + "/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

        self.cnn_model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        print("Start fitting model...")
        training_history = self.cnn_model.fit(
            x_train,
            y_train,
            epochs=epochs,
            validation_data=(x_test, y_test),
            batch_size=batch_size,
            verbose=0,
            callbacks=[tensorboard_callback, TqdmCallback(verbose=2)],
        )

        return training_history

    def save_model(self, prefix: str):
        """Save trained model."""
        time_version = (
            f"{prefix}-{datetime.now():%y%m%d%H%M%S}-{random.getrandbits(64):08x}"
        )

        tf.keras.models.save_model(
            self.cnn_model,
            f"{Configuration.CHECKPOINT_FOLDER}/{time_version}/",
            overwrite=True,
            include_optimizer=True,
            save_format=None,
            signatures=None,
            options=None,
        )
