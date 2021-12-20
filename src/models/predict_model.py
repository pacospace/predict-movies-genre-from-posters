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

"""Inference model code."""

import tensorflow as tf
import numpy as np

from src.configuration import Configuration
from src.data.processing import Processing


class Inference:
    """Base class for inference methods."""

    def __init__(self, model_name: str) -> None:
        """Load model once when app starts."""
        loaded_model = tf.keras.models.load_model(
            f"{Configuration.CHECKPOINT_FOLDER}/{model_name}", compile=False
        )

        self.model = loaded_model
        self.model_version = model_name

    def predict(self, image, classes):
        """Make prediction using ML model."""
        # Default is TensorFlow model
        processed_image = Processing.process_image(image)
        prediction = self.model.predict(np.array([processed_image]))
        print(prediction)
        top_3 = np.argsort(prediction[0])[:-4:-1]
        print(top_3)

        for i in range(3):
            print(
                "{}".format(classes[top_3[i]])
                + " ({:.3})".format(prediction[0][top_3[i]])
            )
