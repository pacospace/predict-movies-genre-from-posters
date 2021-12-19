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

"""Visualizations code."""

import matplotlib.pyplot as plt
import numpy as np

from .base_component import BaseComponent


class Visualization(BaseComponent):
    """Base class for visualization methods."""

    @staticmethod
    def visualize_image(image_array):
        """Visualize image."""
        plt.imshow(image_array)
        plt.show()

    @staticmethod
    def visualize_training_history(training_history):
        """Visualize training history."""
        print("Average test loss: ", np.average(training_history.history["loss"]))

        # list all data in history
        # print(training_history.history.keys())

        # summarize history for accuracy
        plt.plot(training_history.history["accuracy"])
        plt.plot(training_history.history["val_accuracy"])
        plt.title("model accuracy")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")
        plt.show()

        # summarize history for loss
        plt.plot(training_history.history["loss"])
        plt.plot(training_history.history["val_loss"])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")
        plt.show()
