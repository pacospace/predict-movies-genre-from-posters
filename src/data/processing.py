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

"""Processing code."""

import tqdm
import numpy as np

from .base_component import BaseComponent
from .data_handler import AnnotationsData
from .data_handler import ImagesData
from src.configuration import Configuration

from sklearn.model_selection import train_test_split
from keras.preprocessing import image


class Processing(BaseComponent):
    """Base class for processing methods."""

    @staticmethod
    def process_image(img):
        """Process image."""
        # Converts a PIL Image instance to a Numpy array.
        img = image.img_to_array(img)

        # Normalize image
        img = img/255

        return img

    @staticmethod
    def process_dataset():
        """Process data."""
        annotations_df = AnnotationsData.retrieve_annotations_dataframe()
        train_image = []

        for i in tqdm(range(annotations_df.shape[0])):
            traid_id = annotations_df.iloc[i]["Id"]
            image = ImagesData.retrieve_image(traid_id=traid_id)
            processed_image = Processing.process_image(image)
            train_image.append(processed_image)

        X = np.array(train_image)
        y = np.array(annotations_df.drop(['Id', 'Genre'],axis=1))

        return X, y
