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

from tqdm import tqdm
import numpy as np

from .data_handler import ImagesData

from keras.preprocessing import image


class Processing:
    """Base class for processing methods."""

    @staticmethod
    def process_image(img):
        """Process image."""
        # Converts a PIL Image instance to a Numpy array.
        img = image.img_to_array(img)

        # Normalize image
        img = img / 255

        return img

    def process_dataset(self, df):
        """Process data."""
        train_image = []

        number_rows = df.shape[0]

        for i in tqdm(range(number_rows)):
            traid_id = df.iloc[i]["Id"]
            image = ImagesData.retrieve_image(traid_id=traid_id)
            processed_image = self.process_image(image)
            train_image.append(processed_image)

        inputs = np.array(train_image)
        y = np.array(df.drop(["Id", "Genre"], axis=1))

        return inputs, y
