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

import pandas as pd

from .base_component import BaseComponent
from src.configuration import Configuration
from keras.preprocessing import image


class ImagesData(BaseComponent):
    """Base class for images data handling."""

    @staticmethod
    def retrieve_image(traid_id: str):
        """Retrieve image file."""
        # Get a A PIL Image instance.
        img = image.load_img(
            str(Configuration.BASE_IMAGE_FOLDER) + "/" + traid_id + ".jpg",
            target_size=Configuration.INPUT_IMAGE_SIZE,
        )
        return img


class AnnotationsData(BaseComponent):
    """Base class for annotations data handling."""

    @staticmethod
    def retrieve_annotations_dataframe():
        """Retrieve annotations file."""
        annotations_df = pd.read_csv(
            Configuration.CSV_ANNOTATION_FILE
        )  # reading the csv file
        return annotations_df
