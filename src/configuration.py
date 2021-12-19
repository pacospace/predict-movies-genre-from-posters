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

"""Configuration class."""

import pathlib


class Configuration:
    """Configuration file for the project."""

    # General config
    ROOT = pathlib.Path.cwd()
    CONFIG_FOLDER = ROOT
    CHECKPOINT_FOLDER = ROOT / "models"
    DATA_FOLDER = ROOT / "data"
    EXTERNAL_DATA_FOLDER = DATA_FOLDER / "external"
    PROCCESED_DATA_FOLDER = DATA_FOLDER / "processed"

    # Dataset config
    BASE_IMAGE_FOLDER = PROCCESED_DATA_FOLDER / "Images"
    CSV_ANNOTATION_FILE = PROCCESED_DATA_FOLDER / "train.csv"

    # Training config
    SEED = 42
    INPUT_IMAGE_SIZE = (400, 400, 3)
    TB_LOGS = CHECKPOINT_FOLDER / "logs/scalars/"
