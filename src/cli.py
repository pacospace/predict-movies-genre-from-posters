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

import click
import logging

from sklearn.model_selection import train_test_split
from src.configuration import Configuration

from src.data.data_handler import AnnotationsData
from src.data.data_handler import ImagesData
from src.data.processing import Processing
from src.visualization import Visualization
from src.models.train_model import Training


@click.group()
@click.pass_context
def cli(ctx: click.Context):
    """Main command line interface."""
    logger = logging.getLogger(__name__)
    logger.info("Run Predict Movies Genre From Poster Images.")


@cli.command("visualize")
def visualize() -> None:
    """Visualize data."""
    annotations_df = AnnotationsData.retrieve_annotations_dataframe()
    click.echo(annotations_df.head())

    traid_id = annotations_df.iloc[0]["Id"]
    image = ImagesData.retrieve_image(traid_id=traid_id)
    processed_image = Processing.process_image(image)
    Visualization.visualize_image(image_array=processed_image)

    training = Training()
    training.show_model()


@cli.command("training")
@click.option(
    "--epochs",
    type=int,
    help="Set epochs for training.",
)
def train(epochs) -> None:
    """Train model."""
    X, y = Processing.process_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        random_state=Configuration.SEED,
        test_size=0.1
    )
    training = Training()
    training.train_model(
        X_train,
        X_test,
        y_train,
        y_test,
        epochs=epochs,
        batch_size=64
    )




if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    cli()
