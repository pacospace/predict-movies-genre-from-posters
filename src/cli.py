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

from src.configuration import Configuration

from src.data.data_handler import AnnotationsData
from src.data.data_handler import ImagesData
from src.data.processing import Processing
from src.visualization import Visualization
from src.models.train_model import Training
from src.models.predict_model import Inference


@click.group()
@click.pass_context
def cli(ctx: click.Context):
    """Run main command line interface."""
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


@cli.command("train")
@click.option(
    "--epochs",
    default=10,
    type=int,
    help="Set epochs for training.",
)
@click.option(
    "--fraction",
    default=0.88,
    type=float,
    help="Fraction to be used to split train/test dataset.",
)
@click.option(
    "--batch-size",
    default=64,
    type=int,
    help="Batch size for training.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    envvar="THOTH_PRESCRIPTIONS_REFRESH_DEBUG",
    help="Be verbose about what's going on.",
)
def train(epochs: int, fraction: float, batch_size: int, verbose: bool = False) -> None:
    """Train model."""
    click.echo(f"Selected {epochs} epochs for training.")
    annotations_df = AnnotationsData.retrieve_annotations_dataframe()
    training_data_df = annotations_df.sample(frac=fraction, random_state=Configuration.SEED)
    testing_data_df = annotations_df.drop(training_data_df.index)

    processing = Processing()
    click.echo("Processing train data...")
    x_train, y_train = processing.process_dataset(df=training_data_df)
    click.echo(f"Inputs Train size: {x_train.shape}")
    click.echo(f"Labels Train size: {y_train.shape}")

    click.echo("Processing test data...")
    x_test, y_test = processing.process_dataset(df=testing_data_df)
    click.echo(f"Inputs Test size: {x_test.shape}")
    click.echo(f"Labels Test size: {y_test.shape}")

    click.echo("Starting training step...")
    training = Training()
    training_history = training.train_model(
        x_train,
        y_train,
        x_test,
        y_test,
        epochs=epochs,
        batch_size=batch_size,
        verbose=int(verbose),
    )

    training.save_model(prefix="tf")
    Visualization.visualize_training_history(training_history)


@cli.command("predict")
@click.option(
    "--model-name",
    type=str,
    required=True,
    help="Model to be used for predictions.",
)
@click.option(
    "--image-name",
    type=str,
    required=True,
    help="Image name input to predict genre.",
)
def predict(model_name: str, image_name: str) -> None:
    """Predict from model."""
    click.echo("Loading Inference model...")
    model = Inference(model_name=model_name)

    click.echo("Retrieving image...")
    annotations_df = AnnotationsData.retrieve_annotations_dataframe()
    classes = [c for c in annotations_df.iloc[0].index if c not in ["Id", "Genre"]]
    click.echo(f"List of genre considered: {classes}")

    image = ImagesData.retrieve_external_image(image_name)
    Visualization.visualize_image(image_array=image)
    click.echo(f"Getting predictions from model {model.model_version}")
    model.predict(image, classes)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    cli()
