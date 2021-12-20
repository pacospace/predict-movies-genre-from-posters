Predict movies genre from posters images
=========================================

Goal
----

The aim is to predict the genre of a movie using just its poster image. It is a multi-label image classification problem: A movie can belong to more than one genre. It doesn’t just have to belong to one category, like action or comedy. The movie can be a combination of two or more genres.

Dataset
-------

The dataset for Movie Genre Classification based on Poster Images with Deep Neural Networks is available [here](https://www.cs.ccu.edu.tw/~wtchu/projects/MoviePoster/index.html).

The dataset in zip files contains the poster images of several multi-genre movies. It is in a structured format, i.e. a folder containing the images and a .csv file for true labels.

Visualize
---------

In order to visualize annotation dataset and model architecture.

```
pipenv run python3 src/cli.py visualize
```

Training
-------

In order to train the model:

```
pipenv run python3 src/cli.py train
```

In order to see tensorboard results created, you have to run the following command from another terminal:

```
pipenv run tensorboard --logdir models/logs/scalars
```


Inference
-------

In order to make prediction with a model:

```
pipenv run python3 src/cli.py predict --model-name <MODEL_NAME> --image-name <IMAGE_NAME>
```

MODEL_NAME: Name of the model to be loaded. Model needs to be available in `models/` folder.

IMAGE_NAME: Name of the image used for predictions. Image needs to be available in `data/external/` folder.