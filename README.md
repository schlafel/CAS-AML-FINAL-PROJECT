# Isolated Sign Language Recognition
#### CAS Advance Machine Learning - Uni Bern

This project aims to classify isolated American Sign Language (ASL) signs using deep learning techniques implemented in PyTorch. The dataset used for this project is provided by Google's "Isolated Sign Language Recognition" competition on Kaggle.

## About the Dataset
The dataset consists of videos of sign language gestures made by individuals. The videos were processed using MediaPipe's holistic model to extract landmark data for the face, left and right hands, and body pose. Each sequence of landmarks is labeled with the corresponding sign.
the dataset can be downloaded from [Kaggle](https://www.kaggle.com/competitions/asl-signs/data)

The goal of this project is to develop a model that can accurately classify the sign gestures from the landmark data.

## Project Structure
The project is structured as follows:

* checkpoints/: contains saved model chechpoints weights and optimizer states.
* data/ \t : contains the raw data downloaded from Kaggle, as well as preprocessed data.
* src/        : contains the source code for the project.
* runs/       : contains the tensorboard logs of the results.
* notebooks/  : contains Jupyter notebooks for data exploration, model training, and visualization.

The project path looks as follows:
```
project
├── checkpoints
│   ├── pytorch
│   │   ├───...
│   └── tensorflow
│       ├───...
├── data
│   ├── processed
│   └── raw
├── notebooks
├── runs
│   ├── pytorch
│   │   ├───...
│   └── tensorflow
│       ├───...
├── src
│   ├── config.py
│   ├── dl_utils.py
│   ├── metrics.py
│   ├── predict_on_camera.py
│   ├── trainer.py
│   ├── tb_logger.py
│   ├── visualizations.py
│   ├── data
│   │   ├── data_utils.py
│   │   ├───dataset.py
│   └── models
│         ├── pytorch
│         │     └─── models.py
│         └── tensorflow
│               └─── models.py
├── README.md
├── License.md
└── conda_env.yaml
```


## Installation
To install the required dependencies for this project, run the following command:

```
conda env create -f conda_env.yaml
```
OR
```
pip install -r requirements.txt`
```

## Usage
To preprocess the data and train the model, run the following command:

Make sure to edit the `src/config.py` file first! 

`python src/trainer.py`


The complete functional documentation on the code can be found under [this link](https://github.com/schlafel/CAS-AML-FINAL-PROJECT/raw/working/docs/build/latex/americansignlanguagerecognition.pdf). 


To evaluate the trained model on the test set, run the following command:

`python src/evaluate.py --config config/eval_config.yaml`

## Results
The results are summarized in the [project report](). 

## Future Work
In the future, the model could be further improved by incorporating additional features, such as hand motion trajectories, or by training on a larger dataset. The model could also be deployed in a mobile app to help parents learn sign language and communicate with their deaf children.

## Contributors
Asad Bin Imtiaz
Felix Schlatter

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/schlafel/CAS-AML-FINAL-PROJECT/blob/working/LICENCE.md) file for details.

## Acknowledgments
- Kaggle for providing the dataset
- MediaPipe for their landmark extraction model
- PyTorch for their deep learning framework
