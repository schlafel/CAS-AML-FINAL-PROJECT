# Isolated Sign Language Recognition
#### CAS Advance Machine Learning - Uni Bern

This project aims to classify isolated American Sign Language (ASL) signs using deep learning techniques implemented in PyTorch. The dataset used for this project is provided by Google's "Isolated Sign Language Recognition" competition on Kaggle.

## About the Dataset
The dataset consists of videos of sign language gestures made by individuals. The videos were processed using MediaPipe's holistic model to extract landmark data for the face, left and right hands, and body pose. Each sequence of landmarks is labeled with the corresponding sign.
the dataset can be downloaded from [Kaggle](https://www.kaggle.com/competitions/asl-signs/data)

The goal of this project is to develop a model that can accurately classify the sign gestures from the landmark data.

## Project Structure
The project is structured as follows:

checkpoints/: contains saved model chechpoints weights and optimizer states.
data/       : contains the raw data downloaded from Kaggle, as well as preprocessed data.
src/        : contains the source code for the project.
config/     : contains configuration files for training the model.
models/     : contains the definition of the transformer model used for classification.
notebooks/  : contains Jupyter notebooks for data exploration, model training, and visualization.

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

`python src/train.py --config config/train_config.yaml`

To evaluate the trained model on the test set, run the following command:

`python src/evaluate.py --config config/eval_config.yaml`

## Results
Remember: This is all bullshit here as just a placeholder.
The trained model achieves an accuracy of !99%! on the test set, demonstrating its effectiveness in classifying sign language gestures from landmark data.

## Future Work
Remember: This is all bullshit here as just a placeholder.
In the future, the model could be further improved by incorporating additional features, such as hand motion trajectories, or by training on a larger dataset. The model could also be deployed in a mobile app to help parents learn sign language and communicate with their deaf children.

## Contributors
Asad Bin Imtiaz
Felix Schlatter

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Kaggle for providing the dataset
- MediaPipe for their landmark extraction model
- PyTorch for their deep learning framework
