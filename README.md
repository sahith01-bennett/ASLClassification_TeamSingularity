# American Sign Language Classification with Machine Learning

## Introduction
This project develops a machine learning model to classify American Sign Language (ASL) hand symbols, aimed at facilitating communication for individuals using ASL.

## Requirements
- Kernal : PyTorch-2.0.1 in HiperGator
- Libraries: pandas, numpy, PIL, torch, torchvision, tqdm, Scikit-learn, Matplotlib, itertools, Scipy
- GPU support (optional but recommended for training)

## Setup and Installation
1. Clone the repository.
2. Ensure CUDA is available for GPU usage (if applicable).

## Dataset Description and Path Setup
The dataset consists of images and labels representing various ASL hand symbols. Both images and labels are stored in `.npy` format, which are NumPy array files.

### Data Path Configuration
To set up the paths for your NumPy data files:

1. **Data File**: Place your image data file (in `.npy` format) in a directory of your choice (e.g., `data/hand_signs_data.npy`).

2. **Labels File**: Similarly, place your labels file (also in `.npy` format) in the same or a different directory (e.g., `data/hand_signs_labels.npy`).

3. **Specifying Paths in Code**: Modify the paths in the notebook to point to these files. Look for the section in the code where data paths are defined, and update them accordingly.

    Example:
    ```python
    training_data_filepath = 'path/to/your/data/file'  # e.g., 'data/hand_signs_data.npy'
    training_labels_filepath = 'path/to/your/labels/file'  # e.g., 'data/hand_signs_labels.npy'
    ```

## Model Overview
A convolutional neural network (CNN) based on the ResNet architecture is used, chosen for its efficiency in image recognition tasks.

## Training the Model
To train the model:
1. Open the train.ipynb file
2. Load the dataset with the specified paths.
    ```python
    training_data_filepath = 'path/to/your/data/file'  # e.g., 'data/hand_signs_data.npy'
    training_labels_filepath = 'path/to/your/labels/file'  # e.g., 'data/hand_signs_labels.npy'
    ```
3. Apply necessary transformations (resizing, flipping, rotation).
4. Split the dataset into training and testing sets.
5. Before training, make sure to specify file path for saving the model for future use. The path can be set in the code.
```python
SAVE_PATH  = 'path/to/save/model'  # e.g., 'models/trained_asl_model.pth'
```
5. To Train the model, Click on Kernal -> Restart Kernal and Run All Cells


## Testing the Model
To test the model:
1. Open the test.ipynb file
2. Give paths to the testing data
	```python
	test_data_filepath = 'path/to/your/data/file'  # e.g., 'data/hand_signs_data.npy'
    test_labels_filepath = 'path/to/your/labels/file'  # e.g., 'data/hand_signs_labels.npy'
	```
    
3. Give path to the Saved model

	```python
	MODEL_PATH = 'path/to/your/model/file'  # e.g., 'models/trained_asl_model.pth'
	```
    
4. To Test the model, Click on Kernal -> Restart Kernal and Run All Cells

5. Model classifies images from Alphabets A to I and gives -1 for unknown class for the images not present in training(Hard Set)

## Model Evaluation
The model's performance is evaluated using accuracy, F1 score,Precision ,Recall.\

Test Loss: 0.0002, Test Accuracy: 99.26% \
Classification Report:

                | precision  |  recall  | f1-score |  support

           A    |    1.00    |    1.00  |    1.00  |      30
           B    |    1.00    |    1.00  |    1.00  |      30
           C    |    1.00    |    0.97  |    0.98  |      30
           D    |    1.00    |    1.00  |    1.00  |      30
           E    |    1.00    |    1.00  |    1.00  |      30
           F    |    1.00    |    1.00  |    1.00  |      30
           G    |    0.94    |    1.00  |    0.97  |      30
           H    |    1.00    |    0.97  |    0.98  |      30
           I    |    1.00    |    1.00  |    1.00  |      30

    accuracy    |                        0.99       270
    macro avg   |    0.99      0.99      0.99       270
    weighted avg|    0.99      0.99      0.99       270


## Acknowledgements
Special thanks to the contributors and dataset providers.

## License
[Include license information here.]
