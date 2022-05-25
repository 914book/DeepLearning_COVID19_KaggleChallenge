# DeepLearning_COVID19_KaggleChallenge
## Our model code has primarily been run using the RTX 3080 GPU from Peter’s Desktop computer. 
## The following packages are used in the project code:
### •	Python 3.9.7
### •	Pytorch 1.10.2
### •	NumPy 1.21.5
### •	Torchvision 0.11.3
### •	SciKit-Learn 1.0.2
### •	Pandas 1.4.1
### •	Tqdm 4.63.0
### •	Matplotlib 3.5.1
### •	Nibabel 3.2.2
### •	Cudatoolkit 11.3.1

### Preprocessing.ipynb: 
#### Preprocessing steps are present here. This notebook does not include the conversion to NifTi which was completed in a different notebook on Google Colab, but we are not sure where it went. This file includes splitting the original dataset (which is already converted into NifTi) into the training, validation, and testing sets. It also creates separate .csv files corresponding to each set and saves images into the folders corresponding to training, validation, or testing.
### Final_Project.ipynb: Model code is written here for execution on my on personal computer. This can work on the HPC with a bit of tweaking (file paths, folders, etc.). This includes the training, validation, and testing of the model and the generation of all of the images corresponding to metrics and confusion matrices.
### model_train.py: This file was used to execute all of the code from Final_Project.ipynb on the Skynet HPC. 
### dataset.py: This file defines our dataset class which returns a CT image and its corresponding label. It has some lines that need to be omitted/added for running the 3D version of our CNN. It also has two types of pixel normalization techniques present.
### cnn_model.py: This file defines our 2D and 3D basic CNNs. We did not end up using the 2D CNN, but the 3D version was tested for comparison against ResNet.
### train_test_new.py: This file holds the functions for training and testing. They return metrics and/or predictions corresponding to the model output.
### cf_matrix.py: This file is slightly tweaked from the version in class. It generates a Confusion Matrix figure based on the output of our model. It also automatically saves the confusion matrix images into a specified folder. 
