
# WASL Translator
A project that performs deep learning on ASL data to train a model to predict letters in a live environment through a python webcam application. The project has a interactive element with juypter notebooks to create various image processing inputs to train it. There is also a script to run the project training pipeline for the desired model. This outputs evaluation data and can be tested in the webcam application.


## Instructions
1. Set up Project and confirm python 3.11 is being used.
```
git clone https://github.com/1WrongfulGoose6/WASL-translator.git
pip install -r requirements.txt
python --version
```
2. Download Kaggle dataset for (e.g. https://www.kaggle.com/datasets/grassknoted/asl-alphabet?resource=download). Structure data into raw file (e.g. `WASL-translator\data\raw\asl_alphabet_train\...`).

3. Run data_preprocessing.py first to generate preferred input data (inside \data\preprocessed).

4. Run project_runner.py to and enter preferred model or point to specific model to automatically train CNN and save outputs and evaluation data. 

5. Run webcam_runner.py to deploy a live environment and test prediction accuracy.



## Techniques

**Preprocessing:** Resizing, gaussian blurs, grayscale, edge detection (Sobel, Robert, Prewitt), Otsu thresholding, normalisation.

**Training:** Optimiser (Adam), early stopping, train/validate/test split, activation functions (Relu, softmax).

**Evaluation:** Confusion Matrix, accuracy, DR/FAR. Recall, F1, ROC/AUC, Accuracy/Validation loss curves. 



## Data
While any dataset can be used, the particular dataset used was `ASL Alphabet` by Akash