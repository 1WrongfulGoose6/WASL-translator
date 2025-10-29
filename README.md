
# WASL Translator
A WASL Transltor 


## Instructions
### Running in a local repository

1. Into a local repository `git clone https://github.com/1WrongfulGoose6/WASL-translator.git`

2. Make sure all dependencies are installed with pip install `-r requirements.txt`.

3. Confirm python 3.11.0 is being used with `python --version` (for library conflicts).

4. Download Kaggle dataset fomr `https://www.kaggle.com/datasets/grassknoted/asl-alphabet?resource=download`.

5. Unzip into `data/raw/`. File structure must be `WASL-translator\data\raw\asl_alphabet_test` and `WASL-translator\data\raw\asl_alphabet_train`

6. Run notebooks with data_preprocessing first and model_training second

### Running on Google Colab
1. Go to `https://colab.research.google.com/`

2. On left panel click GitHub, paste `https://github.com/1WrongfulGoose6/WASL-translator.git` and find project.

3. Open your selected notebook and run drive mount sections to connect to google drive.

4. Download Kaggle dataset fomr `https://www.kaggle.com/datasets/grassknoted/asl-alphabet?resource=download`.

5. Unzip into `data/raw/`. File structure must be `WASL-translator\data\raw\asl_alphabet_test` and `WASL-translator\data\raw\asl_alphabet_train`
  
6. Run notebooks with data_preprocessing first and model_training second (Optional: run git pull section if needed)
## Techniques

**Preprocessing:** Resizing, grayscale, normalisation


