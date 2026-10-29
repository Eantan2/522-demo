**Skin Lesion Image Classifier (educational demo)**

A small web app that takes a photo of a skin lesion and shows the three most likely lesion types out of nine. Built for a University of Washington course (MSIS 522). It is a learning project, not a medical tool. Do not use it to make health decisions.

What it does
You upload an image in the browser (built with Gradio).
The app converts it to RGB, resizes it to 160 x 160 pixels, scales the pixel values to 0 to 1, and adds a batch dimension.
A TensorFlow/Keras model (a pretrained MobileNetV2 base with a small classification head) scores the image against nine classes.
The app converts the scores to probabilities with softmax and shows the top three.

The nine classes are actinic keratosis, basal cell carcinoma, dermatofibroma, melanoma, nevus, pigmented benign keratosis, seborrheic keratosis, squamous cell carcinoma, and vascular lesion.

Run it locally

Requires Python 3.11.

bash
git clone https://github.com/Eantan2/522-demo.git
cd 522-demo
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py

Open the local address the terminal prints (usually http://127.0.0.1:7860) and upload an image.

Files
File	Purpose
app.py	Loads the model, preprocesses the image, predicts, and builds the Gradio page
skin_cancer_model.h5	The trained model (about 24 MB)
requirements.txt	Python dependencies with pinned versions
Data

Images come from the International Skin Imaging Collaboration (ISIC) collection, as packaged in the Kaggle dataset "Skin Cancer ISIC" by Andrey Katanskiy (2019): 2,357 images in nine classes. The images are not included in this repository. Check the dataset's license before reusing it.

Results and limitations
This repository holds an early demo model. In the course notebook, the first versions reached roughly 40% to 51% validation accuracy and 22% to 31% accuracy on a small test set (106 images). That is low, and the model should not be trusted.
The class counts are uneven (about 60 images for the smallest class and about 370 for the largest), so the model tends to favor common classes.
The team's final model was stronger. Our course team's final ResNet50 model with data augmentation reached about 88% validation accuracy (about 92% training accuracy). That model is not in this repository.
The training images may not represent all skin tones or photo conditions, so results will vary for real-world photos.
Background

This started as a team project in MSIS 522 (University of Washington, 2025). I ran the model training and built the demo app. AI coding assistants were used for parts of the notebook code.

Disclaimer:

For educational use only. It is not medical advice and does not replace a dermatologist or any other health professional.
