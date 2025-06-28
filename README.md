# Iris Flower Classifier

![Iris Flowers](./images/READMEpic.png)

An end-to-end machine learning pipeline for classifying Iris species. From data preparation and model training to evaluation and a Flask API for live predictions.

---

## 🚀 Features

- **Data Preparation**  
  - Load raw CSV, minimal cleaning, stratified train/test split  
- **Model Training**  
  - Standard scaling + Logistic Regression  
- **Evaluation**  
  - Accuracy, classification report, confusion matrix  
- **API Service**  
  - Flask app exposing `/predict` endpoint for JSON input  
- **Serialization**  
  - Model saved as `models/iris_classifier.joblib`

---

## 📁 Repository Structure

iris-classifier/
├── data/
│ ├── raw/ # raw iris.csv
│ └── processed/ # train.csv & test.csv
│
├── images/
│ └── iris_flowers.png # illustration used in README
│
├── models/
│ └── iris_classifier.joblib # trained model pipeline
│
├── src/
│ ├── data_prep.py # data loading & train/test split
│ ├── train.py # build & train model pipeline
│ ├── evaluate.py # compute metrics & print reports
│ └── app.py # Flask API for live inference
│
├── requirements.txt # project dependencies
└── README.md # this file


---

## ⚙️ Setup & Installation

1. **Install dependencies**  

pip install -r requirements.txt



2. **Prepare the data**  

python src/data_prep.py



3. **Train the model**

python src/train.py \
  --input data/processed/train.csv \
  --output models/iris_classifier.joblib



4. **Evaluate Performance**

python src/evaluate.py \
  --model-path models/iris_classifier.joblib \
  --test-data data/processed/test.csv



**🖥️ Running the Flask API**

cd src
flask run --host=0.0.0.0 --port=5000


**/predict Endpoint**

Send a POST request with JSON body:
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}

**Response**

{
  "predicted_species": "Setosa",
  "confidence": 0.96
}


