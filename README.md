Credit Card Fraud Detection using Artificial Neural Network (ANN)

Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Download creditcard.csv and place it inside the **data** folder before running the project.

 📌 Project Overview

This project builds a Credit Card Fraud Detection System** using an Artificial Neural Network (ANN).
The model analyzes transaction features and predicts whether a transaction is fraudulent or genuine.

Credit card fraud is a major problem in financial systems. Machine learning models help detect suspicious transactions and prevent financial loss.


🎯 Objective

* Detect fraudulent credit card transactions
* Build and train an Artificial Neural Network model
* Handle imbalanced datasets using class weights
* Evaluate the model using confusion matrix and classification metrics

Tools & Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* TensorFlow / Keras
* Matplotlib
* Seaborn
* Streamlit

Project Structure

credit-card-fraud-detection-ann
│
├── models
│   └── ann_model.h5
│
├── src
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│
├── app.py
├── requirements.txt
└── README.md

Installation

Install dependencies:

```
pip install -r requirements.txt
```

---

## ▶️ Run the Project

### Train Model

```
python src/train.py
```

### Evaluate Model

```
python src/evaluate.py
```

### Run Web App

```
streamlit run app.py
```

---

## 📊 Model Performance

Example Output:

```
Confusion Matrix:
[[56610   254]
 [   10    88]]
```

### Key Metrics

* Accuracy ≈ 98%
* Fraud Recall ≈ 90%
* ROC AUC ≈ 0.98

---

## 🚀 Future Improvements

* ROC Curve visualization
* Streamlit UI enhancements
* Deploy as API
* Improve fraud detection using SMOTE

---

## 👨‍💻 Author

Suraj Kumar
B.Tech Computer Science Engineering
Project: Credit Card Fraud Detection using ANN

