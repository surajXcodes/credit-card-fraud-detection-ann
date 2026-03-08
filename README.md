# Credit Card Fraud Detection using Artificial Neural Network (ANN)

## 📌 Project Overview

This project builds a **Credit Card Fraud Detection System** using an **Artificial Neural Network (ANN)**.
The model analyzes transaction features and predicts whether a transaction is **fraudulent or genuine**.

Credit card fraud is a major problem in financial systems. Machine learning models can help detect suspicious transactions and prevent financial loss.

---

## 🎯 Objective

The objective of this project is to:

* Detect fraudulent credit card transactions
* Build and train an Artificial Neural Network model
* Handle imbalanced datasets using class weights
* Evaluate the model using confusion matrix and classification metrics

---

## 🛠 Tools & Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **TensorFlow / Keras**
* **Matplotlib**
* **Seaborn**

---

## 📂 Project Structure

```
CreditCard_Fraud_Detection
│
├── data
│   └── creditcard.csv
│
├── models
│   └── ann_model.h5
│
├── src
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

The dataset used for this project is the **Credit Card Fraud Detection Dataset**.

It contains transactions made by credit cards in September 2013 by European cardholders.

Features include:

* **V1 – V28:** PCA transformed features
* **Time**
* **Amount**
* **Class**

  * 0 → Genuine transaction
  * 1 → Fraudulent transaction

---

## ⚙️ Installation

1. Clone the repository or download the project.

2. Install required libraries:

```
pip install -r requirements.txt
```

---

## ▶️ How to Run the Project

### Step 1: Train the Model

```
python src/train.py
```

This will train the ANN model and save it in the **models** folder.

---

### Step 2: Evaluate the Model

```
python src/evaluate.py
```

This will display:

* Confusion Matrix
* Classification Report
* Model performance metrics

---

## 📈 Model Performance

Example Output:

```
Confusion Matrix:
[[56610   254]
 [   10    88]]
```

### Key Metrics

* **Accuracy:** ~98%
* **Fraud Recall:** ~90%
* **Precision Improved Using Threshold Tuning**

Threshold tuning was applied to reduce false positives while maintaining high fraud detection recall.

---

## 🧠 Machine Learning Approach

1. Data preprocessing
2. Feature scaling
3. Train-test split
4. ANN model training
5. Handling class imbalance using class weights
6. Model evaluation using classification metrics

---

## 🚀 Future Improvements

* Implement **ROC Curve visualization**
* Build **Streamlit Web App Interface**
* Deploy the model as an **API**
* Improve precision using advanced techniques like **SMOTE**

---

## 👨‍💻 Author

**Suraj Kumar**

B.Tech Computer Science Engineering
Project: Credit Card Fraud Detection using ANN

---
