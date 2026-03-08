# import os
# import tensorflow as tf
# from preprocess import load_and_preprocess
# from sklearn.metrics import confusion_matrix, classification_report

# def evaluate():

#     base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     model_path = os.path.join(base_dir, "models", "ann_model.h5")  # or .keras

#     model = tf.keras.models.load_model(model_path)

#     X_train, X_test, y_train, y_test = load_and_preprocess()

#     predictions = model.predict(X_test)
#     predictions = (predictions > 0.8).astype(int)

#     print("Confusion Matrix:")
#     print(confusion_matrix(y_test, predictions))

#     print("\nClassification Report:")
#     print(classification_report(y_test, predictions))

# if __name__ == "__main__":
#     evaluate()



# roc curve added

import os
import tensorflow as tf
from preprocess import load_and_preprocess
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt


def evaluate():

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "ann_model.h5")

    model = tf.keras.models.load_model(model_path)

    X_train, X_test, y_train, y_test = load_and_preprocess()

    # probability predictions
    predictions_prob = model.predict(X_test)

    # threshold tuning
    predictions = (predictions_prob > 0.8).astype(int)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, predictions_prob)
    roc_auc = roc_auc_score(y_test, predictions_prob)

    plt.figure()
    plt.plot(fpr, tpr, label="ROC Curve (AUC = %0.2f)" % roc_auc)
    plt.plot([0,1],[0,1],'r--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Credit Card Fraud Detection")

    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    evaluate()