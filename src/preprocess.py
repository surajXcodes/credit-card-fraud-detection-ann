import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess():

    # Load dataset
    df = pd.read_csv("data/creditcard.csv")

    # Separate features and target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Scale Amount and Time
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test