import numpy as np
from sklearn.datasets import load_files
import re


def data_cleaning(X):
    X_cleaned = []
    for email in range(0, len(X)):
        # Remove special characters
        text = re.sub(r'\\r\\n', ' ', str(X[email]))
        text = re.sub(r'\W', ' ', text)
        # Remove single characters from a letter
        text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
        text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text, flags=re.I)
        # Remove the 'b' that appears at the beginning
        text = re.sub(r'^b\s+', '', text)
        # Convert to lower case
        text = text.lower()

        X_cleaned.append(text)
    return X_cleaned


def load_cleaned_data():
    X, y = [], []
    for i in range(1, 7):
        emails = load_files(f"./Enron-Spam-Dataset/enron{i}")
        X = np.append(X, emails.data)
        y = np.append(y, emails.target)

    X = data_cleaning(X)
    return X, y


if __name__ == '__main__':
    X, y = load_cleaned_data()
    print(f"X.size: {len(X):}")
    print(f"y.size: {len(y):}")
    print("\n")
    print(f"Example X[0]: {X[0]}")
    print("\n")
    print(f"Class X[0]: {y[0]}")
