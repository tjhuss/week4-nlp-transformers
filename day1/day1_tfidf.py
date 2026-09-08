import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("news_dataset.csv")

X_train_text, X_test_text, y_train_labels, y_test_labels = train_test_split(
    df["Title"], df["Category"], test_size=0.2, random_state=42
)

print("Train shape:", X_train_text.shape)
print("Test shape:", X_test_text.shape)
