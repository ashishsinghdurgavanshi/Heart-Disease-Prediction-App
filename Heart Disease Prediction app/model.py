import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv("heart.csv")

# ------------------ GRAPH SECTION ------------------

plt.figure()
plt.hist(df['age'])
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# ---------------------------------------------------

# Features and target
X = df.drop("target", axis=1)
y = df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------ MODEL 1 ------------------
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)

print(f"Logistic Regression Accuracy: {lr_accuracy * 100:.2f}%")

# ------------------ MODEL 2 ------------------
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")

# ------------------ SELECT BEST MODEL ------------------
if rf_accuracy > lr_accuracy:
    best_model = rf_model
    print("✅ Random Forest selected as best model")
else:
    best_model = lr_model
    print("✅ Logistic Regression selected as best model")

# Save best model
pickle.dump(best_model, open("model.pkl", "wb"))

print("Model trained and saved successfully!")