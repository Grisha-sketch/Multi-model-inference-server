"""
create_sample_model.py
Run this script once to generate a sample trained model (.pkl)
that you can upload and test with the inference server.

Model: Iris flower classifier (3 classes)
Input: 4 features [sepal_length, sepal_width, petal_length, petal_width]
Output: 0 = Setosa, 1 = Versicolor, 2 = Virginica
"""

import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model trained successfully!")
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save the model as .pkl
os.makedirs("models", exist_ok=True)
save_path = "models/iris_classifier_v1.pkl"

with open(save_path, "wb") as f:
    pickle.dump(model, f)

print(f"Model saved to: {save_path}")
print()
print("=" * 50)
print("You can now test predictions with this input:")
print("  Setosa     → [[5.1, 3.5, 1.4, 0.2]]")
print("  Versicolor → [[6.0, 2.7, 5.1, 1.6]]")
print("  Virginica  → [[6.7, 3.1, 5.6, 2.4]]")
print("=" * 50)