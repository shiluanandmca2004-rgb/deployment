import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load dataset
X, y = load_iris(return_X_y=True)

# Train logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Save trained model
joblib.dump(model, "iris_model.pkl")
print("✅ Model trained and saved as iris_model.pkl")
