from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Load model using environment variable (fallback to local file)
MODEL_PATH = os.environ.get("MODEL_PATH", "iris_model.pkl")
model = joblib.load(MODEL_PATH)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Iris Model API is running 🚀"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = data["features"]   # expects list of 4 numbers
        prediction = model.predict([features]).tolist()
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
