from flask import Flask, render_template, request, jsonify
import joblib

model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")
accuracy = joblib.load("models/accuracy.pkl")   

app = Flask(__name__)

print("succesfully deploy to the local host")
model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze-text", methods=["POST"])
def analyze_text():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    vec = vectorizer.transform([data["text"]])
    prediction = model.predict(vec)[0]
    probability = model.predict_proba(vec)[0][1]

    # Convert prediction to friendly label
    if str(prediction) in ["1", "spam", "Spam"]:
        label = "Spam"
    else:
        label = "Not Spam"

    return jsonify({
        "prediction": label,
        "probability": float(probability),
        "accuracy": float(accuracy)   # send model accuracy too
    })

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    content = file.read().decode("utf-8", errors="ignore")

    vec = vectorizer.transform([content])
    prediction = model.predict(vec)[0]
    probability = model.predict_proba(vec)[0][1]

    prediction = str(prediction)
    probability = float(probability)

    return jsonify({
        "prediction": prediction,
        "probability": probability
    })

if __name__ == "__main__":

    app.run(debug=True)
