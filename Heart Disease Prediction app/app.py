from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = [float(x) for x in request.form.values()]
        final_input = np.array([data])

        prediction = model.predict(final_input)

        if prediction[0] == 1:
            result = "⚠️ Heart Disease Detected"
        else:
            result = "✅ No Heart Disease"

        return render_template("index.html", prediction_text=result)

    except:
        return render_template("index.html", prediction_text="Invalid Input")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)