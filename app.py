from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get values from form
        features = [
            request.form.get("clump_thickness"),
            request.form.get("uniformity_of_cell_size"),
            request.form.get("uniformity_of_cell_shape"),
            request.form.get("marginal_adhesion"),
            request.form.get("single_epithelial_cell_size"),
            request.form.get("bare_nuclei"),
            request.form.get("bland_chromatin"),
            request.form.get("normal_nucleoli"),
            request.form.get("mitoses")
        ]

        # Convert to 2D array and scale
        features = [list(map(int, features))]
        scaled_features = scaler.transform(features)

        # Predict
        prediction = model.predict(scaled_features)[0]
        prediction = "Benign" if prediction == 0 else "Malignant"

        return render_template("index.html", prediction=prediction)

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)