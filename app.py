import os
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load model (ensure model.pkl is in the root directory)
try:
    model = pickle.load(open("model.pkl", "rb"))
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


# Fixed routes with @app.route decorator
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        input_data = {
            'age': float(request.form['age']),
            'bp': float(request.form['bp']),
            'sg': float(request.form['sg']),
            'al': float(request.form['al']),
            'su': float(request.form['su']),
            'bgr': float(request.form['bgr']),
            'bu': float(request.form['bu']),
            'sc': float(request.form['sc']),
            'sod': float(request.form['sod']),
            'pot': float(request.form['pot']),
            'hemo': float(request.form['hemo']),
            'pcv': float(request.form['pcv']),
            'wc': float(request.form['wc']),
            'rc': float(request.form['rc']),
            'rbc_normal': request.form['rbc'],
            'pc_normal': request.form['pc'],
            'pcc_present': request.form['pcc'],
            'ba_present': request.form['ba'],
            'htn_yes': request.form['htn'],
            'dm_yes': request.form['dm'],
            'cad_yes': request.form['cad'],
            'appet_good': request.form['appet'],
            'pe_yes': request.form['pe'],
            'ane_yes': request.form['ane']
        }

        df = pd.DataFrame([input_data])
        categorical_cols = ['rbc_normal', 'pc_normal', 'pcc_present', 'ba_present',
                            'htn_yes', 'dm_yes', 'cad_yes', 'appet_good', 'pe_yes', 'ane_yes']

        for col in categorical_cols:
            df[col] = df[col].apply(lambda x: 1 if str(x).lower() in ['yes', 'present', 'normal', 'good'] else 0)

        prediction = model.predict(df)
        result = "Chronic Kidney Disease" if prediction[0] == 'ckd' else "No Kidney Disease"
        return render_template("index.html", prediction_text=f"Diagnosis: {result}")

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render uses PORT env variable
    app.run(host="0.0.0.0", port=port, debug=False)  # Disable debug in production