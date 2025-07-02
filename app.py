import os

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))


@flask_app.route("/")
def home():
    return render_template("index.html")


@flask_app.route("/predict", methods=["POST"])
def predict():
    # Get all form values
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

    # Convert to DataFrame with proper one-hot encoding
    df = pd.DataFrame([input_data])

    # Ensure all categorical columns are in correct format
    categorical_cols = ['rbc_normal', 'pc_normal', 'pcc_present', 'ba_present',
                        'htn_yes', 'dm_yes', 'cad_yes', 'appet_good', 'pe_yes', 'ane_yes']

    for col in categorical_cols:
        # Convert to binary (1 for 'yes'/'present'/'normal'/'good', 0 otherwise)
        df[col] = df[col].apply(lambda x: 1 if x.lower() in ['yes', 'present', 'normal', 'good'] else 0)

    # Make prediction
    prediction = model.predict(df)
    result = "Chronic Kidney Disease" if prediction[0] == 'ckd' else "No Kidney Disease"

    return render_template("index.html", prediction_text=f"Diagnosis: {result}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)