from flask import Flask, jsonify
import joblib
import pandas as pd
import os
import json

app = Flask(__name__)

def load_co2_data(filepath, window=30):
    df = pd.read_csv(filepath)
    return df['CO2_Level(ppm)'].values[-window:]

def load_threshold_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config['lower_threshold'], config['upper_threshold']

@app.route("/co2_status", methods=["GET"])
def co2_status():
    model = joblib.load("arima_model.pkl")
    co2_values = load_co2_data("../data/co2_data.csv", window=30)
    lower, upper = load_threshold_config("config.json")

    model.update(co2_values)
    forecast = model.predict(n_periods=1)[0]

    if forecast < lower:
        status = "low"
    elif forecast > upper:
        status = "high"
    else:
        status = "normal"

    return jsonify({
        "forecast": round(float(forecast), 2),
        "status": status
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

