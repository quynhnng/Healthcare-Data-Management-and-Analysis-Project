
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import threading

# initialize the Flask app
app = Flask(__name__)

#load the trained model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return "Welcome to the Healthcare Data Management and Analysis API"

# route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    data_df = pd.DataFrame([data])
    prediction = model.predict(data_df)
    return jsonify({'prediction': prediction.tolist()})

def flask_cs210():
    app.run(debug=True, use_reloader=False, port=5001)  # Specify a different port

# Start the Flask app
thread = threading.Thread(target=flask_cs210)
thread.start()
