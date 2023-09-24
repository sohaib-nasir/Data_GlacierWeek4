import numpy as np
from flask import Flask, request, jsonify

import joblib  # Used to load your machine learning model

app = Flask(__name__)

# Load your machine learning model
model = joblib.load("KNeighborsClassifier.pkl")

@app.route('/')
def hello():
    return "Hello, World!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from POST request
    
    data = request.get_json()
    print(data)
    data=np.expand_dims(data, axis=0)

    # Make predictions using your model
    prediction = model.predict(data)

    # Return the prediction as JSON
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
