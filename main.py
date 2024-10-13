from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

with open('RandomForestModel.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('RandomForestScaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        input_data = np.array([[data['Insulin'],
                                data['Glucose'],
                                data['BMI'],
                                data['AST'],
                                data['Triglycerides'],
                                data['Red Blood Cells'],
                                data['Platelets'],
                                data['Cholesterol'],
                                data['Diastolic Blood Pressure'],
                                data['C-reactive Protein'],
                                data['Mean Corpuscular Hemoglobin Concentration'],
                                data['Mean Corpuscular Hemoglobin']]])

        scaled_input = scaler.transform(input_data)

        prediction = model.predict(scaled_input)

        return jsonify({'prediction': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)