import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = joblib.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

# ✅ For HTML Form Submissions
@app.route('/predict', methods=['POST'])
def predict_form():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text=f'Insurance Charges {output}')

# ✅ For API (JSON Input)
@app.route('/predict_api', methods=['POST'])
def predict_api():
    '''
    For direct API calls (JSON input)
    '''
    try:
        data = request.get_json(force=True)
        features = np.array(data['input']).reshape(1, -1)  # Expect: {"input": [age, bmi, children, ...]}
        prediction = model.predict(features)
        output = round(prediction[0], 2)
        return jsonify({'prediction': output})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
