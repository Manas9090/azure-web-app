import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import joblib as joblib
from joblib import load, dump 

app = Flask(__name__)
model = joblib.load(open('model.pkl', 'rb'))
#model = joblib.load("E:\\Statistics\\Model_Deployment\\New\\Insur_charges_prd\\model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Insurance Charges {}'.format(output))

if __name__ == "__main__":
    #app.run(debug = True)
    app.run(host = '0.0.0.0', port = 5000, debug=True)
#app.run(port = 4995)
    
    
    
    
    

    