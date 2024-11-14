
import numpy as np
import joblib
from flask import Flask, render_template, request
from sklearn.datasets import load_iris

# Loading Iris Dataset
iris = load_iris()

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        v1 = float(request.form['s_length'])
        v2 = float(request.form['s_width'])
        v3 = float(request.form['p_length'])
        v4 = float(request.form['p_width'])
        
        
        pred_args = [v1,v2,v3,v4]        
        pred_arr = np.array(pred_args)        
        preds = pred_arr.reshape(1,-1)  # preds variable is numpy arr of shape 2d
    
       # model = open(r"IRIS_dataset_model.pki","rb")
        
        lris_model = joblib.load(r"IRIS_dataset_model.pki")
        
        prediction = lris_model.predict(preds)       

        
    return render_template('home.html',output = iris.target_names[prediction[0]] )

if __name__=='__main__':
    app.run(host='0.0.0.0')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
