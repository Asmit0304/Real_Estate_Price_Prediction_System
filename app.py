import traceback
from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app=Flask(__name__)
data = pd.read_csv('final.csv')
pipe = pickle.load(open("banglore_home_prices_model.pkl",'rb'))

@app.route('/')
def index():
    location = sorted(data['location'].unique())
    return render_template('index.html',location = location)

@app.route('/predict',methods=['POST'])
def predict():
    try:
        location = request.form.get('location')
        bhk = request.form.get('bhk')
        bath = request.form.get('bath')
        sqft = request.form.get("total_sqft")

        if not location or not bhk or not bath or not sqft:
            return "Please fill all the fields..."

        try:
            bhk = int(bhk)
            bath = int(bath)
            sqft = float(sqft)
        except:
            return "Invalid numeric values..."
            
        input = pd.DataFrame([[location,sqft,bath,bhk]],columns=['location','total_sqft','bath','bhk'])
        Prediction = float(pipe.predict(input)[0] * 1e5)
        return f"{Prediction:.2f}"
     
    except Exception as e:
        print("Error during prediction:"+e)
        traceback.print_exc()
        return "Something went wrong during prediction. Please check your input values..."

if __name__ =="__main__":
    app.run(debug=True,port = 5005)
