from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib
import config

app = Flask(__name__)

pipeline = joblib.load(filename=config.PIPELINE_NAME)


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html', config=config)
    
    
    
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        region = str(request.form['region'])
        typee = str(request.form['type'])
        sqfeet = float(request.form['sqfeet'])
        beds = int(request.form['beds'])
        baths = int(request.form['baths'])
        smoking_allowed = str(request.form['smoking_allowed'])
        wheelchair_access = str(request.form['wheelchair_access'])
        electric_vehicle_charge = str(request.form['electric_vehicle_charge'])
        comes_furnished = str(request.form['comes_furnished'])
        laundry_options = str(request.form['laundry_options'])
        parking_options = str(request.form['parking_options'])
        state = str(request.form['state'])

        json_data = {
            "region": region,
            "type": typee,
            "sqfeet": sqfeet,
            "beds": beds,
            "baths": baths,
            "smoking_allowed": smoking_allowed,
            "wheelchair_access": wheelchair_access,
            "electric_vehicle_charge": electric_vehicle_charge,
            "comes_furnished": comes_furnished,
            "laundry_options": laundry_options,
            "parking_options": parking_options,
            "state": state
        }

        data = pd.DataFrame(json_data, index=[0], columns=config.FEATURES)
        prediction = pipeline.predict(data)
        result = "The monthly house rent will be " + str(np.round(prediction[0])) + "."
        return render_template('index.html', result=result, config=config)


@app.route('/predict_via_postman', methods=['POST'])
def predict_via_postman():
    if request.method == 'POST':
        json_data = request.get_json()
        data = pd.DataFrame(json_data, index=[0], columns=config.FEATURES)

        prediction = pipeline.predict(data)
        result = "The monthly house rent will be " + str(prediction[0]) + "."
        return result


if __name__ == "__main__":
    app.run(debug=True)
