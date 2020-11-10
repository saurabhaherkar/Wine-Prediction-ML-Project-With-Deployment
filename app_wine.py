import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load('wine_predictor.pkl')


@app.route('/')
def home():
    return render_template('index_wine.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    input_features = [float(i) for i in request.form.values()]
    feature_value = np.array(input_features)
    feature_names = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
                     'total_phenols', 'flavanoid', 'nonflavanoid_phenols', 'proanthocyanins',
                     'color_intensity', 'hue', 'od280/od315_of_diluted_wine', 'proline']

    df = pd.DataFrame([feature_value], columns=feature_names)
    output = model.predict(df)
    return render_template('index_wine.html', prediction_text='This wine of the class {}'.format(output))


if __name__ == '__main__':
    app.run(debug=True)