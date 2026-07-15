from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load saved model, scaler, and feature names
model = joblib.load("floods.save")
scaler = joblib.load("transform.save")
feature_names = joblib.load("feature_names.save")


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/Predict')
def predict_page():
    return render_template('index.html', features=feature_names)


@app.route('/predict', methods=['POST'])
def predict():
    # Dynamically read all form inputs based on saved feature names
    input_values = []
    for feat in feature_names:
        input_values.append(float(request.form[feat]))

    input_df = pd.DataFrame([input_values], columns=feature_names)

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)

    if prediction[0] == 1:
        return render_template('chance.html')
    else:
        return render_template('no_chance.html')


if __name__ == '__main__':
    app.run(debug=True)
