import pickle
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
import numpy as np
from joblib import load 

# Specify the file path where the pickled model is located
model_file_path = 'C:\\Users\\hanna\\Downloads\\deployment\\best_model.plk'

# Load the pickled model
with open(model_file_path, 'rb') as f:
    best_model_loaded = pickle.load(f)


app = Flask(__name__)

# Create a StandardScaler instance
scaler = StandardScaler()

# Preprocessing function to scale and transform user inputs to be the same datatype as the dataset
def preprocess_input(user_input):
    # Assuming user_input is a list or array
    user_input = np.array(user_input).reshape(1, -1)

    # Fit and transform user_input using StandardScaler
    user_input = scaler.transform(user_input)

    return user_input

@app.route('/predict', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user inputs from the HTML form
        user_input = [float(request.form['input_feature'])]  # Replace 'input_feature' with the actual name from the form

        # Preprocess user inputs
        user_input = preprocess_input(user_input)

        # Assign the model with scaler
        model = best_model_loaded.named_steps['pipeline-3']

        # Predict using the trained AI model
        prediction = best_model_loaded.steps[-1][-1].predict(user_input)

        # Display the prediction
        return render_template('htmlnew.html', prediction=prediction)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

