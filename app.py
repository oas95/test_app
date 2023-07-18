# Import the required libraries
from flask import Flask, jsonify, render_template, request
import pandas as pd

from models.ml_model import run_ml_model
from models.model_utils import combined_prediction, Line_Up1, Line_Up2
from label_encoder_wrapper import LabelEncoderWrapper
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBRegressor
from sklearn.utils.class_weight import compute_class_weight
import itertools
import joblib

class LabelEncoderWrapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoder = LabelEncoder()

    def fit(self, X, y=None):
        self.label_encoder.fit(X.squeeze())  # Squeeze the input X to handle a single-column DataFrame
        return self

    def transform(self, X, y=None):
        return self.label_encoder.transform(X.squeeze()).reshape(-1, 1)  # Squeeze and reshape the output
'''from models.train_model import train_model'''

app = Flask(__name__)

# Creating app routes
@app.route("/")
def home():
    return render_template('index.html')

@app.route('/riders', methods=['GET'])
def get_riders():
    df = pd.read_csv('Resources/finalv1.csv')
    riders = df['rider'].unique().tolist()
    return jsonify(riders)

@app.route('/bulls', methods=['GET'])
def get_bulls():
    df = pd.read_csv('Resources/finalv1.csv')
    bulls = df['bull'].unique().tolist()
    return jsonify(bulls)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input from the client
    input_data = request.get_json()

    # Extract the riders and bulls from the input
    riders = [input_data['rider' + str(i+1)] for i in range(7) if input_data.get('rider' + str(i+1))]
    bulls = [input_data['bull' + str(i+1)] for i in range(5) if input_data.get('bull' + str(i+1))]
    
    # Print the extracted values for debugging
    print("Riders:", riders)
    print("Bulls:", bulls)

    # Run the ML model and get the prediction
    df_optimal = run_ml_model(riders, bulls)

    # Convert the dataframe to a dictionary and return as JSON
    return jsonify(df_optimal.to_dict(orient='records'))

@app.route('/merged', methods=['POST'])
def merged():
    # Get the input from the client
    input_data = request.get_json()

    # Extract the riders and bulls from the input
    riders = [input_data['rider' + str(i+1)] for i in range(7) if input_data.get('rider' + str(i+1))]
    bulls = [input_data['bull' + str(i+1)] for i in range(5) if input_data.get('bull' + str(i+1))]
    
    # Print the extracted values for debugging
    print("Riders:", riders)
    print("Bulls:", bulls)

    # Run the ML model and get the prediction
    df_reordered = combined_prediction(riders, bulls)

    # Convert the dataframe to a dictionary and return as JSON
    return jsonify(df_reordered.to_dict(orient='records'))

@app.route('/altline', methods=['POST'])
def altline():
    # Get the input from the client
    input_data = request.get_json()

    # Extract the riders and bulls from the input
    riders = [input_data['rider' + str(i+1)] for i in range(7) if input_data.get('rider' + str(i+1))]
    bulls = [input_data['bull' + str(i+1)] for i in range(5) if input_data.get('bull' + str(i+1))]

    # Run the prediction script and get the result
    result_df = Line_Up2(riders, bulls)

    # Convert the result dataframe to a dictionary and return as JSON
    return jsonify(result_df.to_dict(orient='records'))

@app.route('/combined', methods=['POST'])
def combined():
    # Get the input from the client
    input_data = request.get_json()

    # Extract the riders and bulls from the input
    riders = [input_data['rider' + str(i+1)] for i in range(7) if input_data.get('rider' + str(i+1))]
    bulls = [input_data['bull' + str(i+1)] for i in range(5) if input_data.get('bull' + str(i+1))]

    # Run the combined prediction function and get the result
    result_df = Line_Up1(riders, bulls)

    # Convert the result dataframe to a dictionary and return as JSON
    return jsonify(result_df.to_dict(orient='records'))

'''@app.route('/train', methods=['POST'])
def train():
    response = train_model()

    return jsonify({'message': response})'''

if __name__ == '__main__':
    app.run(debug=True)
