import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import itertools
import pickle
import os

#Importing Sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, func, inspect, MetaData, select, distinct
#Importing Flask
from flask import Flask, jsonify, render_template
from models.train_model import train_model
from sqlalchemy import create_engine


def run_ml_pivot(riders, bulls):
    
    # Load the trained models
    with open('./models/encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    with open('./models/logreg.pkl', 'rb') as f:
        logreg = pickle.load(f)
    
    combinations = list(itertools.product(riders, bulls))

    # Prepare the data for prediction
    new_data = pd.DataFrame(combinations, columns=['rider', 'bname'])

    # Apply one-hot encoding
    encoded_new_data = encoder.transform(new_data)

    # Use the trained model to predict probabilities
    new_probabilities = logreg.predict_proba(encoded_new_data)

    # Add the probabilities to our dataframe
    new_data['prob_0'] = new_probabilities[:, 0]
    new_data['prob_1'] = new_probabilities[:, 1]

    # Sort the dataframe by probability of success
    new_data = new_data.sort_values('prob_1', ascending=False)

    # Reset the index
    new_data.reset_index(drop=True, inplace=True)
    new_data = new_data.rename(columns={"bname": "bull",
                                        "prob_0": "Unsuccessful_Probability",
                                        "prob_1": "Successful_Probability"
                                        })

    # Drop the 'Unsuccessful_Probability' column
    df_dropped = new_data.drop('Unsuccessful_Probability', axis=1)

    # Reorder columns
    df_reordered = df_dropped[['bull', 'rider', 'Successful_Probability']]
    
        # Convert the 'Success Probability' column to a percentage
    df_reordered['Successful_Probability'] = df_reordered['Successful_Probability'].map("{:.2%}".format)

    return df_reordered
