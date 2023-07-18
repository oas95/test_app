import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.metrics import accuracy_score
import itertools
import pickle


def run_ml_model(riders, bulls):
    
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
    # Drop a single column
    df_dropped = new_data.drop('Unsuccessful_Probability', axis=1)
    # Reorder columns
    df_reordered = df_dropped[['bull', 'rider', 'Successful_Probability']]

    df_pivot = df_reordered.pivot(index='rider', columns='bull', values='Successful_Probability')

    # Fill missing values with a large negative number
    df_pivot.fillna(-9999, inplace=True)

    # We want to maximize successful probability, but the function minimizes the cost,
    # so we negate the values to convert our maximization problem into a minimization problem
    cost_matrix = -df_pivot.values

    # Reset index to get the dataframe in the original format
    optimal_df = df_pivot.reset_index()

    # We want to maximize successful probability, but the function minimizes the cost, 
    # so we negate the values to convert our maximization problem into a minimization problem
    cost_matrix = -optimal_df.set_index('rider').values

    # Apply the Hungarian Algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Get the optimal assignment
    optimal_assignment = [(optimal_df.iloc[i, 0], optimal_df.columns[j+1], -cost_matrix[i, j]) for i, j in zip(row_ind, col_ind)]

    # Create a dataframe from the optimal assignment
    df_optimal = pd.DataFrame(optimal_assignment, columns=['Rider', 'Bull', 'Success Probability'])

    df_optimal = df_optimal.sort_values(by='Success Probability', ascending=False)

    # Convert the 'Success Probability' column to a percentage
    df_optimal['Success Probability'] = df_optimal['Success Probability'].map("{:.2%}".format)

    return df_optimal


