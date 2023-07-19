import pandas as pd
from scipy.optimize import linear_sum_assignment
import itertools
import pickle
import os


import pandas as pd
from .label_encoder_wrapper import LabelEncoderWrapper
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBRegressor
from sklearn.utils.class_weight import compute_class_weight
import itertools
import dill


def run_prediction_script(riders, bulls):
    
   # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.realpath(__file__))

# Verify if the preprocessor joblib file exists
    preprocessor_file = os.path.join(script_dir, 'score_preprocessor1.pkl')
    if os.path.exists(preprocessor_file):
        print(f"Loading preprocessor from {preprocessor_file}")
        with open(preprocessor_file, 'rb') as f:
            preprocessor = dill.load(f)
    else:
        raise FileNotFoundError(f"{preprocessor_file} not found")

    # Verify if the model joblib file exists
    model_file = os.path.join(script_dir, 'score_model1.pkl')
    if os.path.exists(model_file):
        print(f"Loading model from {model_file}")
        with open(model_file, 'rb') as f:
            model = dill.load(f)
    else:
        raise FileNotFoundError(f"{model_file} not found")

    
   # Generate unique combinations of riders and bulls
    combinations = list(itertools.product(riders, bulls))
    unique_combinations = list(set(combinations))

    # Prepare the data for prediction
    new_data = pd.DataFrame(unique_combinations, columns=['rider', 'bull'])

    # Apply the same preprocessing to the new data
    new_data_preprocessed = preprocessor.transform(new_data)

    # Make predictions for the new data
    predictions = model.predict(new_data_preprocessed)

    # Round the predicted scores to two decimal places
    predictions = [round(score, 2) for score in predictions]
    
    # Create a result DataFrame
    result_df = pd.DataFrame(list(itertools.product(riders, bulls)), columns=['Rider', 'Bull'])
    result_df['Predicted Score'] = predictions
    
    result_df['Predicted Score'] = result_df['Predicted Score'].apply(lambda x: '{:.2f}'.format(x))
    result_df = result_df.sort_values(by='Predicted Score', ascending=False).reset_index(drop=True)

    return result_df

# Model Long List

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

def combined_prediction(riders, bulls):
    # Run the prediction script and the ml pivot script
    result_df = run_prediction_script(riders, bulls)
    print("Results from run_prediction_script:\n", result_df.head())  # print first few rows of result_df

    df_reordered = run_ml_pivot(riders, bulls)
    print("Results from run_ml_pivot:\n", df_reordered.head())  # print first few rows of df_reordered

    # Merge the two dataframes on the 'Rider' column
    merged_df = pd.merge(result_df, df_reordered, how='inner', left_on=['Rider', 'Bull'], right_on=['rider', 'bull'])
    print("Merged DataFrame:\n", merged_df.head())  # print first few rows of merged_df

    # Drop the duplicate columns 'bull' and 'rider' after the merge
    merged_df = merged_df.drop(columns=['bull', 'rider'])
    print("Merged DataFrame after dropping duplicates:\n", merged_df.head())  # print first few rows after dropping duplicates

    merged_df = merged_df.sort_values('Rider')
    
    return merged_df


def Line_Up1(riders, bulls):
    # Run the prediction script and the ml pivot script
    result_df = run_prediction_script(riders, bulls)
    print("Results from run_prediction_script:\n", result_df.head())  # print first few rows of result_df

    df_reordered = run_ml_pivot(riders, bulls)
    print("Results from run_ml_pivot:\n", df_reordered.head())  # print first few rows of df_reordered

    # Merge the two dataframes on the 'Rider' column
    merged_df = pd.merge(result_df, df_reordered, how='inner', left_on=['Rider', 'Bull'], right_on=['rider', 'bull'])
    print("Merged DataFrame:\n", merged_df.head())  # print first few rows of merged_df

    # Drop the duplicate columns 'bull' and 'rider' after the merge
    merged_df = merged_df.drop(columns=['bull', 'rider'])
    print("Merged DataFrame after dropping duplicates:\n", merged_df.head())  # print first few rows after dropping duplicates

    # Convert 'Successful_Probability' from str to float and remove the '%' at the end
    merged_df['Successful_Probability'] = merged_df['Successful_Probability'].str.rstrip('%').astype('float')

    # Convert 'Predicted Score' and 'Successful_Probability' columns to numeric
    merged_df['Predicted Score'] = pd.to_numeric(merged_df['Predicted Score'], errors='coerce')
    merged_df['Successful_Probability'] = pd.to_numeric(merged_df['Successful_Probability'], errors='coerce')

    # Create a new column 'Combined Score' that is the sum of 'Ride Probability' and 'Predicted Score'
    # Create a new column 'Combined Score' that is the weighted sum of 'Successful_Probability' and 'Predicted Score'
    merged_df['Combined Score'] = (0.75 * merged_df['Successful_Probability']) + (0.25 * merged_df['Predicted Score'])

    print(merged_df)
    
    # Pivot the DataFrame to create a cost matrix
    cost_matrix = merged_df.pivot(index='Rider', columns='Bull', values='Combined Score').fillna(0).to_numpy()

    # Apply the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)

    # Get the bull names
    bulls = merged_df['Bull'].unique()
    riders = merged_df['Rider'].unique()

    # Create a DataFrame to hold the results
    result_df = pd.DataFrame()

    # Add the optimal assignments to the result DataFrame
    for rider, bull in zip(row_ind, col_ind):
        temp_df = merged_df[(merged_df['Rider'] == riders[rider]) & (merged_df['Bull'] == bulls[bull])]
        result_df = result_df.append(temp_df)

    # Reset the index of the result DataFrame
    optimal_df = result_df.reset_index(drop=True)
    optimal_df['Successful_Probability'] = optimal_df['Successful_Probability'].apply(lambda x: '{:.2%}'.format(x / 100.0))
    top_5 = optimal_df.nlargest(5, 'Combined Score')
    optimal_df = top_5.drop('Combined Score', axis=1)

    
    print(top_5)
    return optimal_df




def Line_Up2(riders, bulls):
    # Run the prediction script and the ml pivot script
    result_df = run_prediction_script(riders, bulls)

    df_reordered = run_ml_pivot(riders, bulls)

    # Merge the two dataframes on the 'Rider' column
    merged_df = pd.merge(result_df, df_reordered, how='inner', left_on=['Rider', 'Bull'], right_on=['rider', 'bull'])

    # Drop the duplicate columns 'bull' and 'rider' after the merge
    merged_df = merged_df.drop(columns=['bull', 'rider'])

    # Convert 'Successful_Probability' from str to float and remove the '%' at the end
    merged_df['Successful_Probability'] = merged_df['Successful_Probability'].str.rstrip('%').astype('float')

    # Convert 'Predicted Score' and 'Successful_Probability' columns to numeric
    merged_df['Predicted Score'] = pd.to_numeric(merged_df['Predicted Score'], errors='coerce')
    merged_df['Successful_Probability'] = pd.to_numeric(merged_df['Successful_Probability'], errors='coerce')

    # Create a new column 'Combined Score' that is the weighted sum of 'Successful_Probability' and 'Predicted Score'
    merged_df['Combined Score'] = (0.75 * merged_df['Successful_Probability']) + (0.25 * merged_df['Predicted Score'])

    # Pivot the DataFrame to create a cost matrix
    cost_matrix = merged_df.pivot(index='Rider', columns='Bull', values='Combined Score').fillna(0).to_numpy()

    # Apply the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)

    # Store the best lineup
    lineup1 = (row_ind, col_ind)

    # Create a copy of cost_matrix and remove one of the optimal assignments
    cost_matrix_2 = cost_matrix.copy()
    cost_matrix_2[row_ind[0], col_ind[0]] = 0

    # Apply the Hungarian algorithm again
    row_ind_2, col_ind_2 = linear_sum_assignment(-cost_matrix_2)

    # Store the second best lineup
    lineup2 = (row_ind_2, col_ind_2)

    # Get the bull names
    bulls = merged_df['Bull'].unique()
    riders = merged_df['Rider'].unique()

    # Create DataFrames to hold the results
    result_df1 = pd.DataFrame()
    result_df2 = pd.DataFrame()

    # Add the optimal assignments to the result DataFrames
    for rider, bull in zip(lineup1[0], lineup1[1]):
        temp_df = merged_df[(merged_df['Rider'] == riders[rider]) & (merged_df['Bull'] == bulls[bull])]
        result_df1 = result_df1.append(temp_df)

    for rider, bull in zip(lineup2[0], lineup2[1]):
        temp_df = merged_df[(merged_df['Rider'] == riders[rider]) & (merged_df['Bull'] == bulls[bull])]
        result_df2 = result_df2.append(temp_df)

    # Reset the index of the result DataFrames
    optimal_df1 = result_df1.reset_index(drop=True)
    optimal_df2 = result_df2.reset_index(drop=True)
    optimal_df1['Successful_Probability'] = optimal_df1['Successful_Probability'].apply(lambda x: '{:.2%}'.format(x))
    optimal_df2['Successful_Probability'] = optimal_df2['Successful_Probability'].apply(lambda x: '{:.2%}'.format(x / 100.0))
    
    print("First best lineup:\n", optimal_df1)
    print("Second best lineup:\n", optimal_df2)
    
    return optimal_df2

