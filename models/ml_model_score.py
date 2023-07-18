from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBRegressor
import pandas as pd
import itertools

class LabelEncoderWrapper(LabelEncoder):
    def fit_transform(self, y, *args, **kwargs):
        return super().fit_transform(y).reshape(-1, 1)

    def transform(self, y, *args, **kwargs):
        return super().transform(y).reshape(-1, 1)


def run_prediction_script(riders, bulls):
    # Load the data
    df = pd.read_csv('Resources/finalv1.csv')

    # Filter out the zero scores
    df_non_zero = df[df['score'] != 0]

    # Define the features and the target
    X = df_non_zero[['rider', 'bull', 'vsleft_perc', 'vsright_perc',
                     'vsavg_bull_power', 'hand', 'high_score', 'time', 'round', 'bull_power_rating', 'bullscore',
                     'buckoff_perc_vs_rh_riders',
                     'buckoff_perc_vs_lh_riders']]
    y = df_non_zero['score']

    # Get the unique class labels
    classes = y.unique()

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=classes, y=y)

    # Define the preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['rider']),
            ('num', LabelEncoderWrapper(), ['bull'])
        ])

    # Preprocess the data
    X_preprocessed = preprocessor.fit_transform(X)

    # Train the XGBoost regressor with class weighting
    model = XGBRegressor(objective='reg:squarederror', eval_metric='rmse', scale_pos_weight=(1 / class_weights[1]))
    model.fit(X_preprocessed, y)

    # Preprocess the new data
    new_data = pd.DataFrame(list(itertools.product(riders, bulls)), columns=['rider', 'bull'])
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


