from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin

class LabelEncoderWrapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoder = LabelEncoder()

    def fit(self, X, y=None):
        self.label_encoder.fit(X.squeeze())  # Squeeze the input X to handle a single-column DataFrame
        return self

    def transform(self, X, y=None):
        if len(X.shape) > 1:
            return self.label_encoder.transform(X.squeeze()).reshape(-1, 1)  # Squeeze and reshape the output
        else:
            return self.label_encoder.transform(X.reshape(-1, 1))

