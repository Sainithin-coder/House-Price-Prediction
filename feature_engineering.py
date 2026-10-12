"""
Feature Engineering Pipeline for Heterogeneous House Price Data.
Handles numerical, categorical, and derived features for the Joint Self-Attention model.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import os


CATEGORICAL_COLS = ['city', 'neighborhood', 'property_type', 'condition', 'garage_type', 'school_rating']
NUMERICAL_COLS = ['sqft', 'bedrooms', 'bathrooms', 'year_built', 'lot_size', 'floors',
                  'has_pool', 'has_basement', 'distance_cbd', 'crime_rate', 'walk_score', 'renovation_year']
TARGET_COL = 'price'


class FeatureEngineer:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        self.feature_names = []
        self.cat_dims = {}   # category → vocab size
        self.is_fitted = False

    def fit_transform(self, df: pd.DataFrame):
        df = df.copy()
        # Derived features
        df = self._add_derived_features(df)

        # Encode categoricals
        for col in CATEGORICAL_COLS:
            le = LabelEncoder()
            df[col + '_enc'] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
            self.cat_dims[col] = len(le.classes_)

        # Numerical features (including derived)
        num_cols = NUMERICAL_COLS + ['age', 'renovation_age', 'price_per_sqft_est', 'bed_bath_ratio', 'total_rooms']
        X_num = df[num_cols].values
        X_num_scaled = self.scaler.fit_transform(X_num)

        # Categorical encoded
        cat_enc_cols = [c + '_enc' for c in CATEGORICAL_COLS]
        X_cat = df[cat_enc_cols].values

        # Target
        y = df[TARGET_COL].values.reshape(-1, 1)
        y_scaled = self.target_scaler.fit_transform(y).ravel()

        self.feature_names = num_cols
        self.is_fitted = True

        os.makedirs('models', exist_ok=True)
        joblib.dump(self, 'models/feature_engineer.pkl')
        print("[FeatureEngineer] Fitted and saved.")

        return X_num_scaled, X_cat, y_scaled, y.ravel()

    def transform(self, df: pd.DataFrame):
        df = df.copy()
        df = self._add_derived_features(df)

        for col in CATEGORICAL_COLS:
            le = self.label_encoders[col]
            df[col + '_enc'] = df[col].astype(str).map(
                lambda x, le=le: le.transform([x])[0] if x in le.classes_ else 0
            )

        num_cols = NUMERICAL_COLS + ['age', 'renovation_age', 'price_per_sqft_est', 'bed_bath_ratio', 'total_rooms']
        X_num = df[num_cols].values
        X_num_scaled = self.scaler.transform(X_num)

        cat_enc_cols = [c + '_enc' for c in CATEGORICAL_COLS]
        X_cat = df[cat_enc_cols].values

        return X_num_scaled, X_cat

    def inverse_transform_target(self, y_scaled):
        return self.target_scaler.inverse_transform(
            np.array(y_scaled).reshape(-1, 1)
        ).ravel()

    def _add_derived_features(self, df):
        df['age'] = 2024 - df['year_built']
        df['renovation_age'] = 2024 - df['renovation_year']
        # Rough price per sqft estimate based on city (mean from training perspective)
        df['price_per_sqft_est'] = df['sqft'] * 0.05  # normalized proxy
        df['bed_bath_ratio'] = df['bedrooms'] / (df['bathrooms'] + 1e-5)
        df['total_rooms'] = df['bedrooms'] + df['bathrooms'] + df['floors']
        return df


def load_and_prepare(csv_path='data/house_data.csv'):
    df = pd.read_csv(csv_path)
    fe = FeatureEngineer()
    X_num, X_cat, y_scaled, y_raw = fe.fit_transform(df)

    X_num_tr, X_num_val, X_cat_tr, X_cat_val, y_tr, y_val, y_raw_tr, y_raw_val = train_test_split(
        X_num, X_cat, y_scaled, y_raw, test_size=0.15, random_state=42
    )

    return {
        'X_num_train': X_num_tr, 'X_num_val': X_num_val,
        'X_cat_train': X_cat_tr, 'X_cat_val': X_cat_val,
        'y_train': y_tr, 'y_val': y_val,
        'y_raw_train': y_raw_tr, 'y_raw_val': y_raw_val,
        'fe': fe
    }
