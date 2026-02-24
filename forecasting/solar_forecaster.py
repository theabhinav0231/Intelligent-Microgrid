import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import os
import joblib

# Feature columns used for training
FEATURE_COLS = [
    'ghi', 'temp_air', 'wind_speed', 'temp_cell',
    'hour', 'month', 'day_of_week',
    'lat', 'lon',
    'power_lag_1h',
    'city_encoded'
]

TARGET_COL = 'power_output'

class SolarForecaster:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.model = None
        self.label_encoder = LabelEncoder()
        self._is_trained = False

    # ------------------------------------------------------------------
    # DATA LOADING
    # ------------------------------------------------------------------
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Load the curated North India training dataset."""
        print(f"Loading dataset from: {csv_path}")
        df = pd.read_csv(csv_path, parse_dates=['timestamp'])
        print(f"  Loaded {len(df):,} rows from {df['city'].nunique()} cities.")
        return df

    # ------------------------------------------------------------------
    # PREPROCESSING
    # ------------------------------------------------------------------
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features and validate data quality."""
        df = df.copy()

        # Encode city names to integers
        df['city_encoded'] = self.label_encoder.fit_transform(df['city'])

        # Drop rows where power_output is NaN (shouldn't be any)
        df.dropna(subset=[TARGET_COL], inplace=True)

        # Clip any stray negatives (physics cannot produce negative energy)
        df['power_output'] = df['power_output'].clip(lower=0)
        df['power_lag_1h'] = df['power_lag_1h'].clip(lower=0)

        print(f"  After preprocessing: {len(df):,} rows, {df['city'].nunique()} cities.")
        return df

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------
    def train(self, df: pd.DataFrame) -> dict:
        """Train the XGBoost model with a time-based train/test split."""
        df = self.preprocess(df)

        # Sort chronologically and use the last 20% as test set
        df.sort_values('timestamp', inplace=True)
        split_idx = int(len(df) * 0.8)
        
        train_df = df.iloc[:split_idx]
        test_df  = df.iloc[split_idx:]

        X_train = train_df[FEATURE_COLS]
        y_train = train_df[TARGET_COL]
        X_test  = test_df[FEATURE_COLS]
        y_test  = test_df[TARGET_COL]

        print(f"\nTraining split  : {len(X_train):,} rows")
        print(f"Test split      : {len(X_test):,} rows")

        self.model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            random_state=42,
            n_jobs=-1,
        )

        print("\nTraining XGBoost model...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=100
        )

        # Evaluate
        preds = self.model.predict(X_test)
        preds = np.clip(preds, 0, None)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        
        # MAPE: only calculate on rows where actual > 0.01 kW (exclude night hours)
        mask = y_test > 0.01
        mape = mean_absolute_percentage_error(y_test[mask], preds[mask]) * 100

        metrics = {"rmse": round(rmse, 4), "mape": round(mape, 2)}

        print(f"\n{'='*40}")
        print(f" RMSE : {metrics['rmse']:.4f} kW")
        print(f" MAPE : {metrics['mape']:.2f}%  (on daytime hours only)")
        print(f"{'='*40}")

        self._is_trained = True
        self.save_model()
        return metrics

    # ------------------------------------------------------------------
    # PERSISTENCE
    # ------------------------------------------------------------------
    def save_model(self):
        os.makedirs(self.model_dir, exist_ok=True)
        self.model.save_model(os.path.join(self.model_dir, "solar_model.json"))
        joblib.dump(self.label_encoder, os.path.join(self.model_dir, "label_encoder.pkl"))
        print(f"Model saved to '{self.model_dir}/'")

    def load_model(self) -> bool:
        model_path   = os.path.join(self.model_dir, "solar_model.json")
        encoder_path = os.path.join(self.model_dir, "label_encoder.pkl")
        if os.path.exists(model_path) and os.path.exists(encoder_path):
            self.model = xgb.XGBRegressor()
            self.model.load_model(model_path)
            self.label_encoder = joblib.load(encoder_path)
            self._is_trained = True
            return True
        return False

    # ------------------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------------------
    def predict_24h(self, context_df: pd.DataFrame, city: str) -> list:
        """
        Generate a 24-hour (hourly) forecast for a city.

        Parameters
        ----------
        context_df : pd.DataFrame
            DataFrame containing at least the last 2 hours of data for the city,
            with columns: [timestamp, ghi, temp_air, wind_speed, temp_cell,
                           lat, lon, power_output]
        city : str
            City name — must be one of the 5 trained cities.
        
        Returns
        -------
        list of 24 floats — predicted power_output (kW) for the next 24 hours.
        """
        if not self._is_trained:
            if not self.load_model():
                raise RuntimeError("Model not trained or found. Run train() first.")

        context = context_df.copy().sort_values('timestamp')
        last_row = context.iloc[-1]
        last_ts  = pd.to_datetime(last_row['timestamp'])
        last_power = last_row['power_output']

        try:
            city_enc = int(self.label_encoder.transform([city])[0])
        except ValueError:
            raise ValueError(f"City '{city}' was not seen during training. "
                             f"Valid cities: {list(self.label_encoder.classes_)}")

        predictions = []
        for i in range(1, 25):
            next_ts = last_ts + pd.Timedelta(hours=i)

            row = {
                'ghi'          : last_row['ghi'],        # Replace with weather forecast
                'temp_air'     : last_row['temp_air'],   # Replace with weather forecast
                'wind_speed'   : last_row['wind_speed'], # Replace with weather forecast
                'temp_cell'    : last_row['temp_cell'],
                'hour'         : next_ts.hour,
                'month'        : next_ts.month,
                'day_of_week'  : next_ts.dayofweek,
                'lat'          : last_row['lat'],
                'lon'          : last_row['lon'],
                'power_lag_1h' : last_power,
                'city_encoded' : city_enc,
            }

            X = pd.DataFrame([row])[FEATURE_COLS]
            pred = float(self.model.predict(X)[0])
            pred = max(0.0, pred)

            predictions.append(round(pred, 4))
            last_power = pred  # Roll forward lag

        return predictions

    # ------------------------------------------------------------------
    # FEATURE IMPORTANCE (Bonus)
    # ------------------------------------------------------------------
    def feature_importance(self) -> pd.DataFrame:
        """Return a sorted DataFrame of feature importances."""
        if not self._is_trained:
            raise RuntimeError("Model not trained yet.")
        scores = self.model.feature_importances_
        return (
            pd.DataFrame({'feature': FEATURE_COLS, 'importance': scores})
            .sort_values('importance', ascending=False)
            .reset_index(drop=True)
        )
