import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import os
import joblib

# Feature columns used for training
# These mirror the curated dataset schema
FEATURE_COLS = [
    'temp_air', 'humidity',
    'hour', 'month', 'day_of_week', 'is_weekend',
    'lat', 'lon',
    'load_lag_1h', 'load_lag_24h',
    'city_encoded'
]

TARGET_COL = 'load_kw'

class LoadForecaster:
    def __init__(self, model_dir="models/load forecaster"):
        """
        Initialize the LoadForecaster.
        
        Parameters
        ----------
        model_dir : str
            Directory to save/load model artifacts.
        """
        self.model_dir = model_dir
        self.model = None
        self.label_encoder = LabelEncoder()
        self._is_trained = False

    # ------------------------------------------------------------------
    # DATA LOADING
    # ------------------------------------------------------------------
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Load the curated North India load dataset."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found at: {csv_path}")
            
        print(f"Loading dataset from: {csv_path}")
        df = pd.read_csv(csv_path, parse_dates=['timestamp'])
        
        cities = df['city'].nunique()
        homes = df['home_id'].nunique()
        print(f"  Loaded {len(df):,} rows from {cities} cities, {homes} homes.")
        return df

    # ------------------------------------------------------------------
    # PREPROCESSING
    # ------------------------------------------------------------------
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features and validate data quality."""
        df = df.copy()

        # Encode city names to integers
        # We use a global LabelEncoder so it's consistent across training and prediction
        df['city_encoded'] = self.label_encoder.fit_transform(df['city'])

        # Drop rows where target is NaN (shouldn't be any in curated data)
        df.dropna(subset=[TARGET_COL], inplace=True)

        # Integrity Check: Load should never be zero in a residential setting (standby draw)
        # We clip to 0.04 kW based on synthesis parameters
        df[TARGET_COL] = df[TARGET_COL].clip(lower=0.04)
        
        # Ensure lat/lon are floats
        df['lat'] = df['lat'].astype(float)
        df['lon'] = df['lon'].astype(float)

        print(f"  After preprocessing: {len(df):,} rows.")
        return df

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------
    def train(self, df: pd.DataFrame) -> dict:
        """Train the XGBoost model with a chronological split."""
        df = self.preprocess(df)

        # Sort chronologically to prevent temporal leakage
        # Since we have many homes, we sort by timestamp
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

        # Hyperparameters tuned for load patterns (more complexity + regularization)
        self.model = xgb.XGBRegressor(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=7,
            subsample=0.75,
            colsample_bytree=0.8,
            min_child_weight=10,
            reg_alpha=0.1,
            reg_lambda=1.0,
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
        preds = np.clip(preds, 0.04, None)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mape = mean_absolute_percentage_error(y_test, preds) * 100

        metrics = {"rmse": round(rmse, 4), "mape": round(mape, 2)}

        print(f"\n{'='*40}")
        print(f" RMSE : {metrics['rmse']:.4f} kW")
        print(f" MAPE : {metrics['mape']:.2f}%  (all hours)")
        print(f"{'='*40}")

        self._is_trained = True
        self.save_model()
        return metrics

    # ------------------------------------------------------------------
    # PERSISTENCE
    # ------------------------------------------------------------------
    def save_model(self):
        """Save model weights and metadata."""
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Save XGBoost model in native JSON format
        self.model.save_model(os.path.join(self.model_dir, "load_model.json"))
        
        # Save LabelEncoder and metadata in a pickle file
        metadata = {
            'label_encoder': self.label_encoder,
            'feature_cols': FEATURE_COLS,
            'target_col': TARGET_COL
        }
        joblib.dump(metadata, os.path.join(self.model_dir, "load_forecaster.pkl"))
        
        print(f"Model saved to '{self.model_dir}/'")

    def load_model(self) -> bool:
        """Load model weights and metadata from disk."""
        model_path = os.path.join(self.model_dir, "load_model.json")
        pickle_path = os.path.join(self.model_dir, "load_forecaster.pkl")

        if not os.path.exists(model_path) or not os.path.exists(pickle_path):
            return False

        try:
            # Load XGBoost
            self.model = xgb.XGBRegressor()
            self.model.load_model(model_path)
            
            # Load metadata
            metadata = joblib.load(pickle_path)
            self.label_encoder = metadata['label_encoder']
            
            self._is_trained = True
            print(f"Model loaded from '{self.model_dir}/'")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    # ------------------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------------------
    def predict_24h(self, context_df: pd.DataFrame, city: str, home_id: str = None) -> list:
        """
        Generate a 24-hour (hourly) forecast for a specific city/home.

        Parameters
        ----------
        context_df : pd.DataFrame
            DataFrame containing at least the last 25 hours of data for the city/home.
            Must include: [timestamp, temp_air, humidity, lat, lon, load_kw]
        city : str
            City name.
        home_id : str, optional
            Home ID (e.g., 'Delhi_00'). If provided, uses home-specific context.
        
        Returns
        -------
        list of 24 floats — predicted load_kw for the next 24 hours.
        """
        if not self._is_trained:
            if not self.load_model():
                raise RuntimeError("Model not trained or found. Run train() first.")

        # Sort context and isolate target home if provided
        context = context_df.copy().sort_values('timestamp')
        if home_id:
            context = context[context['home_id'] == home_id]
        
        last_row = context.iloc[-1]
        last_ts  = pd.to_datetime(last_row['timestamp'])
        last_load = last_row['load_kw']

        try:
            city_enc = int(self.label_encoder.transform([city])[0])
        except ValueError:
            raise ValueError(f"City '{city}' was not seen during training. "
                             f"Valid cities: {list(self.label_encoder.classes_)}")

        predictions = []
        
        # Iteratively predict the next 24 hours
        for i in range(1, 25):
            next_ts = last_ts + pd.Timedelta(hours=i)
            
            # Calculate 24h lag from context or previous predictions
            # If i <= 24, we can look back in context
            ts_24h_ago = next_ts - pd.Timedelta(hours=24)
            mask_24h = context['timestamp'] == ts_24h_ago
            
            if mask_24h.any():
                load_lag_24h = context.loc[mask_24h, 'load_kw'].iloc[0]
            else:
                # Fallback to last known if context is too short (shouldn't happen with 25h context)
                load_lag_24h = last_load 

            row = {
                'temp_air'     : last_row['temp_air'],    # Assuming constant for now (or use weather forecast)
                'humidity'     : last_row['humidity'],    # Assuming constant for now
                'hour'         : next_ts.hour,
                'month'        : next_ts.month,
                'day_of_week'  : next_ts.dayofweek,
                'is_weekend'   : int(next_ts.dayofweek >= 5),
                'lat'          : last_row['lat'],
                'lon'          : last_row['lon'],
                'load_lag_1h'  : last_load,               # Auto-regressive roll
                'load_lag_24h' : load_lag_24h,            # Daily seasonal lag
                'city_encoded' : city_enc,
            }

            X = pd.DataFrame([row])[FEATURE_COLS]
            pred = float(self.model.predict(X)[0])
            pred = max(0.04, pred)  # Standby draw constraint

            predictions.append(round(pred, 4))
            last_load = pred  # Roll forward for next hour's lag_1h

        return predictions

    # ------------------------------------------------------------------
    # FEATURE IMPORTANCE
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
