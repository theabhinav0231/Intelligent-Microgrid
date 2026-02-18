import pandas as pd
import numpy as np
import xgboost as xgb
import os
import joblib
from datetime import timedelta

class SolarForecaster:
    def __init__(self, model_path="models/solar_model.json"):
        self.model_path = model_path
        self.model = None
        self.feature_cols = [
            'hour', 'day_of_week', 'month', 
            'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION',
            'lag_1h', 'lag_24h', 'rolling_mean_3h'
        ]

    def load_data(self, gen_path, weather_path):
        """Load and merge generation and weather datasets."""
        gen_df = pd.read_csv(gen_path)
        weather_df = pd.read_csv(weather_path)

        # Convert DATE_TIME to datetime
        gen_df['DATE_TIME'] = pd.to_datetime(gen_df['DATE_TIME'])
        weather_df['DATE_TIME'] = pd.to_datetime(weather_df['DATE_TIME'])

        # Aggregate generation data by DATE_TIME (sum DC_POWER across inverters)
        gen_df = gen_df.groupby('DATE_TIME')[['DC_POWER', 'AC_POWER', 'DAILY_YIELD', 'TOTAL_YIELD']].sum().reset_index()

        # Merge on DATE_TIME
        df = pd.merge(gen_df, weather_df, on='DATE_TIME', how='inner')
        return df

    def engineer_features(self, df):
        """Create time-based and lag features."""
        df = df.copy()
        df.sort_values('DATE_TIME', inplace=True)

        # Time features
        df['hour'] = df['DATE_TIME'].dt.hour
        df['day_of_week'] = df['DATE_TIME'].dt.dayofweek
        df['month'] = df['DATE_TIME'].dt.month

        # Lag features (15-min intervals, so 1h = 4 steps, 24h = 96 steps)
        df['lag_1h'] = df['DC_POWER'].shift(4)
        df['lag_24h'] = df['DC_POWER'].shift(96)
        
        # Rolling mean (3h = 12 steps)
        df['rolling_mean_3h'] = df['DC_POWER'].shift(1).rolling(window=12).mean()

        # Drop rows with NaN from shifts
        df.dropna(inplace=True)
        return df

    def train(self, df):
        """Train the XGBoost model."""
        df_eng = self.engineer_features(df)
        
        X = df_eng[self.feature_cols]
        y = df_eng['DC_POWER']

        # Simple split (could be improved with TimeSeriesSplit)
        split_idx = int(len(df_eng) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        self.model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        preds = self.model.predict(X_test)
        rmse = np.sqrt(np.mean((y_test - preds)**2))
        mape = np.mean(np.abs((y_test - preds) / (y_test + 1e-5))) * 100
        
        print(f"Model Training Complete. RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
        
        self.save_model()
        return mape

    def save_model(self):
        """Save the model to disk."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save_model(self.model_path)

    def load_model(self):
        """Load the model from disk."""
        if os.path.exists(self.model_path):
            self.model = xgb.XGBRegressor()
            self.model.load_model(self.model_path)
            return True
        return False

    def predict_24h(self, initial_state_df):
        """
        Predict next 24 hours (96 steps of 15-min).
        initial_state_df: last 100 steps of data to compute lags.
        """
        if self.model is None:
            if not self.load_model():
                raise ValueError("Model not trained or loaded.")

        current_data = initial_state_df.copy().tail(100)
        predictions = []

        last_time = current_data['DATE_TIME'].iloc[-1]

        for _ in range(96):
            # Create features for the next step
            next_time = last_time + timedelta(minutes=15)
            
            # For simplicity in this demo, we assume weather stays same as last known 
            # or we would normally have a weather forecast.
            # Here we use the last known values for AMBIENT_TEMPERATURE, etc.
            # In a real system, we'd take weather forecast as input.
            
            row = {
                'hour': next_time.hour,
                'day_of_week': next_time.dayofweek,
                'month': next_time.month,
                'AMBIENT_TEMPERATURE': current_data['AMBIENT_TEMPERATURE'].iloc[-1],
                'MODULE_TEMPERATURE': current_data['MODULE_TEMPERATURE'].iloc[-1],
                'IRRADIATION': current_data['IRRADIATION'].iloc[-1],
                'lag_1h': current_data['DC_POWER'].iloc[-4],
                'lag_24h': current_data['DC_POWER'].iloc[-96],
                'rolling_mean_3h': current_data['DC_POWER'].tail(12).mean()
            }
            
            X_next = pd.DataFrame([row])[self.feature_cols]
            pred = self.model.predict(X_next)[0]
            pred = max(0, pred) # Energy cannot be negative
            
            # Update current_data for next iteration lags
            new_row = current_data.iloc[-1].copy()
            new_row['DATE_TIME'] = next_time
            new_row['DC_POWER'] = pred
            current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
            
            predictions.append(pred)
            last_time = next_time

        return predictions
