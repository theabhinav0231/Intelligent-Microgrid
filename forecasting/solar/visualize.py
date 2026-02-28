import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Ensure we can import from the project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, BASE_DIR)

from forecasting.solar.forecaster import SolarForecaster

DATA_PATH = os.path.join(BASE_DIR, "forecasting", "data", "solar", "solar_forecaster_training_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at: {DATA_PATH}")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)

    forecaster = SolarForecaster(model_dir=MODEL_DIR)
    if not forecaster.load_model():
        print("Model not found. Please train the model first.")
        return

    # Load data
    df = forecaster.load_data(DATA_PATH)
    df = forecaster.preprocess(df)
    
    # Sort and pick a test window (last 3 days of the dataset for one city)
    city_to_plot = "Delhi"
    city_df = df[df['city'] == city_to_plot].sort_values('timestamp')
    
    # Take the last 72 hours
    test_window = city_df.tail(72)
    
    print(f"Generating plot for {city_to_plot} (last 72 hours)...")
    
    # Prepare features for prediction
    feature_cols = [
        'ghi', 'temp_air', 'wind_speed', 'temp_cell',
        'hour', 'month', 'day_of_week',
        'lat', 'lon',
        'power_lag_1h',
        'city_encoded'
    ]
    
    X_test = test_window[feature_cols]
    y_actual = test_window['power_output']
    
    # Predict
    y_pred = forecaster.model.predict(X_test)
    y_pred = [max(0, p) for p in y_pred]

    # Plot
    plt.figure(figsize=(15, 7))
    plt.plot(test_window['timestamp'], y_actual, label='Actual Power (kW)', color='orange', linewidth=2, marker='o')
    plt.plot(test_window['timestamp'], y_pred, label='Predicted Power (kW)', color='blue', linestyle='--', linewidth=2)
    
    plt.title(f'Solar Generation Forecasting: Actual vs Predicted ({city_to_plot})', fontsize=14)
    plt.xlabel('Timestamp', fontsize=12)
    plt.ylabel('Power Output (kW)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save
    plot_path = os.path.join(RESULTS_DIR, "solar_forecast_plot.png")
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")

    # Feature Importance Plot
    fi = forecaster.feature_importance()
    plt.figure(figsize=(10, 6))
    plt.barh(fi['feature'], fi['importance'], color='teal')
    plt.xlabel('Importance Score')
    plt.title('Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    fi_path = os.path.join(RESULTS_DIR, "feature_importance.png")
    plt.savefig(fi_path)
    print(f"Feature importance plot saved to: {fi_path}")

if __name__ == "__main__":
    main()
