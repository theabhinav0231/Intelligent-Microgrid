import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Ensure we can import from the project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, BASE_DIR)

from forecasting.load.forecaster import LoadForecaster, FEATURE_COLS

DATA_PATH = os.path.join(BASE_DIR, "forecasting", "data", "load", "load_data_north_india.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models", "load forecaster")
RESULTS_DIR = os.path.join(MODEL_DIR, "results")

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at: {DATA_PATH}")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load trained model
    forecaster = LoadForecaster(model_dir=MODEL_DIR)
    if not forecaster.load_model():
        print("Model not found. Please train the model first.")
        return

    # 2. Load and preprocess data
    df = forecaster.load_data(DATA_PATH)
    # We don't call preprocess() fully here because we want to keep the home_id etc for selection
    df['city_encoded'] = forecaster.label_encoder.transform(df['city'])
    
    # Selection: Delhi_00 (last 72 hours of the dataset)
    target_home = "Delhi_00"
    home_df = df[df['home_id'] == target_home].sort_values('timestamp')
    
    # Take the last 72 hours for plotting
    plot_window = home_df.tail(72)
    
    print(f"Generating plots for {target_home} (last 72 hours)...")
    
    # 3. Actual vs Predicted Power
    X_plot = plot_window[FEATURE_COLS]
    y_actual = plot_window['load_kw']
    
    # We use direct model.predict for the visual validation of mapping
    y_pred = forecaster.model.predict(X_plot)
    y_pred = [max(0.04, p) for p in y_pred]

    # Plot 1: Actual vs Predicted
    plt.figure(figsize=(15, 7))
    plt.plot(plot_window['timestamp'], y_actual, label='Actual Load (kW)', color='#FF8C00', linewidth=2, marker='o', markersize=4)
    plt.plot(plot_window['timestamp'], y_pred, label='Predicted Load (kW)', color='#4169E1', linestyle='--', linewidth=2)
    
    plt.title(f'Load Forecasting: Actual vs Predicted ({target_home})', fontsize=14, fontweight='bold')
    plt.xlabel('Timestamp', fontsize=12)
    plt.ylabel('Load (kW)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save
    plot_path = os.path.join(RESULTS_DIR, "load_forecast_plot.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to: {plot_path}")

    # Plot 2: Feature Importance
    fi = forecaster.feature_importance()
    plt.figure(figsize=(10, 6))
    # Use a nice color consistent with the project
    plt.barh(fi['feature'], fi['importance'], color='#20B2AA')
    plt.xlabel('Importance Score', fontsize=12)
    plt.title('Feature Importances (Load Model)', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    fi_path = os.path.join(RESULTS_DIR, "load_feature_importance.png")
    plt.savefig(fi_path, dpi=150)
    print(f"Feature importance plot saved to: {fi_path}")

if __name__ == "__main__":
    main()
