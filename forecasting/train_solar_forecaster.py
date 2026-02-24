import os
import sys

# Ensure we can import from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from forecasting.solar_forecaster import SolarForecaster

DATA_PATH  = "forecasting/forecasting/data/training_data_north_india.csv"
MODEL_DIR  = "models"

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at: {DATA_PATH}")
        print("Please run forecasting/data_curator.py first.")
        return

    forecaster = SolarForecaster(model_dir=MODEL_DIR)

    print("=" * 50)
    print("  Solar Generation Forecasting — Training")
    print("=" * 50)

    df = forecaster.load_data(DATA_PATH)
    metrics = forecaster.train(df)

    print("\n--- Feature Importances ---")
    print(forecaster.feature_importance().to_string(index=False))

    print(f"\nFinal Results:")
    print(f"  RMSE : {metrics['rmse']} kW")
    print(f"  MAPE : {metrics['mape']}% (daytime hours)")

    if metrics['mape'] < 15:
        print(f"\n  Target MAPE < 15% achieved!")
    else:
        print(f"\n  MAPE above target. Consider tuning hyperparameters.")

if __name__ == "__main__":
    main()
