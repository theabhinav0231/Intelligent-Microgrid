import os
import sys

# Ensure we can import from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from forecasting.load.forecaster import LoadForecaster

DATA_PATH  = "forecasting/data/load/load_data_north_india.csv"
MODEL_DIR  = "models/load forecaster"

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at: {DATA_PATH}")
        print("Please run forecasting/load/data_curator.py first.")
        return

    # Initialize forecaster
    forecaster = LoadForecaster(model_dir=MODEL_DIR)

    print("=" * 50)
    print("  Load Demand Forecasting — Training")
    print("=" * 50)

    # 1. Load data
    try:
        df = forecaster.load_data(DATA_PATH)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Train model (Auto-saves on completion)
    # This will perform 80/20 chronological split and evaluate MAPE
    metrics = forecaster.train(df)

    # 3. Print Results
    print("\n--- Feature Importances ---")
    fi = forecaster.feature_importance()
    print(fi.to_string(index=False))

    print(f"\nFinal Results:")
    print(f"  RMSE : {metrics['rmse']} kW")
    print(f"  MAPE : {metrics['mape']}% (all hours)")

    if metrics['mape'] < 15:
        print(f"\n  [SUCCESS] Target MAPE < 15% achieved!")
    else:
        print(f"\n  [WARNING] MAPE above 15% target. Consider further hyperparameter tuning.")

if __name__ == "__main__":
    main()
