from forecasting.solar_forecaster import SolarForecaster
import os

def main():
    gen_path = "forecasting/data/Plant_1_Generation_Data.csv"
    weather_path = "forecasting/data/Plant_1_Weather_Sensor_Data.csv"
    
    if not os.path.exists(gen_path) or not os.path.exists(weather_path):
        print("Data files not found. Please run the download commands first.")
        return

    print("Initializing SolarForecaster...")
    forecaster = SolarForecaster()
    
    print("Loading data...")
    df = forecaster.load_data(gen_path, weather_path)
    
    print(f"Dataset loaded. Total records: {len(df)}")
    
    print("Starting model training...")
    mape = forecaster.train(df)
    
    print(f"Training complete. Final MAPE: {mape:.2f}%")
    print("Model saved to models/solar_model.json")

if __name__ == "__main__":
    main()
