import os
import requests
import pandas as pd
import numpy as np
import time
from io import StringIO

# --- CONFIGURATION ---
CITIES = {
    "Delhi": {"lat": 28.6139, "lon": 77.2090, "elev": 216},
    "Noida": {"lat": 28.5355, "lon": 77.3910, "elev": 200},
    "Gurugram": {"lat": 28.4595, "lon": 77.0266, "elev": 217},
    "Chandigarh": {"lat": 30.7333, "lon": 76.7794, "elev": 321},
    "Dehradun": {"lat": 30.3165, "lon": 78.0322, "elev": 640}
}

YEARS = [2019, 2020, 2021, 2022, 2023]
HOMES_PER_CITY = 15   # Generate 15 unique residential profiles per city
BASE_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
OUTPUT_DIR = "forecasting/data"
RAW_DIR = os.path.join(OUTPUT_DIR, "raw_weather")

def fetch_nasa_weather(city_name, lat, lon, year):
    """Fetch hourly temperature and humidity data from NASA POWER API."""
    print(f"  Fetching NASA weather for {city_name} in {year}...")
    
    start_date = f"{year}0101"
    end_date = f"{year}1231"
    
    params = {
        "parameters": "T2M,RH2M",
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": start_date,
        "end": end_date,
        "format": "CSV",
        "time-standard": "LST"
    }
    
    try:
        response = requests.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        
        content = response.text
        csv_start = content.find("YEAR,MO,DY,HR")
        if csv_start == -1:
            print(f"Error: Could not find CSV header in NASA response for {city_name} {year}")
            return None
            
        return pd.read_csv(StringIO(content[csv_start:]))
    except Exception as e:
        print(f"    Failed to fetch weather for {city_name} {year}: {e}")
        return None

def synthesize_load(df, city_name, home_index):
    """
    Generate synthetic load for a specific home with unique habits.
    """
    df = df.copy()
    
    # Create timestamp
    df['timestamp'] = pd.to_datetime({
        'year': df['YEAR'],
        'month': df['MO'],
        'day': df['DY'],
        'hour': df['HR']
    })
    
    # Feature Extraction
    hour = df['HR']
    month = df['MO']
    day_of_week = df['timestamp'].dt.dayofweek
    temp = df['T2M']
    humidity = df['RH2M']
    
    # --- HOME INDIVIDUALIZATION ---
    # Use home_index as a seed for repeatable variation
    rng = np.random.default_rng(seed=home_index * 1337)
    
    # Shift peaks (some wake up early, some late)
    morning_peak_hour = rng.uniform(6.5, 9.0)
    evening_peak_hour = rng.uniform(18.5, 21.5)
    
    # Baseline power (standby usage: electronics, fridge)
    base_load = rng.uniform(0.1, 0.25)
    
    # Peak magnitudes (some have big families, some live alone)
    morning_mag = rng.uniform(0.4, 0.8)
    evening_mag = rng.uniform(0.7, 1.4)
    
    # Climate sensitivity (some are heavy AC users, some are frugal)
    ac_threshold = rng.uniform(24, 28)
    heating_threshold = rng.uniform(12, 16)
    ac_sensitivity = rng.uniform(0.04, 0.08)
    
    # 1. Base Daily Profile
    morning_peak = morning_mag * np.exp(-((hour - morning_peak_hour)**2) / (2 * 1.5**2))
    evening_peak = evening_mag * np.exp(-((hour - evening_peak_hour)**2) / (2 * 2.5**2))
    total_base = base_load + morning_peak + evening_peak
    
    # 2. Weather Modulation
    cooling_load = np.where(temp > ac_threshold, ac_sensitivity * (temp - ac_threshold)**1.5, 0)
    cooling_load *= (1 + (humidity - 50) / 100).clip(0.8, 1.5)
    heating_load = np.where(temp < heating_threshold, 0.04 * (heating_threshold - temp)**1.2, 0)
    
    # 3. Weekend & Seasonal
    weekend_multiplier = np.where(day_of_week >= 5, 1.15, 1.0)
    seasonal_factor = 1 + 0.1 * np.cos(2 * np.pi * (month - 6) / 12)
    
    # Combined Load
    load = (total_base + cooling_load + heating_load) * weekend_multiplier * seasonal_factor
    
    # 4. Hourly High-Freq Noise (Random appliances turning on/off)
    noise = rng.normal(1.0, 0.12, size=len(load))
    load *= noise
    
    df['load_kw'] = load.clip(lower=0.04)
    df['home_id'] = f"{city_name}_{home_index:02d}"
    df.rename(columns={'T2M': 'temp_air', 'RH2M': 'humidity'}, inplace=True)
    
    return df

def main():
    if not os.path.exists(RAW_DIR):
        os.makedirs(RAW_DIR)
        
    all_city_data = []
    
    print(f"Starting Multi-City Multi-Home Data Curation ({HOMES_PER_CITY} homes/city)...")
    
    for city_name, coords in CITIES.items():
        city_weather_dfs = []
        for year in YEARS:
            cache_file = os.path.join(RAW_DIR, f"{city_name}_{year}_weather.csv")
            if os.path.exists(cache_file):
                print(f"  Using cached weather for {city_name} {year}")
                df_year = pd.read_csv(cache_file)
            else:
                df_year = fetch_nasa_weather(city_name, coords['lat'], coords['lon'], year)
                if df_year is not None:
                    df_year.to_csv(cache_file, index=False)
                    time.sleep(1)
            
            if df_year is not None:
                city_weather_dfs.append(df_year)
        
        if not city_weather_dfs:
            continue
            
        weather_combined = pd.concat(city_weather_dfs, ignore_index=True)
        
        # Generate 15 unique homes using the same weather base
        print(f"  Synthesizing {HOMES_PER_CITY} homes for {city_name}...")
        for i in range(HOMES_PER_CITY):
            home_df = synthesize_load(weather_combined, city_name, i)
            home_df['city'] = city_name
            home_df['lat'] = coords['lat']
            home_df['lon'] = coords['lon']
            all_city_data.append(home_df)
    
    print("Merging into final master dataset (this may take a moment)...")
    final_df = pd.concat(all_city_data, ignore_index=True)
    
    # Extract time features
    final_df['hour'] = final_df['timestamp'].dt.hour
    final_df['month'] = final_df['timestamp'].dt.month
    final_df['day_of_week'] = final_df['timestamp'].dt.dayofweek
    final_df['is_weekend'] = (final_df['day_of_week'] >= 5).astype(int)
    
    # Lag features (Grouped by home so lags don't spill between houses)
    print("  Calculating lag features...")
    final_df.sort_values(['home_id', 'timestamp'], inplace=True)
    final_df['load_lag_1h'] = final_df.groupby('home_id')['load_kw'].shift(1)
    final_df['load_lag_24h'] = final_df.groupby('home_id')['load_kw'].shift(24)
    
    final_df.dropna(inplace=True)
    
    cols = [
        'timestamp', 'home_id', 'city', 'lat', 'lon', 'temp_air', 'humidity',
        'hour', 'month', 'day_of_week', 'is_weekend',
        'load_lag_1h', 'load_lag_24h', 'load_kw'
    ]
    final_df = final_df[cols]
    
    output_path = os.path.join(OUTPUT_DIR, "load_data_north_india.csv")
    final_df.to_csv(output_path, index=False)
    
    print(f"SUCCESS! Load dataset saved to: {output_path}")
    print(f"Total Rows: {len(final_df):,}")
    print(f"Total Unique Homes: {final_df['home_id'].nunique()}")

if __name__ == "__main__":
    main()
