import os
import requests
import pandas as pd
import time
from io import StringIO
import pvlib
from pvlib.pvsystem import PVSystem
from pvlib.location import Location
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

# --- CONFIGURATION ---
CITIES = {
    "Delhi": {"lat": 28.6139, "lon": 77.2090, "elev": 216},
    "Noida": {"lat": 28.5355, "lon": 77.3910, "elev": 200},
    "Gurugram": {"lat": 28.4595, "lon": 77.0266, "elev": 217},
    "Chandigarh": {"lat": 30.7333, "lon": 76.7794, "elev": 321},
    "Dehradun": {"lat": 30.3165, "lon": 78.0322, "elev": 640}
}

YEARS = [2019, 2020, 2021, 2022, 2023]
BASE_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
OUTPUT_DIR = "forecasting/data"
RAW_DIR = os.path.join(OUTPUT_DIR, "raw")

# Panel Configuration (1kW system)
PANEL_CONFIG = {
    "tilt": 28,          # Degrees
    "azimuth": 180,      # South-facing
    "capacity_kw": 1.0,
    "losses": 0.14       # 14% system losses
}

def fetch_nasa_data(city_name, lat, lon, year):
    """Fetch hourly weather data from NASA POWER API for a specific city and year."""
    print(f"  Fetching NASA data for {city_name} in {year}...")
    
    start_date = f"{year}0101"
    end_date = f"{year}1231"
    
    params = {
        "parameters": "ALLSKY_SFC_SW_DWN,T2M,WS2M",
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
        
        # Skip NASA headers (usually everything before the 'YEAR,MO,DY,HR' line)
        content = response.text
        csv_start = content.find("YEAR,MO,DY,HR")
        if csv_start == -1:
            print(f"Error: Could not find CSV header in NASA response for {city_name} {year}")
            return None
            
        return pd.read_csv(StringIO(content[csv_start:]))
    except Exception as e:
        print(f"    Failed to fetch data for {city_name} {year}: {e}")
        return None

def simulate_generation(df, city_name, lat, lon, elev):
    """Use pvlib to simulate power generation based on weather data."""
    print(f"  Simulating PV generation for {city_name}...")
    
    # Create location object
    loc = Location(lat, lon, altitude=elev, name=city_name)
    
    # Create a datetime index from NASA year, month, day, hour columns
    df['timestamp'] = pd.to_datetime({
        'year': df['YEAR'],
        'month': df['MO'],
        'day': df['DY'],
        'hour': df['HR']
    })
    df.set_index('timestamp', inplace=True)
    
    # Rename NASA columns to PVLib expected names
    # ALLSKY_SFC_SW_DWN -> ghi
    # T2M -> temp_air
    # WS2M -> wind_speed
    df.rename(columns={
        'ALLSKY_SFC_SW_DWN': 'ghi',
        'T2M': 'temp_air',
        'WS2M': 'wind_speed'
    }, inplace=True)
    
    # pvlib expects positive values for GHI. NASA uses -999 for missing, 
    # and night hours might have tiny negative noise.
    df['ghi'] = df['ghi'].clip(lower=0)
    
    # 1. Calculate Sun Position
    solar_position = loc.get_solarposition(df.index)
    
    # 2. Decompose GHI into Direct/Diffuse (using Erbs model)
    dni_dhi = pvlib.irradiance.erbs(df['ghi'], solar_position['zenith'], df.index)
    
    # 3. Calculate Plane-of-Array Irradiance
    poa_irradiance = pvlib.irradiance.get_total_irradiance(
        surface_tilt=PANEL_CONFIG['tilt'],
        surface_azimuth=PANEL_CONFIG['azimuth'],
        solar_zenith=solar_position['zenith'],
        solar_azimuth=solar_position['azimuth'],
        dni=dni_dhi['dni'],
        ghi=df['ghi'],
        dhi=dni_dhi['dhi']
    )
    
    # 4. Calculate Cell Temperature
    temp_params = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
    cell_temperature = pvlib.temperature.sapm_cell(
        poa_irradiance['poa_global'],
        df['temp_air'],
        df['wind_speed'],
        **temp_params
    )
    
    # 5. Calculate DC Power (PVWatts model)
    # pdc = (P_ref * (poa_global / 1000)) * (1 + gamma_pdc * (T_cell - T_ref))
    # Standard P_ref = 1.0 (1kW), gamma_pdc = -0.004 (poly-si)
    gamma_pdc = -0.004
    power_output = pvlib.pvsystem.pvwatts_dc(
        poa_irradiance['poa_global'],
        cell_temperature,
        pdc0=PANEL_CONFIG['capacity_kw'],
        gamma_pdc=gamma_pdc
    )
    
    # Apply system losses
    df['power_output'] = (power_output * (1 - PANEL_CONFIG['losses'])).clip(lower=0)
    df['temp_cell'] = cell_temperature
    
    return df.reset_index()

def main():
    if not os.path.exists(RAW_DIR):
        os.makedirs(RAW_DIR)
        
    all_city_data = []
    
    print(f"Starting Multi-City Data Curation for {len(CITIES)} cities...")
    
    for city_name, coords in CITIES.items():
        city_dfs = []
        for year in YEARS:
            # Check if we already have a cached raw file to save API calls
            cache_file = os.path.join(RAW_DIR, f"{city_name}_{year}_raw.csv")
            
            if os.path.exists(cache_file):
                print(f"  Using cached data for {city_name} {year}")
                df_year = pd.read_csv(cache_file)
            else:
                df_year = fetch_nasa_data(city_name, coords['lat'], coords.get('lon', coords.get('lg')), year)
                if df_year is not None:
                    df_year.to_csv(cache_file, index=False)
                    time.sleep(1) # Rate limiting courtesy
            
            if df_year is not None:
                city_dfs.append(df_year)
        
        if not city_dfs:
            print(f"No data collected for {city_name}. Skipping simulation.")
            continue
            
        # Combine all years for the city
        full_city_df = pd.concat(city_dfs, ignore_index=True)
        
        # Simulate
        simulated_df = simulate_generation(full_city_df, city_name, coords['lat'], coords.get('lon', coords.get('lg')), coords['elev'])
        
        # Add metadata
        simulated_df['city'] = city_name
        simulated_df['lat'] = coords['lat']
        simulated_df['lon'] = coords.get('lon', coords.get('lg'))
        
        all_city_data.append(simulated_df)
    
    if not all_city_data:
        print("No data collected at all. Check internet and API status.")
        return
        
    # Final merge
    print("Merging all cities into final master dataset...")
    final_df = pd.concat(all_city_data, ignore_index=True)
    
    # Final cleaning
    columns_to_keep = [
        'timestamp', 'city', 'lat', 'lon', 'ghi', 'temp_air', 
        'wind_speed', 'temp_cell', 'power_output'
    ]
    final_df = final_df[columns_to_keep]
    
    # Extract extra time features
    final_df['timestamp'] = pd.to_datetime(final_df['timestamp'])
    final_df['hour'] = final_df['timestamp'].dt.hour
    final_df['month'] = final_df['timestamp'].dt.month
    final_df['day_of_week'] = final_df['timestamp'].dt.dayofweek
    
    # Add Lag features (1h lag)
    final_df.sort_values(['city', 'timestamp'], inplace=True)
    final_df['power_lag_1h'] = final_df.groupby('city')['power_output'].shift(1)
    
    # Drop rows with NaN from lag
    final_df.dropna(inplace=True)
    
    output_path = os.path.join(OUTPUT_DIR, "training_data_north_india.csv")
    final_df.to_csv(output_path, index=False)
    
    print(f"SUCCESS! Master dataset saved to: {output_path}")
    print(f"Total Rows: {len(final_df)}")
    print(f"Cities included: {', '.join(final_df['city'].unique())}")

if __name__ == "__main__":
    main()
