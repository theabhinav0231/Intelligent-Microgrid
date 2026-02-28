import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# Ensure imports from project root work
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)

from forecasting.load.forecaster import LoadForecaster, FEATURE_COLS, TARGET_COL

# ── Config ──────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join(BASE_DIR, "forecasting", "data", "load", "load_data_north_india.csv")
MODEL_DIR   = os.path.join(BASE_DIR, "models", "load forecaster")
RESULTS_DIR = os.path.join(MODEL_DIR, "results")

NOISE_LEVELS  = [0, 5, 10, 15, 20]          # % standard-deviation of noise
# Load is sensitive to Temperature and Humidity
WEATHER_COLS  = ["temp_air", "humidity"]
N_TRIALS      = 10                            # Monte-Carlo repetitions per level
RANDOM_SEED   = 42

def load_test_set(forecaster: LoadForecaster, csv_path: str):
    """Load data and reproduce the same 80/20 chronological split used in training."""
    df = forecaster.load_data(csv_path)
    df = forecaster.preprocess(df)
    df.sort_values("timestamp", inplace=True)

    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()
    print(f"  Test set size: {len(test_df):,} rows")
    return test_df

def inject_noise(df: pd.DataFrame, noise_pct: float, rng: np.random.Generator):
    """Add Gaussian noise to weather columns."""
    noisy = df.copy()
    sigma = noise_pct / 100.0

    for col in WEATHER_COLS:
        noise = rng.normal(0, sigma, size=len(noisy))
        noisy[col] = noisy[col] * (1 + noise)

    # Physics constraints
    if "humidity" in noisy.columns:
        noisy["humidity"] = noisy["humidity"].clip(0, 100)
    
    return noisy

def evaluate_mape(model, df: pd.DataFrame) -> float:
    """MAPE on all hours."""
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # Model predictions clipped to standby draw
    preds = np.clip(model.predict(X), 0.04, None)
    
    mape = np.mean(np.abs((y - preds) / y)) * 100
    return round(mape, 4)

def run_sensitivity(forecaster, test_df):
    """Monte-Carlo sensitivity sweep over noise levels."""
    rng = np.random.default_rng(RANDOM_SEED)

    results = []
    for noise_pct in NOISE_LEVELS:
        trial_mapes = []
        print(f"  Processing noise level {noise_pct}%...")
        for trial in range(N_TRIALS):
            noisy_df = inject_noise(test_df, noise_pct, rng)
            mape = evaluate_mape(forecaster.model, noisy_df)
            trial_mapes.append(mape)

        mean_mape = round(np.mean(trial_mapes), 2)
        std_mape  = round(np.std(trial_mapes), 2)
        min_mape  = round(np.min(trial_mapes), 2)
        max_mape  = round(np.max(trial_mapes), 2)

        results.append({
            "noise_pct": noise_pct,
            "mean_mape": mean_mape,
            "std_mape":  std_mape,
            "min_mape":  min_mape,
            "max_mape":  max_mape,
        })
        print(f"    -> MAPE = {mean_mape:.2f}% +/- {std_mape:.2f}%")

    return pd.DataFrame(results)

def plot_sensitivity(results_df: pd.DataFrame, save_path: str):
    """Publication-quality sensitivity chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = results_df["noise_pct"]
    y = results_df["mean_mape"]
    yerr_lo = y - results_df["min_mape"]
    yerr_hi = results_df["max_mape"] - y

    # Styled area fill
    ax.fill_between(x, 0, y, alpha=0.1, color="#7209B7")
    ax.plot(x, y, "o-", color="#7209B7", linewidth=2.5, markersize=9,
            markerfacecolor="white", markeredgewidth=2.5, label="Mean MAPE")
    ax.errorbar(x, y, yerr=[yerr_lo, yerr_hi], fmt="none",
                ecolor="#7209B7", elinewidth=1.2, capsize=5, capthick=1.2, alpha=0.6)

    # Industry benchmark band (15% is the target)
    ax.axhline(15, color="#F72585", linestyle="--", linewidth=1.5, alpha=0.6, label="Industry Target (15%)")
    
    # Annotations
    for _, row in results_df.iterrows():
        ax.annotate(f'{row["mean_mape"]:.2f}%',
                    xy=(row["noise_pct"], row["mean_mape"]),
                    xytext=(0, 15),
                    textcoords="offset points", ha="center", fontsize=10,
                    fontweight="bold", color="#480CA8")

    ax.set_xlabel("Weather Forecast Error (% Std Dev of Gaussian Noise)", fontsize=12)
    ax.set_ylabel("MAPE (%)", fontsize=12)
    ax.set_title("Load Model Sensitivity: Error vs Weather Uncertainty",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(NOISE_LEVELS)
    ax.set_xticklabels([f"{n}%" for n in NOISE_LEVELS])
    ax.set_ylim(0, max(y.max() + 5, 20))
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    print(f"  Chart saved -> {save_path}")
    plt.close(fig)

def main():
    print("=" * 60)
    print("  Load Model Sensitivity Analysis — Weather Uncertainty")
    print("=" * 60)

    # 1. Load model
    forecaster = LoadForecaster(model_dir=MODEL_DIR)
    if not forecaster.load_model():
        print("ERROR: Trained model not found.")
        return

    # 2. Load test data
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Dataset not found.")
        return
    test_df = load_test_set(forecaster, DATA_PATH)

    # 3. Run sweep
    print("\nRunning sensitivity sweep (10 trials per noise level)...\n")
    results_df = run_sensitivity(forecaster, test_df)

    # 4. Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path  = os.path.join(RESULTS_DIR, "load_sensitivity_results.csv")
    plot_path = os.path.join(RESULTS_DIR, "load_sensitivity_analysis.png")

    results_df.to_csv(csv_path, index=False)
    print(f"\n  Results CSV saved -> {csv_path}")

    plot_sensitivity(results_df, plot_path)

    # 5. Summary
    baseline = results_df.loc[results_df["noise_pct"] == 0, "mean_mape"].values[0]
    worst    = results_df.loc[results_df["noise_pct"] == NOISE_LEVELS[-1], "mean_mape"].values[0]

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Baseline (0% noise)    : {baseline:.2f}% MAPE")
    print(f"  Worst-case ({NOISE_LEVELS[-1]}% noise) : {worst:.2f}% MAPE")
    if worst < 15:
        print(f"  [PASSED] Model stays Robust (MAPE < 15%) even with {NOISE_LEVELS[-1]}% noise.")
    else:
        print(f"  [WARNING] Model sensitivity exceeds limits at high noise levels.")
    print("=" * 60)

if __name__ == "__main__":
    main()
