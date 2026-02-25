"""
Sensitivity Analysis — Weather Forecast Uncertainty
=====================================================
This script quantifies how the Solar Forecasting model's accuracy degrades
when the weather inputs (GHI, Temperature, Wind Speed) contain forecast
errors. This simulates the realistic scenario where a weather *forecast*
(with inherent uncertainty) is used instead of *observed* conditions.

Methodology:
    For each noise level (0%, 5%, 10%, 15%, 20%):
        1. Add Gaussian noise to weather features in the TEST set.
        2. Run inference with the trained model.
        3. Calculate daytime MAPE.
        4. Repeat 10 times and average for statistical robustness.

Output:
    - sensitivity_analysis.png   (chart)
    - sensitivity_results.csv    (raw numbers)
    - Console summary
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# Ensure imports from project root work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from forecasting.solar_forecaster import SolarForecaster, FEATURE_COLS, TARGET_COL

# ── Config ──────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join("forecasting", "data", "solar_forecaster_training_data.csv")
MODEL_DIR   = os.path.join("models", "solar forecaster")
RESULTS_DIR = os.path.join(MODEL_DIR, "results")

NOISE_LEVELS  = [0, 5, 10, 15, 20]          # % standard-deviation of noise
WEATHER_COLS  = ["ghi", "temp_air", "wind_speed"]
N_TRIALS      = 10                            # Monte-Carlo repetitions per level
RANDOM_SEED   = 42


def load_test_set(forecaster: SolarForecaster, csv_path: str):
    """Load data and reproduce the same 80/20 chronological split used in training."""
    df = forecaster.load_data(csv_path)
    df = forecaster.preprocess(df)
    df.sort_values("timestamp", inplace=True)

    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()
    print(f"  Test set size: {len(test_df):,} rows")
    return test_df


def inject_noise(df: pd.DataFrame, noise_pct: float, rng: np.random.Generator):
    """
    Add Gaussian noise to weather columns.

    For each weather column the noise is:
        value_noisy = value * (1 + N(0, noise_pct / 100))

    GHI is clipped to ≥ 0 (negative irradiance is physically impossible).
    """
    noisy = df.copy()
    sigma = noise_pct / 100.0

    for col in WEATHER_COLS:
        noise = rng.normal(0, sigma, size=len(noisy))
        noisy[col] = noisy[col] * (1 + noise)

    # Physics: GHI cannot be negative
    noisy["ghi"] = noisy["ghi"].clip(lower=0)
    return noisy


def evaluate_mape(model, df: pd.DataFrame) -> float:
    """MAPE on daytime hours only (power_output > 0.01 kW)."""
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    preds = np.clip(model.predict(X), 0, None)
    mask  = y > 0.01
    if mask.sum() == 0:
        return np.nan

    mape = np.mean(np.abs((y[mask] - preds[mask]) / y[mask])) * 100
    return round(mape, 4)


def run_sensitivity(forecaster, test_df):
    """Monte-Carlo sensitivity sweep over noise levels."""
    rng = np.random.default_rng(RANDOM_SEED)

    results = []
    for noise_pct in NOISE_LEVELS:
        trial_mapes = []
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
        print(f"  Noise {noise_pct:>3d}%  ->  MAPE = {mean_mape:.2f}% +/- {std_mape:.2f}%")

    return pd.DataFrame(results)


# ── Visualization ───────────────────────────────────────────────────────
def plot_sensitivity(results_df: pd.DataFrame, save_path: str):
    """Publication-quality sensitivity chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = results_df["noise_pct"]
    y = results_df["mean_mape"]
    yerr_lo = y - results_df["min_mape"]
    yerr_hi = results_df["max_mape"] - y

    # Gradient fill under the curve
    ax.fill_between(x, 0, y, alpha=0.12, color="#4361EE")
    ax.plot(x, y, "o-", color="#4361EE", linewidth=2.5, markersize=9,
            markerfacecolor="white", markeredgewidth=2.5, label="Mean MAPE")
    ax.errorbar(x, y, yerr=[yerr_lo, yerr_hi], fmt="none",
                ecolor="#4361EE", elinewidth=1.2, capsize=5, capthick=1.2, alpha=0.6)

    # Industry benchmark band
    ax.axhspan(10, 15, color="#F72585", alpha=0.08, label="Industry Std (10–15%)")
    ax.axhline(15, color="#F72585", linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(10, color="#F72585", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(NOISE_LEVELS[-1] + 0.3, 12.5, "Industry\nStandard", fontsize=9,
            color="#F72585", ha="left", va="center", fontweight="bold")

    # Annotations per point
    for _, row in results_df.iterrows():
        offset_y = 0.6 if row["noise_pct"] != 0 else -0.8
        ax.annotate(f'{row["mean_mape"]:.2f}%',
                    xy=(row["noise_pct"], row["mean_mape"]),
                    xytext=(0, 18 if offset_y > 0 else -22),
                    textcoords="offset points", ha="center", fontsize=10,
                    fontweight="bold", color="#3A0CA3",
                    arrowprops=dict(arrowstyle="-", color="#3A0CA3", lw=0.8))

    ax.set_xlabel("Weather Forecast Error (% Std Dev of Gaussian Noise)", fontsize=12)
    ax.set_ylabel("Daytime MAPE (%)", fontsize=12)
    ax.set_title("Sensitivity Analysis: Model Accuracy vs Weather Uncertainty",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(NOISE_LEVELS)
    ax.set_xticklabels([f"{n}%" for n in NOISE_LEVELS])
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    print(f"  Chart saved -> {save_path}")
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Sensitivity Analysis — Weather Forecast Uncertainty")
    print("=" * 55)

    # 1. Load model
    forecaster = SolarForecaster(model_dir=MODEL_DIR)
    if not forecaster.load_model():
        print("ERROR: Trained model not found. Run train_solar_forecaster.py first.")
        return

    # 2. Load test data (same split as training)
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        return
    test_df = load_test_set(forecaster, DATA_PATH)

    # 3. Run Monte-Carlo sweep
    print("\nRunning sensitivity sweep (10 trials per noise level)...\n")
    results_df = run_sensitivity(forecaster, test_df)

    # 4. Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path  = os.path.join(RESULTS_DIR, "sensitivity_results.csv")
    plot_path = os.path.join(RESULTS_DIR, "sensitivity_analysis.png")

    results_df.to_csv(csv_path, index=False)
    print(f"\n  Results CSV saved -> {csv_path}")

    plot_sensitivity(results_df, plot_path)

    # 5. Summary
    baseline = results_df.loc[results_df["noise_pct"] == 0, "mean_mape"].values[0]
    worst    = results_df.loc[results_df["noise_pct"] == NOISE_LEVELS[-1], "mean_mape"].values[0]

    print("\n" + "=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    print(f"  Baseline (0% noise)    : {baseline:.2f}% MAPE")
    print(f"  Worst-case ({NOISE_LEVELS[-1]}% noise) : {worst:.2f}% MAPE")
    if worst < 15:
        print(f"  [PASSED] Model stays BELOW the 15% industry target even with {NOISE_LEVELS[-1]}% weather error.")
    else:
        print(f"  [WARNING] Model exceeds the 15% industry target at {NOISE_LEVELS[-1]}% weather error.")
    print("=" * 55)


if __name__ == "__main__":
    main()
