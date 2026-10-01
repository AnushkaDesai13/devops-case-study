"""
dataset_generator.py
Generates a synthetic food delivery dataset with 500 rows.
"""

import numpy as np
import pandas as pd

def generate_dataset(n=500, seed=42, save_path="delivery_data.csv"):
    np.random.seed(seed)

    distance_km       = np.round(np.random.uniform(1, 20, n), 2)
    traffic_level     = np.random.choice(["Low", "Medium", "High"], n, p=[0.3, 0.4, 0.3])
    weather           = np.random.choice(["Clear", "Rainy", "Stormy"], n, p=[0.6, 0.3, 0.1])
    prep_time         = np.random.randint(5, 30, n)           # minutes
    partner_rating    = np.round(np.random.uniform(2.5, 5.0, n), 1)
    partner_available = np.random.choice([0, 1], n, p=[0.15, 0.85])
    premium_order     = np.random.choice([0, 1], n, p=[0.7, 0.3])

    # Simulate realistic delivery time
    traffic_map = {"Low": 0, "Medium": 5, "High": 12}
    weather_map = {"Clear": 0, "Rainy": 7, "Stormy": 15}
    traffic_add = np.array([traffic_map[t] for t in traffic_level])
    weather_add = np.array([weather_map[w] for w in weather])
    noise       = np.random.randint(-3, 6, n)

    delivery_time = (
        prep_time
        + (distance_km * 2.5).astype(int)
        + traffic_add
        + weather_add
        + noise
    )
    delivery_time = np.clip(delivery_time, 10, 120)

    df = pd.DataFrame({
        "distance_km":            distance_km,
        "traffic_level":          traffic_level,
        "weather":                weather,
        "prep_time":              prep_time,
        "partner_rating":         partner_rating,
        "partner_available":      partner_available,
        "premium_order":          premium_order,
        "delivery_time_minutes":  delivery_time
    })

    df.to_csv(save_path, index=False)
    print(f"[DatasetGenerator] Saved {n} rows → {save_path}")
    return df


if __name__ == "__main__":
    generate_dataset()
