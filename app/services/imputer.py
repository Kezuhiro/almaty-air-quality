
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances


class SpatialImputer:
    def __init__(self, radius_steps=[1.0, 2.0, 3.0, 5.0]):
        self.radius_steps = radius_steps

    def impute(
        self,
        target_name: str,
        target_coords: dict,
        live_df: pd.DataFrame,
        feature_col: str = "pm25",
    ) -> float:
        if live_df.empty:
            return float("nan")

        target_rad = np.radians([[target_coords["lat"], target_coords["lon"]]])
        live_rad = np.radians(live_df[["lat", "lon"]].values)
        distances = haversine_distances(target_rad, live_rad)[0] * 6371.0

        for radius_km in self.radius_steps:
            valid_indices = np.where(distances <= radius_km)[0]
            print(
                f"[IMPUTE] {target_name} | radius={radius_km}km | "
                f"neighbors={len(valid_indices)}"
            )

            if len(valid_indices) == 0:
                continue

            neighbor_values = live_df.iloc[valid_indices][feature_col].astype(float).values
            neighbor_distances = distances[valid_indices]
            valid_values = np.isfinite(neighbor_values) & (neighbor_values > 0.5)

            if not valid_values.any():
                continue

            neighbor_values = neighbor_values[valid_values]
            neighbor_distances = neighbor_distances[valid_values]
            weights = 1.0 / (neighbor_distances + 0.01)
            return float(np.average(neighbor_values, weights=weights))

        print(f"[IMPUTE-FALLBACK] {target_name} -> city mean")
        valid_city_values = live_df[feature_col].astype(float)
        valid_city_values = valid_city_values[
            np.isfinite(valid_city_values) & (valid_city_values > 0.5)
        ]

        if valid_city_values.empty:
            return float("nan")

        return float(valid_city_values.mean())
