from __future__ import annotations

import ssl
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


def load_energy_efficiency() -> pd.DataFrame:
    """
    Function to load the energy efficiency data from the UCI repository.
    Falls back to cached local copy if remote fetch fails.
    :return: Dataframe of energy efficiency data containing features + targets
    """
    cache_path = Path("data/energy_efficiency.csv")
    
    # Try to fetch from UCI repository
    try:
        ds = fetch_ucirepo(id=242)
        X = ds.data.features.copy()
        y = ds.data.targets.copy()

        # Merge into one DataFrame (features + targets)
        df = pd.concat([X, y], axis=1)

        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        # Map generic column names to meaningful names for the Energy Efficiency dataset
        # Based on UCI ML Repository: y1 = Heating Load, y2 = Cooling Load
        column_mapping = {"y1": "heating_load", "y2": "cooling_load"}
        df = df.rename(columns=column_mapping)
        
        # Cache the data for future use
        cache_path.parent.mkdir(exist_ok=True)
        df.to_csv(cache_path, index=False)
        
        return df
        
    except (ConnectionError, ssl.SSLError, Exception) as e:
        # Fall back to cached version if available
        if cache_path.exists():
            print(f"Warning: Could not fetch from UCI repository ({e}). Using cached data.")
            return pd.read_csv(cache_path)
        else:
            raise RuntimeError(
                f"Failed to fetch data from UCI repository and no cached data found at {cache_path}. "
                f"Original error: {e}"
            )


if __name__ == "__main__":
    df = load_energy_efficiency()
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head())
