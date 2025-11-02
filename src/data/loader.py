from __future__ import annotations

import pandas as pd
from ucimlrepo import fetch_ucirepo


def load_energy_efficiency() -> pd.DataFrame:
    """
    Function to load the energy efficiency data from the UCI repository.
    :return: Dataframe of energy efficiency data containing features + targets
    """
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

    return df


if __name__ == "__main__":
    df = load_energy_efficiency()
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head())
