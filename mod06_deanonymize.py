import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    if not isinstance(anon_df, pd.DataFrame) or not isinstance(aux_df, pd.DataFrame):
        raise TypeError("anon_df and aux_df must both be pandas DataFrames.")
    if "anon_id" not in anon_df.columns:
        raise ValueError("anon_df is missing required column 'anon_id'.")
    if "name" not in aux_df.columns:
        raise ValueError("aux_df is missing required column 'name'.")

    qis = ["age", "gender", "zip3"]
    missing = [c for c in qis if c not in anon_df.columns or c not in aux_df.columns]
    if missing:
        raise ValueError(f"Missing quasi-identifier columns: {missing}")

    anon = anon_df.dropna(subset=qis).copy()
    aux = aux_df.dropna(subset=qis).copy()

    for df in (anon, aux):
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df["gender"] = df["gender"].astype(str).str.strip().str.lower()
        df["zip3"] = (
            df["zip3"].astype(str).str.strip()
            .str.replace(r"\.0$", "", regex=True)
            .str.zfill(3)
        )

    anon = anon.dropna(subset=["age"])
    aux = aux.dropna(subset=["age"])

    # Each row is a candidate (anon record, aux person) pair
    merged = anon[["anon_id"] + qis].merge(aux[["name"] + qis], on=qis, how="inner")

    # Unique match = the anon record matches exactly one aux person
    counts = merged.groupby("anon_id")["name"].transform("size")
    unique = merged[counts == 1]

    return (
        unique[["anon_id", "name"]]
        .rename(columns={"name": "matched_name"})
        .reset_index(drop=True)
    )


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    return len(matches_df) / len(anon_df)