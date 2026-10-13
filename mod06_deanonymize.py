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

    for c in qis:
        if c == "zip3":
            # Force to zero-padded 3-character strings, e.g. 65, 65.0, "065" -> "065"
            anon[c] = anon[c].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(3)
            aux[c] = aux[c].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(3)
        elif pd.api.types.is_string_dtype(anon[c]) or pd.api.types.is_object_dtype(anon[c]):
            anon[c] = anon[c].astype(str).str.strip().str.lower()
            aux[c] = aux[c].astype(str).str.strip().str.lower()

    merged = anon[["anon_id"] + qis].merge(aux[["name"] + qis], on=qis, how="inner")

    counts = merged.groupby("anon_id")["name"].transform("size")
    unique = merged[counts == 1]
    unique = unique[~unique["name"].duplicated(keep=False)]

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
    if not isinstance(matches_df, pd.DataFrame) or not isinstance(anon_df, pd.DataFrame):
        raise TypeError("matches_df and anon_df must both be pandas DataFrames.")
    if "anon_id" not in matches_df.columns:
        raise ValueError("matches_df is missing required column 'anon_id'.")
    if len(anon_df) == 0:
        raise ValueError("anon_df is empty; the rate is undefined.")

    return matches_df["anon_id"].nunique() / len(anon_df)