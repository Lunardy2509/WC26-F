from __future__ import annotations

from functools import lru_cache

import pandas as pd

from config import DATA_DIR, participant_of

PARTICIPANTS_CSV = DATA_DIR / "participants.csv"


@lru_cache(maxsize=1)
def participant_table() -> pd.DataFrame:
    df = pd.read_csv(PARTICIPANTS_CSV, dtype={"subject": str})
    df["mass_kg"] = pd.to_numeric(df["mass_kg"], errors="coerce")
    dup = df.subject[df.subject.duplicated()].unique()
    if len(dup):
        raise ValueError(f"{PARTICIPANTS_CSV}: subject listed more than once: {', '.join(dup)}")
    return df


def body_mass_kg(subject: str) -> float:
    """Body mass for a data folder name (aliases resolved); NaN when the table has no entry."""
    df = participant_table()
    hit = df[df.subject == participant_of(subject)]
    return float(hit.mass_kg.iloc[0]) if len(hit) else float("nan")