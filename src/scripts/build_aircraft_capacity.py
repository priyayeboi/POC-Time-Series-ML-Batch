"""Build src/datasets/aircraft_capacity_jfk_2025.csv: aircraft model and max seats per BTS tail number.

Models come from the OpenSky aircraft database, joined on registration. OpenSky has no seat
counts for these aircraft, so max_seats comes from a per-typecode table of maximum certified
passenger capacity (exit limit). Those values were compiled from published manufacturer
specifications without being re-checked online, so verify them before relying on exact numbers.

Usage: venv/bin/python src/scripts/build_aircraft_capacity.py
"""
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BTS_CSV = ROOT / "src/datasets/bts_flights_jfk_2025.csv"
OUT_CSV = ROOT / "src/datasets/aircraft_capacity_jfk_2025.csv"
OPENSKY_URL = "https://opensky-network.org/datasets/metadata/aircraftDatabase.csv"
OPENSKY_CSV = ROOT / ".cache/opensky_aircraftDatabase.csv"

SEATS_SOURCE = "Manufacturer exit limit, compiled without online verification; verify"

# typecode -> (max certified passengers, body, category). None = not sourced, left blank.
TYPES = {
    "A319": (160, "narrowbody", "mainline"),
    "A320": (180, "narrowbody", "mainline"),
    "A20N": (194, "narrowbody", "mainline"),
    "A321": (220, "narrowbody", "mainline"),
    "A21N": (244, "narrowbody", "mainline"),
    "A332": (406, "widebody", "mainline"),
    "A333": (440, "widebody", "mainline"),
    "A339": (460, "widebody", "mainline"),
    "A359": (440, "widebody", "mainline"),
    "B738": (189, "narrowbody", "mainline"),
    "B739": (220, "narrowbody", "mainline"),
    "B38M": (210, "narrowbody", "mainline"),
    "B39M": (220, "narrowbody", "mainline"),
    "B752": (239, "narrowbody", "mainline"),
    "B753": (295, "narrowbody", "mainline"),
    "B763": (351, "widebody", "mainline"),
    "B764": (375, "widebody", "mainline"),
    "B772": (440, "widebody", "mainline"),
    "B77W": (550, "widebody", "mainline"),
    "B788": (None, "widebody", "mainline"),
    "B789": (420, "widebody", "mainline"),
    "BCS1": (135, "narrowbody", "mainline"),
    "BCS3": (160, "narrowbody", "mainline"),
    "E170": (80, "narrowbody", "regional"),
    "E75L": (88, "narrowbody", "regional"),
    "E190": (114, "narrowbody", "regional"),
    "CRJ9": (90, "narrowbody", "regional"),
}

# OpenSky rows with a model string but no typecode.
MODEL_TO_TYPECODE = {
    "737-9": "B39M",
    "737-9GPER": "B739",
    "ERJ 170-200 LR": "E75L",
    "BD-500-1A11": "BCS3",
    "A320-251N": "A20N",
    "A321-271NX": "A21N",
    "A330-941": "A339",
}


def main():
    if not OPENSKY_CSV.exists():
        OPENSKY_CSV.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(OPENSKY_URL, OPENSKY_CSV)

    flights = pd.read_csv(BTS_CSV, usecols=["Tail_Number", "Reporting_Airline"], low_memory=False)
    tails = (flights.dropna(subset=["Tail_Number"])
             .groupby("Tail_Number")["Reporting_Airline"]
             .agg(lambda s: s.mode().iat[0])
             .rename("airline").reset_index())

    sky = pd.read_csv(OPENSKY_CSV, dtype=str, low_memory=False,
                      usecols=["registration", "manufacturername", "model", "typecode"])
    sky = sky[sky["registration"].isin(tails["Tail_Number"])].drop_duplicates("registration")

    out = tails.merge(sky, left_on="Tail_Number", right_on="registration", how="left")
    out["typecode"] = out["typecode"].fillna(out["model"].map(MODEL_TO_TYPECODE))
    out["matched"] = out["registration"].notna()
    # A match to a non-airliner type means the registration was reused; treat it as wrong.
    out["type_plausible"] = out["typecode"].isin(TYPES.keys())
    info = out["typecode"].map(TYPES)
    out["max_seats"] = info.map(lambda t: t[0] if isinstance(t, tuple) else None).astype("Int64")
    out["body"] = info.map(lambda t: t[1] if isinstance(t, tuple) else None)
    out["category"] = info.map(lambda t: t[2] if isinstance(t, tuple) else None)
    out["seats_source"] = out["max_seats"].notna().map({True: SEATS_SOURCE, False: ""})

    out = out.rename(columns={"Tail_Number": "tail_number", "manufacturername": "manufacturer"})
    cols = ["tail_number", "airline", "manufacturer", "model", "typecode", "max_seats",
            "body", "category", "seats_source", "matched", "type_plausible"]
    out[cols].sort_values("tail_number").to_csv(OUT_CSV, index=False)

    print(f"wrote {len(out)} tails to {OUT_CSV}")
    print(f"matched {out['matched'].sum()}, plausible type {out['type_plausible'].sum()}, "
          f"with max_seats {out['max_seats'].notna().sum()}")
    print("implausible matches:", out.loc[out["matched"] & ~out["type_plausible"],
                                          ["tail_number", "model"]].values.tolist())
    print("unmatched tails by airline:", out.loc[~out["matched"], "airline"].value_counts().to_dict())


if __name__ == "__main__":
    main()
