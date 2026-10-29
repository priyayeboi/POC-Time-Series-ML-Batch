"""Generate a SYNTHETIC stand-in for FAA ASPM hourly JFK airport metrics (2025).

The real ASPM reports require an FAA login, so this approximates what they would
contain. Weather drivers come from the real Open-Meteo hourly file so runway
configuration and capacity react plausibly to wind, visibility and precipitation.
Every value produced here is synthetic.

Usage: python3 src/scripts/generate_synthetic_aspm.py
"""
import csv
import math
import random
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

SEED = 2025
ROOT = Path(__file__).resolve().parents[2]
WEATHER_CSV = ROOT / "src/datasets/openmeteo_forecast_jfk_2025.csv"
OUT_CSV = ROOT / "src/datasets/faa_aspm_jfk_2025.csv"
NY = ZoneInfo("America/New_York")

# Common JFK configurations: (name, arrival runways, departure runways, heading deg,
# VMC arrival rate, VMC departure rate)
CONFIGS = [
    ("31R|31L", "31R", "31L", 314, 58, 50),
    ("22L|22R,31L", "22L", "22R,31L", 224, 56, 52),
    ("13L,22L|13R", "13L,22L", "13R", 134, 54, 46),
    ("4L,4R|4L", "4L,4R", "4L", 44, 52, 44),
]
PREFERRED_CALM = 1  # 22L|22R,31L in light winds

# Share of daily departures/arrivals per local hour (sums to ~1).
DEP_PROFILE = [0.004, 0.001, 0.0, 0.0, 0.0, 0.008, 0.050, 0.062, 0.058, 0.050,
               0.045, 0.048, 0.050, 0.052, 0.058, 0.062, 0.066, 0.070, 0.070,
               0.066, 0.058, 0.048, 0.042, 0.032]
ARR_PROFILE = [0.020, 0.010, 0.004, 0.002, 0.004, 0.018, 0.038, 0.040, 0.042, 0.044,
               0.046, 0.048, 0.050, 0.054, 0.058, 0.060, 0.062, 0.062, 0.060,
               0.058, 0.056, 0.054, 0.058, 0.052]
BASE_DAILY_DEPS = 620
MONTH_FACTOR = {1: 0.90, 2: 0.88, 3: 0.97, 4: 1.00, 5: 1.03, 6: 1.08,
                7: 1.10, 8: 1.09, 9: 1.00, 10: 1.01, 11: 0.98, 12: 1.02}
DOW_FACTOR = [1.02, 0.98, 0.99, 1.03, 1.05, 0.90, 1.00]  # Mon..Sun


def headwind(wind_dir, heading, speed):
    return speed * math.cos(math.radians(wind_dir - heading))


def pick_config(wind_dir, wind_kmh):
    if wind_kmh < 10:
        return PREFERRED_CALM
    return max(range(len(CONFIGS)), key=lambda i: headwind(wind_dir, CONFIGS[i][3], wind_kmh))


def fnum(value, default=0.0):
    return float(value) if value not in ("", None) else default


def main():
    rng = random.Random(SEED)
    with WEATHER_CSV.open() as f:
        weather = list(csv.DictReader(f))

    rows = []
    config_idx = PREFERRED_CALM
    pending_idx, pending_hours = None, 0
    dep_queue = arr_queue = 0.0

    for w in weather:
        t_local = datetime.fromisoformat(w["time_local"]).astimezone(NY)
        wind = fnum(w["wind_speed_10m"])
        gust = fnum(w["wind_gusts_10m"])
        vis_m = fnum(w["visibility"], 20000)
        precip = fnum(w["precipitation"])
        snow = fnum(w["snowfall"])
        low_cloud = fnum(w["cloud_cover_low"])

        # Runway configuration changes only after the new choice persists 2 hours.
        wanted = pick_config(fnum(w["wind_direction_10m"]), wind)
        if wanted != config_idx:
            pending_hours = pending_hours + 1 if wanted == pending_idx else 1
            pending_idx = wanted
            if pending_hours >= 2:
                config_idx, pending_idx, pending_hours = wanted, None, 0
        else:
            pending_idx, pending_hours = None, 0
        name, arr_rwys, dep_rwys, _, aar_vmc, adr_vmc = CONFIGS[config_idx]

        imc = vis_m < 4800 or (low_cloud > 85 and precip > 0.2)
        met = "IMC" if imc else "VMC"
        factor = 0.78 if imc else 1.0
        if gust > 55:
            factor *= 0.85
        if snow > 0.2:
            factor *= 0.70
        elif precip > 2.0:
            factor *= 0.90
        aar = int(round(aar_vmc * factor))
        adr = int(round(adr_vmc * factor))

        h = t_local.hour
        demand = BASE_DAILY_DEPS * MONTH_FACTOR[t_local.month] * DOW_FACTOR[t_local.weekday()]
        sched_deps = max(0, round(demand * DEP_PROFILE[h] * rng.uniform(0.92, 1.08)))
        sched_arrs = max(0, round(demand * ARR_PROFILE[h] * rng.uniform(0.92, 1.08)))

        # Bad weather cancels a share of the schedule.
        cancel_rate = 0.0
        if snow > 0.5:
            cancel_rate = 0.25
        elif imc or gust > 60:
            cancel_rate = 0.06
        cancelled_deps = round(sched_deps * cancel_rate * rng.uniform(0.5, 1.5))
        cancelled_arrs = round(sched_arrs * cancel_rate * rng.uniform(0.5, 1.5))

        dep_demand = sched_deps - cancelled_deps + dep_queue
        arr_demand = sched_arrs - cancelled_arrs + arr_queue
        actual_deps = int(min(dep_demand, adr * rng.uniform(0.92, 1.0)))
        actual_arrs = int(min(arr_demand, aar * rng.uniform(0.92, 1.0)))
        dep_queue = max(0.0, dep_demand - actual_deps)
        arr_queue = max(0.0, arr_demand - actual_arrs)

        utilization = actual_deps / adr if adr else 0.0
        taxi_out = 16 + 10 * utilization ** 2 + 0.6 * dep_queue + (12 if snow > 0.2 else 0)
        taxi_out += rng.gauss(0, 1.5)
        taxi_in = 8 + 3 * (actual_arrs / aar if aar else 0) + rng.gauss(0, 0.8)
        avg_dep_delay = 4 + 2.5 * dep_queue + (15 if imc else 0) * utilization + rng.gauss(0, 2)
        pct_ontime = 100 * math.exp(-max(avg_dep_delay, 0) / 45) * rng.uniform(0.95, 1.0)

        rows.append({
            "hour_local": t_local.isoformat(),
            "hour_utc": w["time_utc"],
            "local_date": t_local.date().isoformat(),
            "local_hour": h,
            "runway_config": name,
            "arrival_runways": arr_rwys,
            "departure_runways": dep_rwys,
            "met_condition": met,
            "airport_arrival_rate": aar,
            "airport_departure_rate": adr,
            "scheduled_departures": sched_deps,
            "scheduled_arrivals": sched_arrs,
            "cancelled_departures": cancelled_deps,
            "cancelled_arrivals": cancelled_arrs,
            "actual_departures": actual_deps,
            "actual_arrivals": actual_arrs,
            "departure_queue_end": round(dep_queue),
            "arrival_queue_end": round(arr_queue),
            "avg_taxi_out_min": round(max(taxi_out, 8), 1) if actual_deps else "",
            "avg_taxi_in_min": round(max(taxi_in, 4), 1) if actual_arrs else "",
            "avg_departure_delay_min": round(max(avg_dep_delay, 0), 1) if actual_deps else "",
            "pct_ontime_departures": round(pct_ontime, 1) if actual_deps else "",
            "is_synthetic": True,
        })

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
