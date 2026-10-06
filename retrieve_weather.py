"""Add postcode/date weather features to a CSV dataset.

Weather data is retrieved from Open-Meteo's geocoding and historical archive
APIs. The archive provides daily values in the postcode's local timezone.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

GEOCODING_URL = "https://nominatim.openstreetmap.org/search"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
# 2001 is a Sydney GPO/PO-box postcode and has no OSM postal boundary.
POSTCODE_FALLBACKS = {"2001": (-33.8698, 151.2083)}
WEATHER_COLUMNS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "relative_humidity_2m_min",
    "relative_humidity_2m_max",
    "relative_humidity_2m_mean",
]
OUTPUT_WEATHER_COLUMNS = {
    "temperature_2m_max": "temperature_max",
    "temperature_2m_min": "temperature_min",
    "relative_humidity_2m_min": "relative_humidity_min",
    "relative_humidity_2m_max": "relative_humidity_max",
    "relative_humidity_2m_mean": "relative_humidity_mean",
}


def get_json(url: str, params: dict[str, object]) -> object:
    query = urlencode(params)
    request = Request(f"{url}?{query}", headers={"User-Agent": "postcode-weather/1.0"})
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def postcode_coordinates(
    postcode: str, fallback_query: str | None = None
) -> tuple[float, float]:
    data = get_json(
        GEOCODING_URL,
        {
            "postalcode": postcode,
            "country": "Australia",
            "format": "jsonv2",
            "limit": 1,
        },
    )
    if not isinstance(data, list) or not data:
        if postcode in POSTCODE_FALLBACKS:
            return POSTCODE_FALLBACKS[postcode]
        if fallback_query:
            query_variants = [
                fallback_query,
                fallback_query.replace(" Repatriation", "").replace(" General", ""),
            ]
            for query in query_variants:
                fallback = get_json(
                    GEOCODING_URL,
                    {
                        "q": f"{query}, Australia",
                        "format": "jsonv2",
                        "limit": 5,
                    },
                )
                matches = [
                    result
                    for result in fallback
                    if "Australia" in result.get("display_name", "")
                ]
                if matches:
                    return float(matches[0]["lat"]), float(matches[0]["lon"])
        raise ValueError(
            f"Australian postcode {postcode} was not found and no fallback location matched"
        )
    return float(data[0]["lat"]), float(data[0]["lon"])


def postcode_weather(
    postcode: str, start_date: str, end_date: str, fallback_query: str | None = None
) -> pd.DataFrame:
    latitude, longitude = postcode_coordinates(postcode, fallback_query)
    data = get_json(
        ARCHIVE_URL,
        {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ",".join(WEATHER_COLUMNS),
            "timezone": "auto",
        },
    )
    daily = data.get("daily")
    if not daily or "time" not in daily:
        raise ValueError(f"No weather data returned for postcode {postcode}")
    result = pd.DataFrame(daily).rename(columns={"time": "Date"})
    result.insert(0, "Postcode", postcode)
    return result


def add_weather(input_path: Path, output_path: Path) -> None:
    df = pd.read_csv(input_path, dtype={"Postcode": "string"})
    required = {"Postcode", "Date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    dates = pd.to_datetime(df["Date"], errors="raise")
    start_date = dates.min().strftime("%Y-%m-%d")
    end_date = dates.max().strftime("%Y-%m-%d")
    postcodes = df["Postcode"].str.strip().str.replace(r"\.0$", "", regex=True)
    hospital_names = (
        df.assign(Postcode=postcodes)
        .groupby("Postcode")["Hospital Name"]
        .first()
        if "Hospital Name" in df
        else pd.Series(dtype="string")
    )

    weather_frames = []
    for postcode in sorted(postcodes.dropna().unique()):
        time.sleep(1)
        weather_frames.append(
            postcode_weather(
                postcode,
                start_date,
                end_date,
                hospital_names.get(postcode),
            )
        )
    weather = pd.concat(weather_frames, ignore_index=True)
    weather["Date"] = pd.to_datetime(weather["Date"])
    df["Postcode"] = postcodes
    df["Date"] = dates
    df = df.drop(columns=[column for column in WEATHER_COLUMNS if column in df])
    enriched = df.merge(weather, on=["Postcode", "Date"], how="left", validate="many_to_one")
    enriched = enriched.rename(columns=OUTPUT_WEATHER_COLUMNS)
    enriched.to_csv(output_path, index=False)
    output_columns = [
        OUTPUT_WEATHER_COLUMNS.get(column, column) for column in WEATHER_COLUMNS
    ]
    missing_weather = enriched[output_columns].isna().any(axis=1).sum()
    print(f"Wrote {len(enriched):,} rows to {output_path}")
    if missing_weather:
        print(f"Warning: {missing_weather:,} rows have incomplete weather data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV containing Postcode and Date")
    parser.add_argument("output", type=Path, help="Destination enriched CSV")
    args = parser.parse_args()
    add_weather(args.input, args.output)


if __name__ == "__main__":
    main()
