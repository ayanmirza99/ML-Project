# Data sources and collection log

Dataset built on: 2026-11-06

## 1. MTA Bus Hourly Ridership: Beginning 2025
- Publisher: MTA via NY Open Data (data.ny.gov), dataset id gxb3-akrn
- Method: Socrata SODA API (`https://data.ny.gov/resource/gxb3-akrn.csv`) with server-side filtering and aggregation; ridership summed over payment method and fare class to one row per route per hour
- Windows: 2025-01-13 to 2025-01-26, 2025-02-10 to 2025-02-23, 2025-04-07 to 2025-04-20, 2025-05-19 to 2025-06-01, 2025-06-09 to 2025-06-22, 2025-07-07 to 2025-07-20, 2025-08-25 to 2025-09-07, 2025-09-15 to 2025-09-28 (plus 7 warm-up days each for lag features)
- Routes: M15+, Q58, Q65, B6, Q27, Q25, M101, Q44+, BX12+, B41, M86+, Q10, B35, B1, M4, M14D+, B38, BX9, BX19, B8, M96, BX36, S79+, BX7
- Licence: the dataset page states that no licence information was provided and that it is covered by different Terms of Use than Data.gov. Cite the NY Open Data terms of use.

## 2. MTA GTFS static schedule (bus)
- Used: yes
- Source: https://www.mta.info/developers (static GTFS feeds for the bus system)
- Method: trips per route per hour of day, split by weekday / Saturday / Sunday, from the first departure of every trip; plus stops, route length proxy and directions per route
- Caveat: the feed is a snapshot of the schedule valid when it was downloaded, which may differ from the 2025 service actually operated.

## 3. Open-Meteo historical weather
- Source: https://open-meteo.com (archive API), New York City coordinates
- Variables: temperature_2m, apparent_temperature, relative_humidity_2m, dew_point_2m, precipitation, snowfall, pressure_msl, cloud_cover, wind_speed_10m, wind_gusts_10m
- Licence: check https://open-meteo.com/en/terms and cite accordingly.

## Calendar features
- Holidays: Python `holidays` package (US federal holidays only; NY state observances such as Lincoln's Birthday are excluded because ridership on those days is normal)
- School days: hand-typed from the NYC public school calendar (https://schools.nyc.gov); verify dates.

## Final dataset
- Rows: 64512; columns: 36; features: 33 (excluding timestamp, window_id and the target `ridership`)
- Personal identifiers: none (aggregated ridership counts only).
