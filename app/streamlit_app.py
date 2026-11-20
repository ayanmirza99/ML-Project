from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

import features

HERE = Path(__file__).resolve().parent
MODEL_PATH = next(p for p in (HERE.parent / "models" / "planning_model.joblib", HERE / "planning_model.joblib") if p.exists())

st.set_page_config(page_title="TransitPulse", page_icon="🚌")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

st.title("TransitPulse")
st.write("Predict hourly bus ridership for one MTA route in New York, to help planners decide where buses are needed.")

route = st.selectbox("Route", features.ROUTES)
date = st.date_input("Date", value=features.default_date().date(),
                     min_value=features.FIRST_DATE.date(), max_value=features.LAST_DATE.date())
hour = st.slider("Hour of day", 0, 23, 8)

weather = None
if st.checkbox("Enter the weather myself (otherwise: recorded weather, or Open-Meteo's forecast for the next 16 days)"):
    columns = st.columns(4)
    weather = {
        "temperature_2m": columns[0].number_input("Temperature (C)", -30.0, 45.0, 15.0),
        "precipitation": columns[1].number_input("Rain (mm/h)", 0.0, 100.0, 0.0),
        "snowfall": columns[2].number_input("Snow (cm/h)", 0.0, 30.0, 0.0),
        "wind_speed_10m": columns[3].number_input("Wind (km/h)", 0.0, 150.0, 10.0),
    }

try:
    result = features.predict(model, route, str(date), hour, weather)
except features.InputError as error:
    st.error(str(error))
    st.stop()

low, high = result["likely_range"]
st.metric(f"Predicted riders on {route} at {hour}:00 on {result['date']}",
          f"{result['predicted_riders']:,} ± {result['typical_error']}")
st.caption(f"Likely between {low:,} and {high:,} riders (± the model's typical error on validation data). "
           f"Weather used: {result['weather_source']}.")
for warning in result["warnings"]:
    st.warning(warning)

st.subheader("Predicted riders through the day")
chart = pd.DataFrame(result["daily_profile"]).set_index("hour")["predicted_riders"]
st.bar_chart(chart)

with st.expander("About this model and its limits"):
    st.write(
        "Gradient Boosting model trained on MTA bus ridership, MTA schedules and weather for 24 routes, "
        "from eight 2-week periods between January and September 2025. On the last 17 days it was held out, "
        "the typical error was about 65 riders per route-hour (about 14.5% of average ridership), "
        "16% better than a simple average-profile baseline. The ± range uses the typical error on validation data (56 riders)."
    )
    st.write(
        f"Dates from {features.FIRST_DATE.date()} to {features.LAST_DATE.date()} are accepted. For 2025 dates the recorded weather "
        "is used; for later dates the weather comes from Open-Meteo (forecasts reach 16 days ahead; further ahead, enter the "
        "weather yourself). The model never saw March or October to December, and it does not know about service changes "
        "after September 2025 (three Queens routes already lost about 35% of riders in summer 2025), so treat those "
        "predictions as rough. The data are from New York because Karachi has no public passenger-count data. "
        "Weather data by Open-Meteo.com (CC BY 4.0)."
    )
