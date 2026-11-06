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
date = st.date_input("Date", value=features.DEFAULT_DATE.date(),
                     min_value=features.FIRST_DATE.date(), max_value=features.LAST_DATE.date())
hour = st.slider("Hour of day", 0, 23, 8)

weather = None
if st.checkbox("Enter weather myself (otherwise the recorded weather for that date is used)"):
    columns = st.columns(4)
    weather = {
        "temperature_2m": columns[0].number_input("Temperature (C)", -30.0, 45.0, 15.0),
        "precipitation": columns[1].number_input("Rain (mm/h)", 0.0, 100.0, 0.0),
        "snowfall": columns[2].number_input("Snow (cm/h)", 0.0, 30.0, 0.0),
        "wind_speed_10m": columns[3].number_input("Wind (km/h)", 0.0, 150.0, 10.0),
    }

try:
    day = features.predict_day(model, route, str(date), weather)
except features.InputError as error:
    st.error(str(error))
    st.stop()

predicted = int(day.loc[day["hour"] == hour, "predicted_riders"].iloc[0])
st.metric(f"Predicted riders on {route} at {hour}:00", predicted)

st.subheader("Predicted riders through the day")
chart = day.set_index("hour")["predicted_riders"]
st.bar_chart(chart)

with st.expander("About this model and its limits"):
    st.write(
        "Gradient Boosting model trained on MTA bus ridership, MTA schedules and weather for 24 routes, "
        "from eight 2-week periods between January and September 2025. On the last 17 days it was held out, "
        "the typical error was about 65 riders per route-hour (about 14.5% of average ridership), "
        "16% better than a simple average-profile baseline."
    )
    st.write(
        "Limits: it only knows these 24 routes and dates from 2025-01-06 to 2025-09-28. The data are from New York "
        "because Karachi has no public passenger-count data. The model does not know about service changes "
        "(for example, three Queens routes lost about 35% of riders in summer 2025, and the model is less accurate there). "
        "Weather changes the prediction a little, but partly because it hints at the season, so extreme weather "
        "on an unlikely date can give odd results."
    )
