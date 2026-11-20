---
title: TransitPulse
emoji: 🚌
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
short_description: Hourly bus ridership predictions for 24 NYC bus routes
---

# TransitPulse

Predict hourly bus ridership for 24 MTA routes in New York. Pick a route, a date and an hour; the page shows the
predicted riders with a ± range and the predicted profile for the whole day. 2025 dates use recorded weather; later
dates use Open-Meteo's forecast (16 days ahead) or weather you enter. Weather data by Open-Meteo.com (CC BY 4.0).

This is a static Space: the Gradient Boosting model runs in your browser. Its 500 trees were exported from
scikit-learn to `model.json`, and `predict.js` adds them up exactly like scikit-learn; the predictions were
checked to be identical to the Python model.

Model card: https://huggingface.co/ayanjawed/transitpulse-planning-model
Code: https://github.com/ayanmirza99/ML-Project
Course project for CT-378 Machine Learning.
