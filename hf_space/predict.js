// TransitPulse planning model, running in the browser (used by the Hugging Face static Space).
//
// The model is the scikit-learn pipeline saved in models/planning_model.joblib:
//   1. one-hot encoding of bus_route and borough (the other 18 inputs pass through unchanged),
//   2. a HistGradientBoostingRegressor with 500 trees.
// app/build_hf_space.py exports that pipeline to model.json (the trees) and the lookup tables
// (route facts, schedule, recorded 2025 weather, calendar) to data.json.
// This file rebuilds the input row exactly like app/features.py and adds up the trees exactly
// like scikit-learn does. app/verify_static_space.py checks that the results are identical.
// The same file runs in the browser (window.TransitPulse) and in Node.js (module.exports) for the check.

(function (root) {
  "use strict";

  // ---- the model --------------------------------------------------------------------------

  // Walk one tree from the root to a leaf. feature = -1 marks a leaf.
  // scikit-learn's rule: go left when the value is <= the threshold (missing values follow missing_left).
  function treeValue(tree, x) {
    let node = 0;
    while (tree.feature[node] !== -1) {
      const v = x[tree.feature[node]];
      const goLeft = Number.isNaN(v) ? tree.missing_left[node] === 1 : v <= tree.threshold[node];
      node = goLeft ? tree.left[node] : tree.right[node];
    }
    return tree.value[node];
  }

  // Prediction = starting value (training mean) + the sum of all 500 tree outputs, in the same order as scikit-learn.
  function rawPredict(model, x) {
    let total = 0;
    total += model.baseline;
    for (const tree of model.trees) total += treeValue(tree, x);
    return total;
  }

  // Same column order as the ColumnTransformer: one-hot columns first, then the numeric columns.
  function encode(model, row) {
    const x = [];
    for (const [column, categories] of model.onehot) {
      for (const category of categories) x.push(row[column] === category ? 1 : 0);
    }
    for (const column of model.numeric) x.push(row[column]);
    return x;
  }

  // numpy.round rounds halves to the nearest even number (2.5 -> 2); Math.round would give 3.
  function roundHalfEven(value) {
    const down = Math.floor(value);
    const diff = value - down;
    if (diff < 0.5) return down;
    if (diff > 0.5) return down + 1;
    return down % 2 === 0 ? down : down + 1;
  }

  // ---- input checks (same messages as app/features.py) ------------------------------------

  class InputError extends Error {}

  function pyFloat(n) {
    // Python prints the float limits as -30.0, 45.0, ...
    return Number.isInteger(n) ? n.toFixed(1) : String(n);
  }

  const has = (obj, key) => Object.prototype.hasOwnProperty.call(obj, key);

  function checkRoute(data, route) {
    if (!has(data.routes, route)) {
      throw new InputError(`Unknown route '${route}'. Choose one of: ${data.route_names.join(", ")}.`);
    }
  }

  function checkDate(data, date) {
    const text = String(date);
    const parsed = new Date(text + "T00:00:00Z");
    if (!/^\d{4}-\d{2}-\d{2}$/.test(text) || Number.isNaN(parsed.getTime()) || parsed.toISOString().slice(0, 10) !== text) {
      throw new InputError(`Date '${date}' is not valid. Use the format YYYY-MM-DD.`);
    }
    if (!has(data.calendar, text)) {
      throw new InputError(`Date must be between ${data.first_date} and ${data.last_date}.`);
    }
    return text;
  }

  function checkHour(hour) {
    if (typeof hour !== "number" || !Number.isInteger(hour) || hour < 0 || hour > 23) {
      throw new InputError("Hour must be a whole number from 0 to 23.");
    }
    return hour;
  }

  function checkWeather(data, weather) {
    const clean = {};
    for (const [name, value] of Object.entries(weather || {})) {
      if (value === null || value === undefined) continue;
      if (!has(data.weather_limits, name)) {
        throw new InputError(`Unknown weather field '${name}'. Allowed: ${Object.keys(data.weather_limits).join(", ")}.`);
      }
      const [low, high] = data.weather_limits[name];
      if (typeof value !== "number" || !Number.isFinite(value) || value < low || value > high) {
        throw new InputError(`${name} must be a number between ${pyFloat(low)} and ${pyFloat(high)}.`);
      }
      clean[name] = value;
    }
    return clean;
  }

  // ---- dates and weather (mirrors features.fetch_weather / day_weather / date_warnings) -----

  const DAY_MS = 86400000;
  const dayNumber = (text) => Date.parse(text + "T00:00:00Z") / DAY_MS;

  function todayInNewYork() {
    return new Date().toLocaleDateString("en-CA", { timeZone: "America/New_York" }); // YYYY-MM-DD
  }

  function addDays(text, days) {
    return new Date(Date.parse(text + "T00:00:00Z") + days * DAY_MS).toISOString().slice(0, 10);
  }

  function defaultDate() {
    return addDays(todayInNewYork(), 1);
  }

  const weatherCache = {};

  // Hourly weather for a date after the saved 2025 file, from Open-Meteo (CC BY 4.0).
  async function fetchWeather(data, date, today = todayInNewYork()) {
    const cached = weatherCache[date];
    if (cached && Date.now() - cached.time < 3600000) return cached.result;
    const ow = data.open_meteo;
    if (dayNumber(date) > dayNumber(today) + ow.forecast_days - 1) {
      throw new InputError(`No weather forecast exists yet for ${date} (forecasts reach ${ow.forecast_days} days ahead). Enter the weather yourself.`);
    }
    const archive = dayNumber(date) < dayNumber(today) - ow.archive_delay_days;
    const params = new URLSearchParams({
      latitude: ow.latitude, longitude: ow.longitude, hourly: Object.keys(data.weather_limits).join(","),
      timezone: "America/New_York", start_date: date, end_date: date,
    });
    const problem = new InputError(`Could not get the weather for ${date} from Open-Meteo. Try again, or enter the weather yourself.`);
    let hourly;
    try {
      const response = await fetch(`${archive ? ow.archive_url : ow.forecast_url}?${params}`);
      if (!response.ok) throw problem;
      hourly = (await response.json()).hourly;
      if (!hourly) throw problem;
    } catch (error) {
      throw problem;
    }
    const values = {};
    for (const column of Object.keys(data.weather_limits)) {
      const byHour = {};
      (hourly[column] || []).forEach((value, i) => {
        const stamp = hourly.time[i];
        const hour = Number(stamp.slice(11, 13));
        if (stamp.startsWith(date) && !(hour in byHour) && value !== null) byHour[hour] = value;
      });
      const known = Object.values(byHour);
      if (known.length === 0) throw problem;
      // a missing hour (for example when clocks change) takes the value of the hour before it, or after it at midnight
      const series = [];
      for (let hour = 0; hour < 24; hour++) series.push(hour in byHour ? byHour[hour] : null);
      for (let hour = 0; hour < 24; hour++) {
        if (series[hour] === null) series[hour] = hour > 0 ? series[hour - 1] : series.find((v) => v !== null);
      }
      values[column] = series;
    }
    const result = { values, source: archive ? "archive" : "forecast" };
    weatherCache[date] = { time: Date.now(), result };
    return result;
  }

  // 24 hourly values per weather column and where they came from. `fetched` is fetchWeather's result when it was needed.
  function dayWeather(data, date, weather, fetched) {
    const columns = Object.keys(data.weather_limits);
    let base = null, source = null;
    if (has(data.weather, date)) {
      base = data.weather[date];
      source = "recorded weather (Open-Meteo)";
    } else if (!columns.every((c) => c in weather)) {
      base = fetched.values;
      source = fetched.source === "archive" ? "Open-Meteo historical weather" : "Open-Meteo forecast";
    }
    const out = {};
    for (const c of columns) out[c] = c in weather ? Array(24).fill(weather[c]) : base[c];
    if (Object.keys(weather).length === columns.length) source = "your weather";
    else if (Object.keys(weather).length > 0) source += ", with your values where given";
    return { columns: out, source };
  }

  function needsFetch(data, date, weather) {
    return !has(data.weather, date) && !Object.keys(data.weather_limits).every((c) => c in weather);
  }

  function dateWarnings(data, date) {
    const months = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
      "October", "November", "December"];
    const month = Number(date.slice(5, 7));
    const warnings = [];
    if (!data.trained_months.includes(month)) {
      warnings.push(`The model never saw ${months[month - 1]} in training (it learned from January, February and April to September 2025), so treat this as a rough guess.`);
    }
    if (dayNumber(date) > dayNumber(data.last_recorded_date)) {
      warnings.push("This date is after the training data (January to September 2025); bus service may have changed since then.");
    }
    return warnings;
  }

  // ---- building the model's input rows (mirrors features.day_features) --------------------

  function dayRows(data, route, date, weather, fetched) {
    const calendar = data.calendar[date];
    const dayType = calendar.day_of_week < 5 ? "weekday" : calendar.day_of_week === 5 ? "saturday" : "sunday";
    const schedule = data.schedule[route][dayType];
    const { columns, source } = dayWeather(data, date, weather, fetched);

    const rows = [];
    for (let hour = 0; hour < 24; hour++) {
      const row = {
        bus_route: route,
        hour: hour,
        day_of_week: calendar.day_of_week,
        is_weekend: calendar.day_of_week >= 5 ? 1 : 0,
        hour_sin: data.hour_sin[hour],
        hour_cos: data.hour_cos[hour],
        is_am_peak: hour >= 7 && hour <= 9 ? 1 : 0,
        is_pm_peak: hour >= 16 && hour <= 19 ? 1 : 0,
        is_holiday: calendar.is_holiday,
        is_school_day: calendar.is_school_day,
      };
      Object.assign(row, data.routes[route]);
      data.schedule_columns.forEach((column, i) => { row[column] = schedule[hour][i]; });
      for (const column of Object.keys(columns)) row[column] = columns[column][hour];
      rows.push(row);
    }
    return { rows, source };
  }

  // Everything the page shows; same fields as features.predict. Checks run in the same order as the Python app.
  async function predict(model, data, route, date, hour, weather, today = todayInNewYork()) {
    checkHour(hour);
    checkRoute(data, route);
    const day = checkDate(data, date);
    const chosen = checkWeather(data, weather);
    const fetched = needsFetch(data, day, chosen) ? await fetchWeather(data, day, today) : null;
    const { rows, source } = dayRows(data, route, day, chosen, fetched);
    const profile = rows.map((row) => roundHalfEven(Math.max(rawPredict(model, encode(model, row)), 0)));
    const value = profile[hour];
    return {
      route, date: day, hour,
      predicted_riders: value,
      typical_error: data.typical_error,
      likely_range: [Math.max(0, value - data.typical_error), value + data.typical_error],
      daily_profile: profile.map((p, h) => ({ hour: h, predicted_riders: p })),
      warnings: dateWarnings(data, day),
      weather_source: source,
    };
  }

  const api = { InputError, treeValue, rawPredict, encode, roundHalfEven, dayRows, predict, fetchWeather,
    todayInNewYork, defaultDate };
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.TransitPulse = api;
})(typeof window !== "undefined" ? window : globalThis);
