// TransitPulse planning model, running in the browser (used by the Hugging Face static Space).
//
// The model is the scikit-learn pipeline saved in models/planning_model.joblib:
//   1. one-hot encoding of bus_route and borough (the other 18 inputs pass through unchanged),
//   2. a HistGradientBoostingRegressor with 500 trees.
// app/build_hf_space.py exports that pipeline to model.json (the trees) and the lookup tables
// (route facts, schedule, recorded weather, calendar) to data.json.
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

  function checkRoute(data, route) {
    if (!Object.prototype.hasOwnProperty.call(data.routes, route)) {
      throw new InputError(`Unknown route '${route}'. Choose one of: ${data.route_names.join(", ")}.`);
    }
  }

  function checkDate(data, date) {
    const text = String(date);
    const parsed = new Date(text + "T00:00:00Z");
    if (!/^\d{4}-\d{2}-\d{2}$/.test(text) || Number.isNaN(parsed.getTime()) || parsed.toISOString().slice(0, 10) !== text) {
      throw new InputError(`Date '${date}' is not valid. Use the format YYYY-MM-DD.`);
    }
    if (!Object.prototype.hasOwnProperty.call(data.calendar, text)) {
      throw new InputError(`Date must be between ${data.first_date} and ${data.last_date}, the period the model was trained on.`);
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
      if (!Object.prototype.hasOwnProperty.call(data.weather_limits, name)) {
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

  // ---- building the model's input rows (mirrors features.day_features) --------------------

  function dayRows(data, route, date, weather) {
    checkRoute(data, route);
    const day = checkDate(data, date);
    const chosenWeather = checkWeather(data, weather);

    const calendar = data.calendar[day];
    const dayType = calendar.day_of_week < 5 ? "weekday" : calendar.day_of_week === 5 ? "saturday" : "sunday";
    const schedule = data.schedule[route][dayType];
    const recorded = data.weather[day];

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
      for (const column of Object.keys(data.weather_limits)) {
        row[column] = column in chosenWeather ? chosenWeather[column] : recorded[column][hour];
      }
      rows.push(row);
    }
    return rows;
  }

  // Predicted riders for each hour of the day: never below 0, rounded to whole riders.
  function predictDay(model, data, route, date, weather) {
    return dayRows(data, route, date, weather).map((row) => ({
      hour: row.hour,
      predicted_riders: roundHalfEven(Math.max(rawPredict(model, encode(model, row)), 0)),
    }));
  }

  function predictHour(model, data, route, date, hour, weather) {
    checkHour(hour);
    const profile = predictDay(model, data, route, date, weather);
    return { route, date, hour, predicted_riders: profile[hour].predicted_riders, daily_profile: profile };
  }

  const api = { InputError, treeValue, rawPredict, encode, roundHalfEven, dayRows, predictDay, predictHour };
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.TransitPulse = api;
})(typeof window !== "undefined" ? window : globalThis);
