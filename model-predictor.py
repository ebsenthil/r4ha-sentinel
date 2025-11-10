from r4ha_xgboost_model import R4HAPredictor

# Load trained model
predictor = R4HAPredictor()
predictor.load_model('r4ha_model.json')

# Current system state (from your mainframe)
current_state = {
    'msu_current': 850,
    'r4ha_msu': 820,
    'cpu_utilization_pct': 75.5,
    'r4ha_lag_1h': 800,
    'r4ha_lag_2h': 780,
    'msu_lag_1h': 830,
    'msu_rolling_mean_2h': 835,
    'msu_rolling_mean_4h': 825,
    'hour_of_day': 14,
    'day_of_week': 3,
    'is_batch_window': 0,
    'batch_jobs_running': 18,
    'batch_cpu_seconds': 210
}

# Predict R4HA 4 hours from now
predicted_r4ha = predictor.predict_real_time(current_state)
print(f"Predicted R4HA in 4 hours: {predicted_r4ha:.2f} MSU")
```

## Expected Results:

With synthetic data, you should see:
```
TEST SET:
  RMSE: 35-45 MSU
  MAE:  25-35 MSU
  R²:   0.80-0.85
  MAPE: 4-6%

Model Accuracy: ~94-96%
```

## Feature Importance (Expected):
```
1. r4ha_lag_1h         (25-30%) ← Most important!
2. r4ha_lag_2h         (15-20%)
3. msu_rolling_mean_2h (10-15%)
4. r4ha_msu            (8-12%)
5. batch_jobs_running  (6-10%)
6. hour_of_day         (5-8%)
7. msu_current         (4-6%)
...
