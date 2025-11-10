## Key Differences from 4-Hour Model:

| Aspect | 1-Hour Model | 4-Hour Model |
|--------|--------------|--------------|
| **Target calculation** | `shift(-4)` | `shift(-16)` |
| **Accuracy** | ~96-97% | ~90-92% |
| **Typical error** | ±18 MSU | ±35 MSU |
| **Use case** | Immediate alerts | Strategic planning |
| **Lead time** | 1 hour warning | 4 hour warning |

---

## Files Created:
```
📁 Your Project Folder
├── r4ha_data_generator_1h.py     ← Script 1: Generate data
├── r4ha_train_model_1h.py        ← Script 2: Train model
├── r4ha_predict_1h.py            ← Script 3: Make predictions
│
├── r4ha_data_1h.csv              ← Generated training data
├── r4ha_model_1h.json            ← Trained XGBoost model
├── r4ha_model_1h_metadata.pkl    ← Model metadata
│
├── feature_importance_1h.png     ← Chart: Which features matter
├── predictions_1h.png            ← Chart: Prediction accuracy
└── error_analysis_1h.png         ← Chart: Error distribution
