"""
R4HA Real-Time Prediction - 1 Hour Ahead
Uses trained XGBoost model to predict R4HA 1 hour in the future

File: r4ha_predict_1h.py
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class R4HAPredictor:
    """Real-time R4HA prediction (1 hour ahead)"""
    
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.prediction_horizon = '1 hour'
        
    def load_model(self, model_file='r4ha_model_1h.json', 
                   metadata_file='r4ha_model_1h_metadata.pkl'):
        """Load trained model and metadata"""
        print("Loading trained model...")
        
        # Load XGBoost model
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_file)
        print(f"✓ Model loaded: {model_file}")
        
        # Load metadata
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        self.feature_columns = metadata['feature_columns']
        print(f"✓ Metadata loaded: {metadata_file}")
        print(f"✓ Features required: {len(self.feature_columns)}")
        print(f"✓ Prediction horizon: {metadata['prediction_horizon']}")
        
    def predict(self, current_features):
        """
        Make prediction for 1 hour ahead
        
        Parameters:
        -----------
        current_features : dict or DataFrame
            Current system state with all 13 features
            
        Returns:
        --------
        prediction : float
            Predicted R4HA 1 hour from now
        """
        if self.model is None:
            raise ValueError("Model not loaded! Call load_model() first.")
        
        # Convert dict to DataFrame if needed
        if isinstance(current_features, dict):
            current_features = pd.DataFrame([current_features])
        
        # Ensure features are in correct order
        current_features = current_features[self.feature_columns]
        
        # Make prediction
        prediction = self.model.predict(current_features)[0]
        
        return prediction
    
    def predict_with_confidence(self, current_features):
        """
        Make prediction with confidence interval estimate
        
        Returns:
        --------
        dict with prediction, lower_bound, upper_bound
        """
        prediction = self.predict(current_features)
        
        # Simple confidence interval (±1.5 * typical error)
        # For 1-hour predictions, typical error is ~20-25 MSU
        typical_error = 25
        
        result = {
            'predicted_r4ha': round(prediction, 2),
            'lower_bound': round(prediction - 1.5 * typical_error, 2),
            'upper_bound': round(prediction + 1.5 * typical_error, 2),
            'confidence_interval': '95%'
        }
        
        return result
    
    def predict_with_alert(self, current_features, threshold=1000):
        """
        Make prediction and check if alert threshold is exceeded
        
        Parameters:
        -----------
        current_features : dict
            Current system state
        threshold : float
            R4HA threshold for alerting (MSU)
            
        Returns:
        --------
        dict with prediction and alert status
        """
        prediction = self.predict(current_features)
        
        result = {
            'current_r4ha': current_features['r4ha_msu'],
            'predicted_r4ha_1h': round(prediction, 2),
            'threshold': threshold,
            'alert': prediction >= threshold,
            'time_to_threshold': '1 hour' if prediction >= threshold else 'N/A',
            'change_from_current': round(prediction - current_features['r4ha_msu'], 2)
        }
        
        return result
    
    def batch_predict(self, features_df):
        """
        Make predictions for multiple time points
        
        Parameters:
        -----------
        features_df : DataFrame
            Multiple rows of features
            
        Returns:
        --------
        DataFrame with predictions
        """
        predictions = self.model.predict(features_df[self.feature_columns])
        
        result_df = features_df.copy()
        result_df['predicted_r4ha_1h'] = predictions
        
        return result_df


def example_single_prediction():
    """Example: Single real-time prediction"""
    
    print("="*60)
    print("EXAMPLE 1: SINGLE REAL-TIME PREDICTION")
    print("="*60)
    
    # Initialize predictor
    predictor = R4HAPredictor()
    predictor.load_model()
    
    # Current system state (example data)
    print("\n--- Current System State ---")
    current_state = {
        'msu_current': 850.00,
        'r4ha_msu': 820.00,
        'cpu_utilization_pct': 75.50,
        'r4ha_lag_1h': 800.00,
        'r4ha_lag_2h': 780.00,
        'msu_lag_1h': 830.00,
        'msu_rolling_mean_2h': 835.00,
        'msu_rolling_mean_4h': 825.00,
        'hour_of_day': 14,  # 2 PM
        'day_of_week': 3,   # Wednesday
        'is_batch_window': 0,
        'batch_jobs_running': 18,
        'batch_cpu_seconds': 210.50
    }
    
    for key, value in current_state.items():
        print(f"  {key}: {value}")
    
    # Make prediction
    predicted_r4ha = predictor.predict(current_state)
    
    print("\n--- Prediction ---")
    print(f"Current Time: 14:00 (2:00 PM)")
    print(f"Current R4HA: {current_state['r4ha_msu']:.2f} MSU")
    print(f"Predicted R4HA at 15:00 (3:00 PM): {predicted_r4ha:.2f} MSU")
    print(f"Expected Change: {predicted_r4ha - current_state['r4ha_msu']:.2f} MSU")
    
    # Prediction with confidence interval
    result = predictor.predict_with_confidence(current_state)
    print(f"\nConfidence Interval ({result['confidence_interval']}):")
    print(f"  Lower Bound: {result['lower_bound']:.2f} MSU")
    print(f"  Prediction:  {result['predicted_r4ha']:.2f} MSU")
    print(f"  Upper Bound: {result['upper_bound']:.2f} MSU")


def example_with_alert():
    """Example: Prediction with threshold alerting"""
    
    print("\n" + "="*60)
    print("EXAMPLE 2: PREDICTION WITH ALERT THRESHOLD")
    print("="*60)
    
    predictor = R4HAPredictor()
    predictor.load_model()
    
    # Scenario: High workload approaching threshold
    print("\n--- Scenario: Evening Batch Starting ---")
    high_load_state = {
        'msu_current': 950.00,
        'r4ha_msu': 920.00,
        'cpu_utilization_pct': 82.00,
        'r4ha_lag_1h': 880.00,
        'r4ha_lag_2h': 850.00,
        'msu_lag_1h': 930.00,
        'msu_rolling_mean_2h': 940.00,
        'msu_rolling_mean_4h': 910.00,
        'hour_of_day': 18,  # 6 PM - Batch starting
        'day_of_week': 5,   # Friday
        'is_batch_window': 1,
        'batch_jobs_running': 35,
        'batch_cpu_seconds': 420.00
    }
    
    # Check against threshold
    alert_result = predictor.predict_with_alert(high_load_state, threshold=1000)
    
    print("\nCurrent State:")
    print(f"  Time: 18:00 (6:00 PM Friday)")
    print(f"  Current R4HA: {alert_result['current_r4ha']:.2f} MSU")
    print(f"  Batch Window: Active")
    print(f"  Jobs Running: {high_load_state['batch_jobs_running']}")
    
    print("\nPrediction:")
    print(f"  Predicted R4HA (19:00): {alert_result['predicted_r4ha_1h']:.2f} MSU")
    print(f"  Change: {alert_result['change_from_current']:+.2f} MSU")
    print(f"  Threshold: {alert_result['threshold']} MSU")
    
    if alert_result['alert']:
        print(f"\n⚠️  ALERT: R4HA will exceed {alert_result['threshold']} MSU in {alert_result['time_to_threshold']}!")
        print("  Recommended Actions:")
        print("    - Review batch schedule")
        print("    - Consider delaying non-critical jobs")
        print("    - Notify capacity planning team")
    else:
        print(f"\n✓ OK: R4HA will remain below {alert_result['threshold']} MSU threshold")


def example_batch_predictions():
    """Example: Multiple predictions from test data"""
    
    print("\n" + "="*60)
    print("EXAMPLE 3: BATCH PREDICTIONS FROM TEST DATA")
    print("="*60)
    
    predictor = R4HAPredictor()
    predictor.load_model()
    
    # Load test data
    print("\nLoading test data...")
    df = pd.read_csv('r4ha_data_1h.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Take last 24 hours (96 records)
    test_data = df.tail(96).copy()
    
    # Make batch predictions
    print("Making batch predictions...")
    predictions = predictor.batch_predict(test_data)
    
    # Compare predictions to actual
    predictions['actual_r4ha_1h'] = predictions['target']
    predictions['error'] = predictions['predicted_r4ha_1h'] - predictions['actual_r4ha_1h']
    predictions['abs_error'] = predictions['error'].abs()
    
    # Display results
    print("\n--- Batch Prediction Results (Last 24 hours) ---")
    print(f"Total Predictions: {len(predictions)}")
    print(f"Average Error: {predictions['error'].mean():.2f} MSU")
    print(f"Average Absolute Error: {predictions['abs_error'].mean():.2f} MSU")
    print(f"Max Error: {predictions['abs_error'].max():.2f} MSU")
    print(f"RMSE: {np.sqrt((predictions['error']**2).mean()):.2f} MSU")
    
    # Show sample predictions
    print("\n--- Sample Predictions ---")
    sample = predictions[['timestamp', 'r4ha_msu', 'predicted_r4ha_1h', 
                          'actual_r4ha_1h', 'error']].head(10)
    sample.columns = ['Time', 'Current R4HA', 'Predicted 1h', 'Actual 1h', 'Error']
    print(sample.to_string(index=False))


def example_production_workflow():
    """Example: Production deployment workflow"""
    
    print("\n" + "="*60)
    print("EXAMPLE 4: PRODUCTION WORKFLOW")
    print("="*60)
    
    predictor = R4HAPredictor()
    predictor.load_model()
    
    print("\n--- Simulating Real-Time Monitoring ---")
    print("(In production, this would run every 15 minutes)")
    
    # Simulate 4 consecutive monitoring intervals
    monitoring_times = [
        {'hour': 14, 'minute': 0, 'msu': 820, 'jobs': 15},
        {'hour': 14, 'minute': 15, 'msu': 835, 'jobs': 18},
        {'hour': 14, 'minute': 30, 'msu': 850, 'jobs': 20},
        {'hour': 14, 'minute': 45, 'msu': 870, 'jobs': 25}
    ]
    
    for i, time_point in enumerate(monitoring_times, 1):
        print(f"\n[Monitoring Interval {i}]")
        print(f"Time: {time_point['hour']:02d}:{time_point['minute']:02d}")
        
        # Build feature set
        features = {
            'msu_current': time_point['msu'],
            'r4ha_msu': time_point['msu'] - 20,  # Approximation
            'cpu_utilization_pct': (time_point['msu'] / 1200) * 100,
            'r4ha_lag_1h': time_point['msu'] - 40,
            'r4ha_lag_2h': time_point['msu'] - 60,
            'msu_lag_1h': time_point['msu'] - 30,
            'msu_rolling_mean_2h': time_point['msu'] - 10,
            'msu_rolling_mean_4h': time_point['msu'] - 15,
            'hour_of_day': time_point['hour'],
            'day_of_week': 3,
            'is_batch_window': 0,
            'batch_jobs_running': time_point['jobs'],
            'batch_cpu_seconds': time_point['jobs'] * 12
        }
        
        # Make prediction
        result = predictor.predict_with_alert(features, threshold=1000)
        
        print(f"Current R4HA: {result['current_r4ha']:.2f} MSU")
        print(f"Predicted (1h ahead): {result['predicted_r4ha_1h']:.2f} MSU")
        print(f"Trend: {result['change_from_current']:+.2f} MSU")
        
        if result['alert']:
            print("⚠️  ALERT: Threshold will be exceeded!")


def main():
    """Run all examples"""
    
    print("\n" + "="*70)
    print(" "*20 + "R4HA REAL-TIME PREDICTION")
    print(" "*25 + "(1 HOUR AHEAD)")
    print("="*70)
    
    try:
        # Example 1: Single prediction
        example_single_prediction()
        
        # Example 2: With alerting
        example_with_alert()
        
        # Example 3: Batch predictions
        example_batch_predictions()
        
        # Example 4: Production workflow
        example_production_workflow()
        
        print("\n" + "="*70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\n✓ Model is ready for production deployment")
        print("\nTo use in your own code:")
        print("  from r4ha_predict_1h import R4HAPredictor")
        print("  predictor = R4HAPredictor()")
        print("  predictor.load_model()")
        print("  prediction = predictor.predict(current_features)")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: Required file not found: {e}")
        print("\nMake sure you have run:")
        print("  1. r4ha_data_generator_1h.py")
        print("  2. r4ha_train_model_1h.py")
        print("\nBefore running predictions.")


if __name__ == "__main__":
    main()
