"""
R4HA Prediction Model using XGBoost
Complete pipeline: Data loading → Training → Evaluation → Prediction
# Run the training script
#python r4ha_xgboost_model.py

# Output:
# - Trained model: r4ha_model.json
# - Plots: feature_importance.png, predictions_test.png, error_analysis_test.png
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class R4HAPredictor:
    """XGBoost model for R4HA prediction"""
    
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.feature_importance = None
        
    def prepare_data(self, df):
        """
        Prepare data for training
        
        Parameters:
        -----------
        df : DataFrame
            Complete dataset with all features and target
            
        Returns:
        --------
        X : DataFrame - Features
        y : Series - Target variable
        """
        print("Preparing data for training...")
        
        # Define feature columns (all except timestamp, system_id, and target)
        self.feature_columns = [
            'msu_current',
            'r4ha_msu',
            'cpu_utilization_pct',
            'r4ha_lag_1h',
            'r4ha_lag_2h',
            'msu_lag_1h',
            'msu_rolling_mean_2h',
            'msu_rolling_mean_4h',
            'hour_of_day',
            'day_of_week',
            'is_batch_window',
            'batch_jobs_running',
            'batch_cpu_seconds'
        ]
        
        # Check for missing values
        missing_before = df[self.feature_columns + ['target']].isnull().sum().sum()
        print(f"  Missing values before cleaning: {missing_before}")
        
        # Remove rows with missing values
        df_clean = df.dropna(subset=self.feature_columns + ['target']).copy()
        
        print(f"  Records after cleaning: {len(df_clean):,}")
        
        # Separate features and target
        X = df_clean[self.feature_columns]
        y = df_clean['target']
        
        print(f"  Features shape: {X.shape}")
        print(f"  Target shape: {y.shape}")
        print(f"  Target range: {y.min():.2f} - {y.max():.2f}")
        
        return X, y, df_clean
    
    def split_data(self, X, y, df_clean, test_size=0.15, val_size=0.10):
        """
        Split data into train, validation, and test sets (time-based split)
        
        Parameters:
        -----------
        X : DataFrame - Features
        y : Series - Target
        df_clean : DataFrame - Complete cleaned data
        test_size : float - Proportion for test set
        val_size : float - Proportion for validation set
        
        Returns:
        --------
        Dictionary with train, val, test splits
        """
        print("\nSplitting data (time-based)...")
        
        # Time-based split (no shuffling to preserve temporal order)
        n_total = len(X)
        n_test = int(n_total * test_size)
        n_val = int(n_total * val_size)
        n_train = n_total - n_test - n_val
        
        # Split indices
        train_idx = range(0, n_train)
        val_idx = range(n_train, n_train + n_val)
        test_idx = range(n_train + n_val, n_total)
        
        # Create splits
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        X_test = X.iloc[test_idx]
        
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        y_test = y.iloc[test_idx]
        
        # Get timestamps for each split
        timestamps_train = df_clean.iloc[train_idx]['timestamp']
        timestamps_val = df_clean.iloc[val_idx]['timestamp']
        timestamps_test = df_clean.iloc[test_idx]['timestamp']
        
        print(f"  Train set: {len(X_train):,} records ({len(X_train)/n_total*100:.1f}%)")
        print(f"    Period: {timestamps_train.min()} to {timestamps_train.max()}")
        print(f"  Validation set: {len(X_val):,} records ({len(X_val)/n_total*100:.1f}%)")
        print(f"    Period: {timestamps_val.min()} to {timestamps_val.max()}")
        print(f"  Test set: {len(X_test):,} records ({len(X_test)/n_total*100:.1f}%)")
        print(f"    Period: {timestamps_test.min()} to {timestamps_test.max()}")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'timestamps_train': timestamps_train,
            'timestamps_val': timestamps_val,
            'timestamps_test': timestamps_test
        }
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """
        Train XGBoost model with optimal hyperparameters
        
        Parameters:
        -----------
        X_train, y_train : Training data
        X_val, y_val : Validation data for early stopping
        """
        print("\n" + "="*60)
        print("TRAINING XGBOOST MODEL")
        print("="*60)
        
        # XGBoost parameters optimized for time-series regression
        params = {
            'objective': 'reg:squarederror',  # Regression task
            'max_depth': 6,                    # Tree depth (prevents overfitting)
            'learning_rate': 0.1,              # Step size (eta)
            'n_estimators': 500,               # Number of trees
            'min_child_weight': 3,             # Minimum samples in leaf
            'subsample': 0.8,                  # Row sampling (prevents overfitting)
            'colsample_bytree': 0.8,           # Column sampling
            'gamma': 0.1,                      # Minimum loss reduction
            'reg_alpha': 0.1,                  # L1 regularization
            'reg_lambda': 1.0,                 # L2 regularization
            'random_state': 42,
            'n_jobs': -1,                      # Use all CPU cores
            'tree_method': 'hist'              # Fast histogram-based algorithm
        }
        
        print("\nModel parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        # Create and train model with early stopping
        print("\nTraining in progress...")
        self.model = xgb.XGBRegressor(**params)
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_metric='rmse',
            early_stopping_rounds=50,  # Stop if no improvement for 50 rounds
            verbose=False
        )
        
        # Get best iteration
        best_iteration = self.model.best_iteration
        print(f"\n✓ Training complete!")
        print(f"  Best iteration: {best_iteration}")
        print(f"  Total trees used: {best_iteration + 1}")
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return self.model
    
    def evaluate_model(self, splits):
        """
        Evaluate model performance on train, validation, and test sets
        
        Parameters:
        -----------
        splits : dict - Dictionary containing all data splits
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        results = {}
        
        for dataset_name in ['train', 'val', 'test']:
            X = splits[f'X_{dataset_name}']
            y_true = splits[f'y_{dataset_name}']
            
            # Make predictions
            y_pred = self.model.predict(X)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            results[dataset_name] = {
                'y_true': y_true,
                'y_pred': y_pred,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape,
                'timestamps': splits[f'timestamps_{dataset_name}']
            }
            
            print(f"\n{dataset_name.upper()} SET:")
            print(f"  RMSE: {rmse:.2f} MSU")
            print(f"  MAE:  {mae:.2f} MSU")
            print(f"  R²:   {r2:.4f}")
            print(f"  MAPE: {mape:.2f}%")
        
        return results
    
    def plot_feature_importance(self, top_n=13):
        """Plot feature importance"""
        plt.figure(figsize=(10, 6))
        
        top_features = self.feature_importance.head(top_n)
        
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance Score')
        plt.title('Feature Importance (XGBoost)')
        plt.gca().invert_yaxis()
        
        # Add percentage labels
        total_importance = top_features['importance'].sum()
        for i, (idx, row) in enumerate(top_features.iterrows()):
            pct = (row['importance'] / total_importance) * 100
            plt.text(row['importance'], i, f' {pct:.1f}%', va='center')
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("\n✓ Feature importance plot saved: feature_importance.png")
        plt.close()
    
    def plot_predictions(self, results, dataset='test', sample_hours=48):
        """
        Plot actual vs predicted values
        
        Parameters:
        -----------
        results : dict - Evaluation results
        dataset : str - Which dataset to plot ('train', 'val', 'test')
        sample_hours : int - Number of hours to plot (for readability)
        """
        y_true = results[dataset]['y_true'].values
        y_pred = results[dataset]['y_pred']
        timestamps = results[dataset]['timestamps'].values
        
        # Sample data for better visualization
        sample_size = sample_hours * 4  # 4 readings per hour (15-min intervals)
        if len(y_true) > sample_size:
            # Take last sample_size points
            y_true = y_true[-sample_size:]
            y_pred = y_pred[-sample_size:]
            timestamps = timestamps[-sample_size:]
        
        plt.figure(figsize=(14, 6))
        
        plt.plot(timestamps, y_true, label='Actual R4HA', linewidth=2, alpha=0.7)
        plt.plot(timestamps, y_pred, label='Predicted R4HA', linewidth=2, alpha=0.7)
        
        plt.xlabel('Time')
        plt.ylabel('R4HA (MSU)')
        plt.title(f'R4HA Prediction - {dataset.upper()} Set (Last {sample_hours} hours)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'predictions_{dataset}.png', dpi=300, bbox_inches='tight')
        print(f"✓ Prediction plot saved: predictions_{dataset}.png")
        plt.close()
    
    def plot_error_distribution(self, results, dataset='test'):
        """Plot prediction error distribution"""
        y_true = results[dataset]['y_true'].values
        y_pred = results[dataset]['y_pred']
        errors = y_pred - y_true
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Error distribution histogram
        axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        axes[0].set_xlabel('Prediction Error (MSU)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Error Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Scatter plot: Actual vs Predicted
        axes[1].scatter(y_true, y_pred, alpha=0.5, s=10)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[1].plot([min_val, max_val], [min_val, max_val], 
                     'r--', linewidth=2, label='Perfect Prediction')
        
        axes[1].set_xlabel('Actual R4HA (MSU)')
        axes[1].set_ylabel('Predicted R4HA (MSU)')
        axes[1].set_title('Actual vs Predicted')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'error_analysis_{dataset}.png', dpi=300, bbox_inches='tight')
        print(f"✓ Error analysis plot saved: error_analysis_{dataset}.png")
        plt.close()
    
    def predict_real_time(self, current_features):
        """
        Make real-time prediction
        
        Parameters:
        -----------
        current_features : dict or DataFrame
            Current system features
            
        Returns:
        --------
        predicted_r4ha : float
            Predicted R4HA 4 hours from now
        """
        if isinstance(current_features, dict):
            current_features = pd.DataFrame([current_features])
        
        # Ensure columns are in correct order
        current_features = current_features[self.feature_columns]
        
        prediction = self.model.predict(current_features)[0]
        
        return prediction
    
    def save_model(self, filename='r4ha_model.json'):
        """Save trained model to file"""
        self.model.save_model(filename)
        print(f"\n✓ Model saved to: {filename}")
    
    def load_model(self, filename='r4ha_model.json'):
        """Load trained model from file"""
        self.model = xgb.XGBRegressor()
        self.model.load_model(filename)
        print(f"✓ Model loaded from: {filename}")


def main():
    """Main execution function"""
    
    print("="*60)
    print("R4HA PREDICTION MODEL - XGBOOST")
    print("="*60)
    
    # Step 1: Load synthetic data
    print("\n[1/6] Loading data...")
    try:
        df = pd.read_csv('r4ha_synthetic_data.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"✓ Loaded {len(df):,} records")
    except FileNotFoundError:
        print("ERROR: r4ha_synthetic_data.csv not found!")
        print("Please run the data generator script first.")
        return
    
    # Step 2: Initialize predictor
    print("\n[2/6] Initializing predictor...")
    predictor = R4HAPredictor()
    
    # Step 3: Prepare data
    print("\n[3/6] Preparing data...")
    X, y, df_clean = predictor.prepare_data(df)
    
    # Step 4: Split data
    print("\n[4/6] Splitting data...")
    splits = predictor.split_data(X, y, df_clean)
    
    # Step 5: Train model
    print("\n[5/6] Training model...")
    predictor.train_model(
        splits['X_train'], splits['y_train'],
        splits['X_val'], splits['y_val']
    )
    
    # Step 6: Evaluate model
    print("\n[6/6] Evaluating model...")
    results = predictor.evaluate_model(splits)
    
    # Display feature importance
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE (Top 10)")
    print("="*60)
    print(predictor.feature_importance.head(10).to_string(index=False))
    
    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    predictor.plot_feature_importance()
    predictor.plot_predictions(results, dataset='test', sample_hours=72)
    predictor.plot_error_distribution(results, dataset='test')
    
    # Save model
    predictor.save_model('r4ha_model.json')
    
    # Example real-time prediction
    print("\n" + "="*60)
    print("EXAMPLE REAL-TIME PREDICTION")
    print("="*60)
    
    # Use last record from test set as example
    last_record = splits['X_test'].iloc[-1].to_dict()
    
    print("\nCurrent System State:")
    print(f"  MSU Current: {last_record['msu_current']:.2f}")
    print(f"  R4HA Current: {last_record['r4ha_msu']:.2f}")
    print(f"  R4HA 1h ago: {last_record['r4ha_lag_1h']:.2f}")
    print(f"  CPU Utilization: {last_record['cpu_utilization_pct']:.2f}%")
    print(f"  Hour: {last_record['hour_of_day']}")
    print(f"  Batch Window: {'Yes' if last_record['is_batch_window'] else 'No'}")
    print(f"  Jobs Running: {last_record['batch_jobs_running']}")
    
    predicted_r4ha = predictor.predict_real_time(last_record)
    actual_r4ha = splits['y_test'].iloc[-1]
    
    print(f"\nPrediction:")
    print(f"  Predicted R4HA (4h ahead): {predicted_r4ha:.2f} MSU")
    print(f"  Actual R4HA (4h ahead): {actual_r4ha:.2f} MSU")
    print(f"  Prediction Error: {abs(predicted_r4ha - actual_r4ha):.2f} MSU")
    print(f"  Error Percentage: {abs(predicted_r4ha - actual_r4ha) / actual_r4ha * 100:.2f}%")
    
    # Model summary
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    print(f"✓ Model Type: XGBoost Regressor")
    print(f"✓ Features Used: {len(predictor.feature_columns)}")
    print(f"✓ Training Records: {len(splits['X_train']):,}")
    print(f"✓ Test RMSE: {results['test']['rmse']:.2f} MSU")
    print(f"✓ Test MAE: {results['test']['mae']:.2f} MSU")
    print(f"✓ Test R²: {results['test']['r2']:.4f}")
    print(f"✓ Test MAPE: {results['test']['mape']:.2f}%")
    
    accuracy_pct = (1 - results['test']['mape']/100) * 100
    print(f"\n✓ Model Accuracy: ~{accuracy_pct:.1f}%")
    print(f"✓ Prediction Horizon: 4 hours")
    print(f"✓ Model saved: r4ha_model.json")
    
    return predictor, results


if __name__ == "__main__":
    predictor, results = main()
    print("\n✓ Model training complete!")
    print("✓ Ready for production deployment!")
