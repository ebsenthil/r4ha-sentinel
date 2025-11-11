"""
R4HA Model Training - 1 Hour Prediction Horizon
Trains XGBoost model to predict R4HA 1 hour ahead

File: r4ha_train_model_1h.py
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')

class R4HAModelTrainer:
    """Train XGBoost model for 1-hour R4HA prediction"""
    
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.feature_importance = None
        self.training_history = {}
        
    def load_data(self, filename='r4ha_data_1h.csv'):
        """Load training data"""
        print(f"Loading data from {filename}...")
        df = pd.read_csv(filename)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"✓ Loaded {len(df):,} records")
        return df
    
    def prepare_features(self, df):
        """Prepare features and target"""
        print("\nPreparing features...")
        
        # Define feature columns (13 features)
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
        missing = df[self.feature_columns + ['target']].isnull().sum().sum()
        print(f"  Missing values: {missing}")
        
        # Remove rows with missing values
        df_clean = df.dropna(subset=self.feature_columns + ['target']).copy()
        print(f"  Clean records: {len(df_clean):,}")
        
        # Separate features and target
        X = df_clean[self.feature_columns]
        y = df_clean['target']
        
        print(f"  Features shape: {X.shape}")
        print(f"  Target range: {y.min():.2f} - {y.max():.2f}")
        
        return X, y, df_clean
    
    def split_data(self, X, y, df_clean):
        """Time-based train/val/test split"""
        print("\nSplitting data (time-based)...")
        
        n_total = len(X)
        n_test = int(n_total * 0.15)
        n_val = int(n_total * 0.10)
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
        
        timestamps_train = df_clean.iloc[train_idx]['timestamp']
        timestamps_val = df_clean.iloc[val_idx]['timestamp']
        timestamps_test = df_clean.iloc[test_idx]['timestamp']
        
        print(f"  Train: {len(X_train):,} ({len(X_train)/n_total*100:.1f}%)")
        print(f"    Period: {timestamps_train.min()} to {timestamps_train.max()}")
        print(f"  Val: {len(X_val):,} ({len(X_val)/n_total*100:.1f}%)")
        print(f"    Period: {timestamps_val.min()} to {timestamps_val.max()}")
        print(f"  Test: {len(X_test):,} ({len(X_test)/n_total*100:.1f}%)")
        print(f"    Period: {timestamps_test.min()} to {timestamps_test.max()}")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'timestamps_train': timestamps_train,
            'timestamps_val': timestamps_val,
            'timestamps_test': timestamps_test
        }
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model"""
        print("\n" + "="*60)
        print("TRAINING XGBOOST MODEL (1-HOUR PREDICTION)")
        print("="*60)
        
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 500,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist'
        }
        
        print("\nModel parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        print("\nTraining in progress...")
        
        # Add early stopping and eval metric to params
        params['early_stopping_rounds'] = 50
        params['eval_metric'] = 'rmse'
        
        self.model = xgb.XGBRegressor(**params)
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
        
        best_iteration = self.model.best_iteration
        print(f"\n✓ Training complete!")
        print(f"  Best iteration: {best_iteration}")
        print(f"  Total trees: {best_iteration + 1}")
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return self.model
    
    def evaluate(self, splits):
        """Evaluate model on all datasets"""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        results = {}
        
        for dataset_name in ['train', 'val', 'test']:
            X = splits[f'X_{dataset_name}']
            y_true = splits[f'y_{dataset_name}']
            
            y_pred = self.model.predict(X)
            
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
    
    def plot_feature_importance(self):
        """Plot feature importance"""
        plt.figure(figsize=(10, 6))
        
        top_features = self.feature_importance.head(13)
        
        plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance Score')
        plt.title('Feature Importance - 1 Hour R4HA Prediction')
        plt.gca().invert_yaxis()
        
        total_importance = top_features['importance'].sum()
        for i, (idx, row) in enumerate(top_features.iterrows()):
            pct = (row['importance'] / total_importance) * 100
            plt.text(row['importance'], i, f' {pct:.1f}%', va='center')
        
        plt.tight_layout()
        plt.savefig('feature_importance_1h.png', dpi=300, bbox_inches='tight')
        print("\n✓ Feature importance plot saved: feature_importance_1h.png")
        plt.close()
    
    def plot_predictions(self, results, hours=48):
        """Plot predictions vs actual"""
        y_true = results['test']['y_true'].values
        y_pred = results['test']['y_pred']
        timestamps = results['test']['timestamps'].values
        
        sample_size = hours * 4
        if len(y_true) > sample_size:
            y_true = y_true[-sample_size:]
            y_pred = y_pred[-sample_size:]
            timestamps = timestamps[-sample_size:]
        
        plt.figure(figsize=(14, 6))
        
        plt.plot(timestamps, y_true, label='Actual R4HA', linewidth=2, alpha=0.7, color='blue')
        plt.plot(timestamps, y_pred, label='Predicted R4HA (1h ahead)', linewidth=2, alpha=0.7, color='red')
        
        plt.xlabel('Time')
        plt.ylabel('R4HA (MSU)')
        plt.title(f'R4HA Prediction - Test Set (Last {hours} hours)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('predictions_1h.png', dpi=300, bbox_inches='tight')
        print("✓ Prediction plot saved: predictions_1h.png")
        plt.close()
    
    def plot_error_analysis(self, results):
        """Plot error distribution"""
        y_true = results['test']['y_true'].values
        y_pred = results['test']['y_pred']
        errors = y_pred - y_true
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Error histogram
        axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        axes[0].set_xlabel('Prediction Error (MSU)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Error Distribution (1-hour predictions)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Scatter plot
        axes[1].scatter(y_true, y_pred, alpha=0.5, s=10, color='steelblue')
        
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
        plt.savefig('error_analysis_1h.png', dpi=300, bbox_inches='tight')
        print("✓ Error analysis plot saved: error_analysis_1h.png")
        plt.close()
    
    def save_model(self):
        """Save trained model"""
        # Save XGBoost model (JSON format)
        self.model.save_model('r4ha_model_1h.json')
        print("\n✓ Model saved: r4ha_model_1h.json")
        
        # Save feature columns (for prediction script)
        model_metadata = {
            'feature_columns': self.feature_columns,
            'feature_importance': self.feature_importance,
            'prediction_horizon': '1 hour'
        }
        with open('r4ha_model_1h_metadata.pkl', 'wb') as f:
            pickle.dump(model_metadata, f)
        print("✓ Metadata saved: r4ha_model_1h_metadata.pkl")


def main():
    """Main training pipeline"""
    
    print("="*60)
    print("R4HA MODEL TRAINING - 1 HOUR PREDICTION")
    print("="*60)
    
    # Initialize trainer
    trainer = R4HAModelTrainer()
    
    # Load data
    print("\n[1/6] Loading data...")
    df = trainer.load_data('r4ha_data_1h.csv')
    
    # Prepare features
    print("\n[2/6] Preparing features...")
    X, y, df_clean = trainer.prepare_features(df)
    
    # Split data
    print("\n[3/6] Splitting data...")
    splits = trainer.split_data(X, y, df_clean)
    
    # Train model
    print("\n[4/6] Training model...")
    trainer.train(splits['X_train'], splits['y_train'],
                  splits['X_val'], splits['y_val'])
    
    # Evaluate model
    print("\n[5/6] Evaluating model...")
    results = trainer.evaluate(splits)
    
    # Generate visualizations
    print("\n[6/6] Generating visualizations...")
    trainer.plot_feature_importance()
    trainer.plot_predictions(results, hours=72)
    trainer.plot_error_analysis(results)
    
    # Display feature importance
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE (Top 10)")
    print("="*60)
    print(trainer.feature_importance.head(10).to_string(index=False))
    
    # Save model
    trainer.save_model()
    
    # Model summary
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    print(f"✓ Model Type: XGBoost Regressor")
    print(f"✓ Prediction Horizon: 1 HOUR")
    print(f"✓ Features Used: {len(trainer.feature_columns)}")
    print(f"✓ Training Records: {len(splits['X_train']):,}")
    print(f"✓ Test RMSE: {results['test']['rmse']:.2f} MSU")
    print(f"✓ Test MAE: {results['test']['mae']:.2f} MSU")
    print(f"✓ Test R²: {results['test']['r2']:.4f}")
    print(f"✓ Test MAPE: {results['test']['mape']:.2f}%")
    
    accuracy_pct = (1 - results['test']['mape']/100) * 100
    print(f"\n✓ Model Accuracy: ~{accuracy_pct:.1f}%")
    print(f"✓ Average Error: ±{results['test']['mae']:.1f} MSU")
    
    print("\n" + "="*60)
    print("FILES CREATED:")
    print("="*60)
    print("  1. r4ha_model_1h.json - Trained XGBoost model")
    print("  2. r4ha_model_1h_metadata.pkl - Model metadata")
    print("  3. feature_importance_1h.png - Feature importance chart")
    print("  4. predictions_1h.png - Prediction accuracy visualization")
    print("  5. error_analysis_1h.png - Error distribution analysis")
    
    print("\n✓ Model training complete!")
    print("\n→ Next step: Run r4ha_predict_1h.py for real-time predictions")
    
    return trainer, results


if __name__ == "__main__":
    trainer, results = main()