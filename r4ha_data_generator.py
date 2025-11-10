"""
R4HA Prediction POC - Synthetic Data Generator
Generates realistic mainframe MSU consumption data with all 15 TIER 1 features
# Run the data generator script first
# python r4ha_data_generator.py
# This creates: r4ha_synthetic_data.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class R4HADataGenerator:
    """Generate synthetic mainframe MSU data with realistic patterns"""
    
    def __init__(self, system_id='PROD01', start_date='2024-01-01', months=12, seed=42):
        """
        Initialize the data generator
        
        Parameters:
        -----------
        system_id : str
            System identifier (e.g., 'PROD01')
        start_date : str
            Start date for data generation (YYYY-MM-DD)
        months : int
            Number of months of data to generate
        seed : int
            Random seed for reproducibility
        """
        self.system_id = system_id
        self.start_date = pd.to_datetime(start_date)
        self.months = months
        np.random.seed(seed)
        
        # Mainframe capacity parameters
        self.baseline_msu = 500  # Baseline MSU consumption
        self.max_capacity_msu = 1200  # Maximum system capacity
        
        # Define batch windows
        self.batch_windows = [
            {'name': 'nightly_batch', 'start': 0, 'end': 6, 'days': [0,1,2,3,4,5,6]},  # Daily 00:00-06:00
            {'name': 'eod_processing', 'start': 18, 'end': 22, 'days': [0,1,2,3,4]},  # Weekdays 18:00-22:00
            {'name': 'weekend_extended', 'start': 22, 'end': 10, 'days': [5,6]}  # Weekend 22:00-10:00
        ]
        
    def _is_batch_window(self, timestamp):
        """Determine if timestamp falls within a batch window"""
        hour = timestamp.hour
        day = timestamp.dayofweek
        
        for window in self.batch_windows:
            if day in window['days']:
                if window['start'] < window['end']:
                    if window['start'] <= hour < window['end']:
                        return 1
                else:  # Crosses midnight
                    if hour >= window['start'] or hour < window['end']:
                        return 1
        return 0
    
    def _is_month_end(self, timestamp):
        """Check if timestamp is within last 3 days of month"""
        days_in_month = pd.Timestamp(timestamp.year, timestamp.month, 1) + pd.offsets.MonthEnd(0)
        return timestamp.day >= (days_in_month.day - 2)
    
    def _calculate_base_msu(self, timestamp):
        """Calculate base MSU consumption based on time patterns"""
        hour = timestamp.hour
        day = timestamp.dayofweek
        is_batch = self._is_batch_window(timestamp)
        is_month_end = self._is_month_end(timestamp)
        
        # Hourly pattern (business hours vs off-hours)
        hourly_pattern = {
            0: 0.9, 1: 0.95, 2: 1.0, 3: 1.0, 4: 0.95, 5: 0.9,  # Nightly batch peak
            6: 0.7, 7: 0.6, 8: 0.75, 9: 0.85, 10: 0.9,  # Morning ramp-up
            11: 0.95, 12: 0.9, 13: 0.92, 14: 0.95, 15: 0.93,  # Business hours
            16: 0.9, 17: 0.85, 18: 0.95, 19: 1.0, 20: 0.98, 21: 0.95,  # EOD processing
            22: 0.85, 23: 0.88  # Evening batch
        }
        
        # Weekly pattern (Monday busiest, Sunday quietest)
        weekly_pattern = {
            0: 1.1,   # Monday (weekend catchup)
            1: 1.0,   # Tuesday
            2: 1.0,   # Wednesday
            3: 1.0,   # Thursday
            4: 1.05,  # Friday (reports)
            5: 0.95,  # Saturday (lighter)
            6: 0.7    # Sunday (minimal)
        }
        
        base = self.baseline_msu
        base *= hourly_pattern.get(hour, 1.0)
        base *= weekly_pattern.get(day, 1.0)
        
        # Batch window boost
        if is_batch:
            base *= 1.4
        
        # Month-end boost
        if is_month_end:
            base *= 1.3
        
        return base
    
    def _simulate_batch_jobs(self, timestamp):
        """Simulate number of batch jobs running and their CPU consumption"""
        hour = timestamp.hour
        day = timestamp.dayofweek
        is_batch = self._is_batch_window(timestamp)
        is_month_end = self._is_month_end(timestamp)
        
        if not is_batch:
            # Online-only workload
            jobs = np.random.randint(5, 15)
            cpu_per_job = np.random.uniform(2, 6)
        else:
            # Batch window patterns
            if 0 <= hour < 6:  # Nightly batch
                base_jobs = 40 if is_month_end else 30
                jobs = np.random.randint(base_jobs - 10, base_jobs + 15)
                cpu_per_job = np.random.uniform(8, 15)
            elif 18 <= hour < 22:  # EOD processing
                base_jobs = 35 if is_month_end else 25
                jobs = np.random.randint(base_jobs - 8, base_jobs + 10)
                cpu_per_job = np.random.uniform(7, 12)
            else:
                jobs = np.random.randint(15, 25)
                cpu_per_job = np.random.uniform(5, 10)
        
        batch_jobs_running = jobs
        batch_cpu_seconds = jobs * cpu_per_job * np.random.uniform(0.9, 1.1)
        
        return batch_jobs_running, batch_cpu_seconds
    
    def generate_raw_data(self):
        """Generate raw 15-minute interval data"""
        # Create timestamp range (15-minute intervals)
        end_date = self.start_date + pd.DateOffset(months=self.months)
        timestamps = pd.date_range(start=self.start_date, end=end_date, freq='15min')
        
        data = []
        
        print(f"Generating {len(timestamps)} records ({self.months} months of data)...")
        
        for i, ts in enumerate(timestamps):
            if i % 2000 == 0:
                print(f"Progress: {i}/{len(timestamps)} ({i/len(timestamps)*100:.1f}%)")
            
            # Calculate base MSU
            base_msu = self._calculate_base_msu(ts)
            
            # Add realistic noise and variability
            noise = np.random.normal(0, 30)  # Random fluctuation
            
            # Add autocorrelation (current value influenced by previous)
            if i > 0:
                previous_msu = data[-1]['msu_current']
                autocorr_factor = 0.7  # Strong autocorrelation
                msu_current = autocorr_factor * previous_msu + (1 - autocorr_factor) * base_msu + noise
            else:
                msu_current = base_msu + noise
            
            # Ensure within realistic bounds
            msu_current = np.clip(msu_current, 200, self.max_capacity_msu)
            
            # Calculate CPU utilization (correlated with MSU)
            cpu_utilization_pct = (msu_current / self.max_capacity_msu) * 100
            cpu_utilization_pct += np.random.normal(0, 3)  # Small noise
            cpu_utilization_pct = np.clip(cpu_utilization_pct, 10, 98)
            
            # Simulate batch jobs
            batch_jobs, batch_cpu = self._simulate_batch_jobs(ts)
            
            # Create record
            record = {
                'timestamp': ts,
                'system_id': self.system_id,
                'msu_current': round(msu_current, 2),
                'cpu_utilization_pct': round(cpu_utilization_pct, 2),
                'hour_of_day': ts.hour,
                'day_of_week': ts.dayofweek + 1,  # 1=Monday, 7=Sunday
                'is_batch_window': self._is_batch_window(ts),
                'batch_jobs_running': int(batch_jobs),
                'batch_cpu_seconds': round(batch_cpu, 2)
            }
            
            data.append(record)
        
        print("Raw data generation complete!")
        return pd.DataFrame(data)
    
    def calculate_features(self, df):
        """Calculate all derived features (R4HA, lags, rolling means)"""
        print("\nCalculating derived features...")
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate R4HA (4-hour rolling average of msu_current)
        # 4 hours = 16 periods of 15 minutes
        print("  - Calculating R4HA (4-hour rolling average)...")
        df['r4ha_msu'] = df['msu_current'].rolling(window=16, min_periods=16).mean()
        
        # Calculate lag features
        print("  - Calculating lag features...")
        df['r4ha_lag_1h'] = df['r4ha_msu'].shift(4)   # 1 hour = 4 periods
        df['r4ha_lag_2h'] = df['r4ha_msu'].shift(8)   # 2 hours = 8 periods
        df['msu_lag_1h'] = df['msu_current'].shift(4)
        
        # Calculate rolling means
        print("  - Calculating rolling means...")
        df['msu_rolling_mean_2h'] = df['msu_current'].rolling(window=8, min_periods=8).mean()
        df['msu_rolling_mean_4h'] = df['msu_current'].rolling(window=16, min_periods=16).mean()
        
        # Create target variable (R4HA 4 hours in the future)
        print("  - Creating target variable (R4HA 4 hours ahead)...")
        df['target'] = df['r4ha_msu'].shift(-16)
        
        # Round all calculated columns
        for col in ['r4ha_msu', 'r4ha_lag_1h', 'r4ha_lag_2h', 'msu_lag_1h', 
                    'msu_rolling_mean_2h', 'msu_rolling_mean_4h', 'target']:
            if col in df.columns:
                df[col] = df[col].round(2)
        
        print("Feature calculation complete!")
        return df
    
    def generate_complete_dataset(self, drop_na=True):
        """Generate complete dataset with all features"""
        print(f"\n{'='*60}")
        print(f"R4HA SYNTHETIC DATA GENERATOR")
        print(f"{'='*60}")
        print(f"System: {self.system_id}")
        print(f"Period: {self.start_date.date()} to {(self.start_date + pd.DateOffset(months=self.months)).date()}")
        print(f"Duration: {self.months} months")
        print(f"{'='*60}\n")
        
        # Generate raw data
        df = self.generate_raw_data()
        
        # Calculate derived features
        df = self.calculate_features(df)
        
        # Remove rows with NaN values (due to rolling windows and lags)
        if drop_na:
            rows_before = len(df)
            df = df.dropna().reset_index(drop=True)
            rows_after = len(df)
            print(f"\nRemoved {rows_before - rows_after} rows with NaN values")
            print(f"Final dataset: {rows_after} complete records")
        
        # Reorder columns to match specification
        column_order = [
            'timestamp', 'system_id', 'msu_current', 'r4ha_msu', 'cpu_utilization_pct',
            'r4ha_lag_1h', 'r4ha_lag_2h', 'msu_lag_1h', 
            'msu_rolling_mean_2h', 'msu_rolling_mean_4h',
            'hour_of_day', 'day_of_week', 'is_batch_window',
            'batch_jobs_running', 'batch_cpu_seconds', 'target'
        ]
        
        df = df[column_order]
        
        return df
    
    def print_data_summary(self, df):
        """Print summary statistics of the generated data"""
        print(f"\n{'='*60}")
        print("DATA SUMMARY")
        print(f"{'='*60}\n")
        
        print(f"Total Records: {len(df):,}")
        print(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
        
        print("\n--- MSU Statistics ---")
        print(f"MSU Current - Mean: {df['msu_current'].mean():.2f}, Std: {df['msu_current'].std():.2f}")
        print(f"MSU Current - Min: {df['msu_current'].min():.2f}, Max: {df['msu_current'].max():.2f}")
        print(f"R4HA - Mean: {df['r4ha_msu'].mean():.2f}, Std: {df['r4ha_msu'].std():.2f}")
        print(f"R4HA - Min: {df['r4ha_msu'].min():.2f}, Max: {df['r4ha_msu'].max():.2f}")
        
        print("\n--- CPU Utilization ---")
        print(f"Mean: {df['cpu_utilization_pct'].mean():.2f}%")
        print(f"Range: {df['cpu_utilization_pct'].min():.2f}% to {df['cpu_utilization_pct'].max():.2f}%")
        
        print("\n--- Batch Activity ---")
        print(f"Batch Window Coverage: {df['is_batch_window'].mean()*100:.1f}% of time")
        print(f"Avg Jobs (Batch Windows): {df[df['is_batch_window']==1]['batch_jobs_running'].mean():.1f}")
        print(f"Avg Jobs (Non-Batch): {df[df['is_batch_window']==0]['batch_jobs_running'].mean():.1f}")
        
        print("\n--- Temporal Distribution ---")
        print("Records by Day of Week:")
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for i, day_name in enumerate(day_names, 1):
            count = len(df[df['day_of_week'] == i])
            print(f"  {day_name}: {count:,} records")
        
        print(f"\n{'='*60}\n")


def main():
    """Main execution function"""
    
    # Initialize generator
    generator = R4HADataGenerator(
        system_id='PROD01',
        start_date='2024-01-01',
        months=12,  # 12 months of data
        seed=42
    )
    
    # Generate complete dataset
    df = generator.generate_complete_dataset(drop_na=True)
    
    # Print summary
    generator.print_data_summary(df)
    
    # Save to CSV
    output_file = 'r4ha_synthetic_data.csv'
    df.to_csv(output_file, index=False)
    print(f"✓ Data saved to: {output_file}")
    print(f"✓ File size: {len(df) * 16 * 8 / 1024 / 1024:.2f} MB")
    
    # Display sample records
    print("\n--- Sample Data (First 10 Records) ---")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(df.head(10).to_string(index=False))
    
    print("\n--- Sample Data (Random 5 Records from Batch Window) ---")
    batch_sample = df[df['is_batch_window'] == 1].sample(5)
    print(batch_sample.to_string(index=False))
    
    return df


if __name__ == "__main__":
    df = main()
    print("\n✓ Synthetic data generation complete!")
    print("✓ Ready for model training!")
