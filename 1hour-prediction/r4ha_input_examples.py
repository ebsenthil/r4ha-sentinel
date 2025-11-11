"""
R4HA Prediction - Input Examples & Usage Guide

This file contains:
1. Example JSON input files
2. Usage examples
3. Helper script to create input files

File: r4ha_input_examples.py
"""

import json

# ============================================================
# EXAMPLE 1: Normal workload (JSON format)
# ============================================================
normal_workload = {
    "features": {
        "msu_current": 750.00,
        "r4ha_msu": 730.00,
        "cpu_utilization_pct": 68.50,
        "r4ha_lag_1h": 720.00,
        "r4ha_lag_2h": 710.00,
        "msu_lag_1h": 740.00,
        "msu_rolling_mean_2h": 745.00,
        "msu_rolling_mean_4h": 735.00,
        "hour_of_day": 10,
        "day_of_week": 3,
        "is_batch_window": 0,
        "batch_jobs_running": 12,
        "batch_cpu_seconds": 145.00
    },
    "threshold": 1000,
    "description": "Normal business hours workload"
}

# ============================================================
# EXAMPLE 2: High workload approaching threshold
# ============================================================
high_workload = {
    "features": {
        "msu_current": 980.00,
        "r4ha_msu": 950.00,
        "cpu_utilization_pct": 85.00,
        "r4ha_lag_1h": 920.00,
        "r4ha_lag_2h": 890.00,
        "msu_lag_1h": 960.00,
        "msu_rolling_mean_2h": 970.00,
        "msu_rolling_mean_4h": 940.00,
        "hour_of_day": 18,
        "day_of_week": 5,
        "is_batch_window": 1,
        "batch_jobs_running": 35,
        "batch_cpu_seconds": 420.00
    },
    "threshold": 1000,
    "description": "High workload during evening batch"
}

# ============================================================
# EXAMPLE 3: Critical - already exceeding threshold
# ============================================================
critical_workload = {
    "features": {
        "msu_current": 1050.00,
        "r4ha_msu": 1020.00,
        "cpu_utilization_pct": 92.00,
        "r4ha_lag_1h": 980.00,
        "r4ha_lag_2h": 950.00,
        "msu_lag_1h": 1030.00,
        "msu_rolling_mean_2h": 1040.00,
        "msu_rolling_mean_4h": 1000.00,
        "hour_of_day": 2,
        "day_of_week": 1,
        "is_batch_window": 1,
        "batch_jobs_running": 55,
        "batch_cpu_seconds": 650.00
    },
    "threshold": 1000,
    "description": "Critical - Month-end batch spike"
}

# ============================================================
# EXAMPLE 4: Low workload (weekend)
# ============================================================
low_workload = {
    "features": {
        "msu_current": 420.00,
        "r4ha_msu": 410.00,
        "cpu_utilization_pct": 38.00,
        "r4ha_lag_1h": 400.00,
        "r4ha_lag_2h": 395.00,
        "msu_lag_1h": 415.00,
        "msu_rolling_mean_2h": 418.00,
        "msu_rolling_mean_4h": 408.00,
        "hour_of_day": 15,
        "day_of_week": 7,
        "is_batch_window": 0,
        "batch_jobs_running": 5,
        "batch_cpu_seconds": 35.00
    },
    "threshold": 1000,
    "description": "Low workload - Sunday afternoon"
}


def create_input_files():
    """Create example JSON input files"""
    
    examples = {
        'input_normal.json': normal_workload,
        'input_high.json': high_workload,
        'input_critical.json': critical_workload,
        'input_low.json': low_workload
    }
    
    print("Creating example input files...\n")
    
    for filename, data in examples.items():
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Created: {filename}")
        print(f"  Description: {data['description']}")
        print(f"  Current R4HA: {data['features']['r4ha_msu']} MSU")
        print(f"  Threshold: {data['threshold']} MSU\n")
    
    print("All example files created!")


def print_usage_examples():
    """Print usage examples"""
    
    print("\n" + "="*70)
    print(" "*20 + "USAGE EXAMPLES")
    print("="*70)
    
    print("\n1. INTERACTIVE MODE (Easiest - prompts you for values)")
    print("-" * 70)
    print("python r4ha_predict_with_input.py --interactive")
    print("python r4ha_predict_with_input.py -i")
    
    print("\n\n2. FROM JSON FILE (Recommended)")
    print("-" * 70)
    print("# Use normal workload example")
    print("python r4ha_predict_with_input.py --json input_normal.json")
    print()
    print("# Use high workload with custom threshold")
    print("python r4ha_predict_with_input.py --json input_high.json --threshold 950")
    print()
    print("# Save result to output file")
    print("python r4ha_predict_with_input.py --json input_critical.json --output result.json")
    
    print("\n\n3. FROM CSV FILE")
    print("-" * 70)
    print("# Use latest row from CSV")
    print("python r4ha_predict_with_input.py --csv r4ha_data_1h.csv")
    print()
    print("# Use specific row (e.g., row 100)")
    print("python r4ha_predict_with_input.py --csv r4ha_data_1h.csv --csv_row 100")
    print()
    print("# With custom threshold")
    print("python r4ha_predict_with_input.py --csv r4ha_data_1h.csv --threshold 900")
    
    print("\n\n4. COMMAND LINE ARGUMENTS (All values)")
    print("-" * 70)
    print("python r4ha_predict_with_input.py \\")
    print("  --msu_current 850 \\")
    print("  --r4ha_msu 820 \\")
    print("  --cpu_util 75.5 \\")
    print("  --r4ha_lag_1h 800 \\")
    print("  --r4ha_lag_2h 780 \\")
    print("  --msu_lag_1h 830 \\")
    print("  --mean_2h 835 \\")
    print("  --mean_4h 825 \\")
    print("  --hour 14 \\")
    print("  --day 3 \\")
    print("  --batch 0 \\")
    print("  --jobs 18 \\")
    print("  --cpu_sec 210.5 \\")
    print("  --threshold 1000")
    
    print("\n\n5. PYTHON SCRIPT INTEGRATION")
    print("-" * 70)
    print("""
# Import the predictor
from r4ha_predict_with_input import R4HAPredictor

# Initialize
predictor = R4HAPredictor()
predictor.load_model()

# Your current system metrics
current_state = {
    'msu_current': 850.00,
    'r4ha_msu': 820.00,
    'cpu_utilization_pct': 75.50,
    'r4ha_lag_1h': 800.00,
    'r4ha_lag_2h': 780.00,
    'msu_lag_1h': 830.00,
    'msu_rolling_mean_2h': 835.00,
    'msu_rolling_mean_4h': 825.00,
    'hour_of_day': 14,
    'day_of_week': 3,
    'is_batch_window': 0,
    'batch_jobs_running': 18,
    'batch_cpu_seconds': 210.50
}

# Make prediction with alert
result = predictor.predict_with_alert(current_state, threshold=1000)

# Check result
if result['current_alert']:
    print(f"🔴 ALERT: Current R4HA exceeds threshold!")
    send_email_alert(result)

if result['future_alert']:
    print(f"🟠 WARNING: Will exceed threshold in 1 hour!")
    
print(f"Predicted R4HA: {result['predicted_r4ha_1h']:.2f} MSU")
""")
    
    print("\n\n6. EXIT CODES (for automation)")
    print("-" * 70)
    print("0 = No alert (both current and predicted below threshold)")
    print("1 = Future alert (predicted will exceed threshold)")
    print("2 = Current alert (current exceeds threshold)")
    print()
    print("Example bash script:")
    print("""
#!/bin/bash
python r4ha_predict_with_input.py --json input.json --threshold 1000
EXIT_CODE=$?

if [ $EXIT_CODE -eq 2 ]; then
    echo "CRITICAL: Current R4HA exceeds threshold!"
    # Send critical alert
elif [ $EXIT_CODE -eq 1 ]; then
    echo "WARNING: Will exceed threshold in 1 hour"
    # Send warning
else
    echo "OK: System normal"
fi
""")
    
    print("\n" + "="*70)


def main():
    """Main function"""
    
    print("\n" + "="*70)
    print(" "*15 + "R4HA PREDICTION - INPUT EXAMPLES")
    print("="*70)
    
    # Create example files
    create_input_files()
    
    # Print usage examples
    print_usage_examples()
    
    print("\n✓ Setup complete!")
    print("\nQuick start:")
    print("  1. python r4ha_predict_with_input.py --interactive")
    print("  2. python r4ha_predict_with_input.py --json input_normal.json")


if __name__ == "__main__":
    main()