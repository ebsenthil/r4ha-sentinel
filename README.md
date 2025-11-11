# R4HA Prediction POC - Data Specification (MVP)

## Overview
This document contains the **15 essential columns** needed to build a working R4HA prediction model with 70-75% accuracy. This is the minimum viable dataset for the Proof of Concept.

**Expected Accuracy:** 70-75%  
**Implementation Time:** 2-3 weeks  
**Data Sources:** SMF 70, SMF 30, Batch Schedule  
**Historical Data Needed:** 6-12 months  

---

## Summary Table: All 15 Columns

| # | Column | Type | Source | Importance | Purpose |
|---|--------|------|--------|------------|---------|
| 1 | timestamp | TIMESTAMP | SMF 70 | Critical | Time alignment |
| 2 | system_id | VARCHAR(8) | SMF 70 | Critical | System identification |
| 3 | msu_current | DECIMAL(10,2) | SMF 70 | Critical | Raw MSU data |
| 4 | r4ha_msu | DECIMAL(10,2) | Calculated | Critical | Current R4HA (input) |
| 5 | cpu_utilization_pct | DECIMAL(5,2) | SMF 70 | High | Capacity indicator |
| 6 | r4ha_lag_1h | DECIMAL(10,2) | Calculated | Very High | Recent trend (25-30%) |
| 7 | r4ha_lag_2h | DECIMAL(10,2) | Calculated | High | Trend confirmation (15-20%) |
| 8 | msu_lag_1h | DECIMAL(10,2) | Calculated | Medium | Spike detection |
| 9 | msu_rolling_mean_2h | DECIMAL(10,2) | Calculated | High | Early trend (10-15%) |
| 10 | msu_rolling_mean_4h | DECIMAL(10,2) | Calculated | Medium | R4HA validation |
| 11 | hour_of_day | INTEGER | Derived | High | Daily patterns (6-10%) |
| 12 | day_of_week | INTEGER | Derived | Medium | Weekly patterns (3-5%) |
| 13 | is_batch_window | BOOLEAN | Schedule | High | Spike context (5-8%) |
| 14 | batch_jobs_running | INTEGER | SMF 30 | High | Workload driver (8-12%) |
| 15 | batch_cpu_seconds | DECIMAL(12,2) | SMF 30 | Medium | Intensity measure |

**TARGET VARIABLE (what we predict):**
- `target = r4ha_msu shifted 16 periods into the future (4 hours ahead)`

---
---

## Column Specification

### 1. TIMESTAMP
**Column Name:** `timestamp`  
**Data Type:** TIMESTAMP (YYYY-MM-DD HH:MI:SS)  
**Source:** SMF 70 Record  
**Frequency:** Every 15 minutes  
**Required:** Yes  

**How to Collect:**
```
SMF Field: SMF70DTE (Date) + SMF70TME (Time)
Parse and combine into single timestamp
Example: 2025-01-15 14:30:00
```

**Relationship to R4HA:**
- Provides temporal context for all measurements
- Enables time-series analysis and trend detection
- Used to create time-based features (hour, day)
- Critical for calculating lag features (looking back in time)

**Why It Matters:**
Without precise timestamps, you cannot:
- Calculate rolling averages (R4HA requires exact 4-hour windows)
- Create lag features (what happened 1 hour ago?)
- Detect daily/weekly patterns

---

### 2. SYSTEM_ID
**Column Name:** `system_id`  
**Data Type:** VARCHAR(8)  
**Source:** SMF 70 Record  
**Frequency:** Every 15 minutes  
**Required:** Yes  

**How to Collect:**
```
SMF Field: SMF70SID (System ID)
Clean whitespace and store as string
Example: 'PROD01', 'LPAR2A'
```

**Relationship to R4HA:**
- Identifies which mainframe system/LPAR the data belongs to
- Allows separate models for different systems (each has unique patterns)
- Prevents mixing data from different systems with different capacities

**Why It Matters:**
- System A with 1000 MSU capacity behaves differently from System B with 500 MSU
- Batch windows differ by system
- Each system needs its own baseline and predictions

---

### 3. MSU_CURRENT
**Column Name:** `msu_current`  
**Data Type:** DECIMAL(10,2)  
**Source:** SMF 70 Record  
**Frequency:** Every 15 minutes  
**Required:** Yes  

**How to Collect:**
```
SMF Field: SMF70CPC (CPU Consumption) or calculated from:
  MSU = (CPU_seconds / Interval_seconds) * CPU_capacity_MSU
  
Example calculation:
  CPU_seconds = 650 seconds in 15-min interval
  Interval = 900 seconds (15 minutes)
  Capacity = 1000 MSU
  MSU_current = (650/900) * 1000 = 722.22 MSU
```

**Relationship to R4HA:**
- **DIRECT BUILDING BLOCK** - R4HA is calculated FROM this value
- R4HA = Average of last 16 msu_current readings (4 hours)
- Current MSU shows instantaneous consumption
- Spikes in msu_current directly impact future R4HA

**Why It Matters:**
This is the RAW DATA that everything else is built from. Without msu_current:
- Cannot calculate R4HA
- Cannot predict future consumption
- Cannot identify patterns

**Formula Relationship:**
```
R4HA at time T = Average(msu_current[T-15min], msu_current[T-30min], ..., msu_current[T-4hours])
```

---

### 4. R4HA_MSU
**Column Name:** `r4ha_msu`  
**Data Type:** DECIMAL(10,2)  
**Source:** Calculated from msu_current  
**Frequency:** Every 15 minutes  
**Required:** Yes  

**How to Collect:**
```python
# Calculate R4HA as rolling 4-hour average
# 4 hours = 16 periods of 15 minutes each
df['r4ha_msu'] = df.groupby('system_id')['msu_current']\
    .rolling(window=16, min_periods=16)\
    .mean()\
    .reset_index(0, drop=True)
```

**Example Calculation:**
```
Time        msu_current    r4ha_msu (avg of last 16 readings)
10:00       700            720
10:15       710            718
10:30       750            722
10:45       800            728  <- This is what we'll try to predict
```

**Relationship to R4HA:**
- **THIS IS THE CURRENT R4HA VALUE** (not the target yet)
- Shows smoothed consumption over 4 hours
- More stable than msu_current (removes short spikes)
- Used as input feature to predict FUTURE R4HA

**Why It Matters:**
- Current R4HA is the best predictor of future R4HA
- Shows trend direction (rising, falling, stable)
- If R4HA is already high, likely to stay high or go higher
- Provides context for current system state

**Target Variable Note:**
The TARGET we predict is: `r4ha_msu shifted 4 hours into the future`
```python
df['target'] = df['r4ha_msu'].shift(-16)  # Predict R4HA 4 hours from now
```

---

### 5. CPU_UTILIZATION_PCT
**Column Name:** `cpu_utilization_pct`  
**Data Type:** DECIMAL(5,2)  
**Source:** SMF 70 Record  
**Frequency:** Every 15 minutes  
**Required:** Yes  

**How to Collect:**
```
SMF Fields: SMF70CPT (CPU Time) and SMF70TME (Interval Time)
Formula: (Total_CPU_seconds / Total_Interval_seconds) * 100

Example:
  CPU consumed = 650 seconds
  Interval = 900 seconds (15 minutes)
  CPU_utilization = (650/900) * 100 = 72.22%
```

**Relationship to R4HA:**
- **CORRELATED WITH MSU** - Higher CPU% usually means higher MSU
- Shows how hard the processors are working
- When CPU hits 80-90%, MSU consumption spikes
- Helps model understand capacity constraints

**Why It Matters:**
- CPU% at 95% = system is maxed out = MSU will spike
- CPU% at 30% = plenty of headroom = MSU stable
- Provides context beyond just MSU numbers
- Different relationship at different utilization levels (non-linear)

**Example Pattern:**
```
CPU%    MSU     Behavior
30%     400     Stable, predictable
60%     700     Normal operations
85%     950     Getting constrained
95%     1100    Spike zone - unpredictable
```

---

### 6. R4HA_LAG_1H
**Column Name:** `r4ha_lag_1h`  
**Data Type:** DECIMAL(10,2)  
**Source:** Calculated from r4ha_msu  
**Frequency:** Every 15 minutes  
**Required:** Yes  

**How to Collect:**
```python
# Lag by 4 periods = 1 hour (4 √ó 15 minutes)
df['r4ha_lag_1h'] = df.groupby('system_id')['r4ha_msu'].shift(4)
```

**Example:**
```
Time        r4ha_msu    r4ha_lag_1h
10:00       720         680  (this was r4ha at 09:00)
10:15       722         682
10:30       725         685
10:45       728         688
```

**Relationship to R4HA:**
- **STRONGEST PREDICTOR** (typically 25-30% feature importance)
- Recent past predicts near future
- If R4HA was 800 one hour ago, likely still near 800 now
- Shows momentum and trend direction

**Why It Matters:**
- Systems don't change instantly - they have inertia
- Batch jobs don't complete in minutes - they take hours
- High R4HA 1 hour ago = high R4HA now = high R4HA in 4 hours
- Captures short-term trends

**Pattern Recognition:**
```
If r4ha_lag_1h = 900 and r4ha_msu = 950:
  ‚Üí Trend is UP, predict higher
  
If r4ha_lag_1h = 900 and r4ha_msu = 850:
  ‚Üí Trend is DOWN, predict lower
```

---

### 7. R4HA_LAG_2H
**Column Name:** `r4ha_lag_2h`  
**Data Type:** DECIMAL(10,2)  
**Source:** Calculated from r4ha_msu  
**Frequency:** Every 15 minutes  
**Required:** Yes  

**How to Collect:**
```python
# Lag by 8 periods = 2 hours (8 √ó 15 minutes)
df['r4ha_lag_2h'] = df.groupby('system_id')['r4ha_msu'].shift(8)
```

**Relationship to R4HA:**
- **SECOND STRONGEST PREDICTOR** (15-20% feature importance)
- Confirms trend detected by 1-hour lag
- Helps identify acceleration or deceleration
- Provides longer-term context

**Why It Matters:**
Compare 3 values to understand trend:
```
r4ha_lag_2h = 700  (2 hours ago)
r4ha_lag_1h = 800  (1 hour ago)  
r4ha_msu = 900     (now)

Pattern: Accelerating upward ‚Üí Predict even higher spike
```

Versus:
```
r4ha_lag_2h = 900
r4ha_lag_1h = 850
r4ha_msu = 820

Pattern: Decelerating downward ‚Üí Predict stabilization
```

**Batch Job Context:**
- Many batch jobs run 2-4 hours
- 2-hour lag captures beginning of batch window
- If batch started 2 hours ago, likely still running

---

### 8. MSU_LAG_1H
**Column Name:** `msu_lag_1h`  
**Data Type:** DECIMAL(10,2)  
**Source:** Calculated from msu_current  
**Frequency:** Every 15 minutes  
**Required:** Yes  

**How to Collect:**
```python
# Lag by 4 periods = 1 hour
df['msu_lag_1h'] = df.groupby('system_id')['msu_current'].shift(4)
```

**Relationship to R4HA:**
- Shows instantaneous consumption 1 hour ago (not smoothed)
- Captures spikes that R4HA smooths out
- Helps identify volatile vs. stable patterns
- Complements r4ha_lag_1h

**Why It Matters:**
**Difference from r4ha_lag_1h:**
- `r4ha_lag_1h` = smooth 4-hour average from 1 hour ago
- `msu_lag_1h` = actual point-in-time MSU from 1 hour ago

**Example Showing Difference:**
```
Time    msu_current    r4ha_msu    msu_lag_1h    r4ha_lag_1h
09:00   600           700         -             -
09:15   1200 (spike)  720         -             -
09:30   700           735         -             -
10:00   750           765         -             -
10:15   800           780         600           700
10:30   850           795         1200          720  <- Spike captured!
```

The model learns:
- When msu_lag_1h spiked but r4ha_lag_1h stayed smooth ‚Üí Short spike, not sustained
- When both spiked ‚Üí Real trend, predict higher

---

### 9. MSU_ROLLING_MEAN_2H
**Column Name:** `msu_rolling_mean_2h`  
**Data Type:** DECIMAL(10,2)  
**Source:** Calculated from msu_current  
**Frequency:** Every 15 minutes  
**Required:** Yes  

**How to Collect:**
```python
# Rolling average over 8 periods = 2 hours
df['msu_rolling_mean_2h'] = df.groupby('system_id')['msu_current']\
    .rolling(window=8, min_periods=8)\
    .mean()\
    .reset_index(0, drop=True)
```

**Relationship to R4HA:**
- Smooths noise from individual 15-minute readings
- More responsive than R4HA (2 hours vs 4 hours)
- Shows emerging trends before they fully appear in R4HA
- Acts as "early warning" signal

**Why It Matters:**
**Comparison of Time Windows:**
```
Time    msu_current    2h_avg    4h_avg(R4HA)
08:00   700           700       700
09:00   750           725       712
10:00   800           750       725
11:00   850           775       737
12:00   900           800       750  <- R4HA lags behind!
```

The 2-hour average rises faster than R4HA, giving the model a signal that R4HA will catch up soon.

**Pattern Recognition:**
```
If 2h_avg > R4HA:  ‚Üí MSU rising, predict R4HA will increase
If 2h_avg < R4HA:  ‚Üí MSU falling, predict R4HA will decrease
If 2h_avg ‚âà R4HA:  ‚Üí Stable, predict continuation
```

---

### 10. MSU_ROLLING_MEAN_4H
**Column Name:** `msu_rolling_mean_4h`  
**Data Type:** DECIMAL(10,2)  
**Source:** Calculated from msu_current  
**Frequency:** Every 15 minutes  
**Required:** Yes  

**How to Collect:**
```python
# Rolling average over 16 periods = 4 hours
df['msu_rolling_mean_4h'] = df.groupby('system_id')['msu_current']\
    .rolling(window=16, min_periods=16)\
    .mean()\
    .reset_index(0, drop=True)
```

**Relationship to R4HA:**
- **SHOULD BE IDENTICAL OR VERY CLOSE TO r4ha_msu**
- Both are 4-hour rolling averages
- Provides validation/cross-check
- May differ slightly due to calculation timing

**Why It Matters:**
- Confirms R4HA calculation is correct
- Provides alternative view of same metric
- Model can compare consistency between two calculations
- Acts as a sanity check on data quality

**Expected Relationship:**
```
msu_rolling_mean_4h ‚âà r4ha_msu (should match within 1-2%)

If they differ significantly:
  ‚Üí Data quality issue
  ‚Üí Check calculation methodology
  ‚Üí Verify time alignment
```

---

### 11. HOUR_OF_DAY
**Column Name:** `hour_of_day`  
**Data Type:** INTEGER (0-23)  
**Source:** Derived from timestamp  
**Frequency:** Every 15 minutes  
**Required:** Yes  

**How to Collect:**
```python
# Extract hour from timestamp
df['hour_of_day'] = df['timestamp'].dt.hour

# Result: 0, 1, 2, ..., 22, 23
```

**Relationship to R4HA:**
- **CAPTURES DAILY PATTERNS** (6-10% feature importance)
- Online transactions peak during business hours
- Batch jobs run at specific times (often midnight-6am)
- Different hours have different baseline MSU levels

**Why It Matters:**
**Typical Mainframe Daily Pattern:**
```
Hour    Pattern             Expected MSU    R4HA Risk
00-06   Heavy batch         800-1200       HIGH (month-end)
06-08   Batch completion    600-800        MEDIUM
08-17   Online peak         700-900        MEDIUM
17-20   Batch starts        800-1000       MEDIUM-HIGH
20-24   Batch window        900-1100       HIGH
```

The model learns:
- Hour 2 (2am) + batch_window = HIGH R4HA expected (normal)
- Hour 2 (2am) + NO batch = LOW R4HA expected
- Hour 14 (2pm) + HIGH MSU = UNUSUAL, investigate

**Pattern Recognition:**
```
Same MSU value means different things at different hours:

MSU = 900 at 03:00 (3am) ‚Üí NORMAL (batch running)
MSU = 900 at 14:00 (2pm) ‚Üí ALERT! (unexpected spike)
```

---

### 12. DAY_OF_WEEK
**Column Name:** `day_of_week`  
**Data Type:** INTEGER (1-7)  
**Source:** Derived from timestamp  
**Frequency:** Every 15 minutes  
**Required:** Yes  

**How to Collect:**
```python
# Extract day of week (1=Monday, 7=Sunday)
df['day_of_week'] = df['timestamp'].dt.dayofweek + 1

# Result: 1, 2, 3, 4, 5, 6, 7
```

**Relationship to R4HA:**
- **CAPTURES WEEKLY PATTERNS** (3-5% feature importance)
- Monday ‚â† Friday ‚â† Sunday in terms of workload
- Month-end processing often on specific days
- Weekend batch windows are longer

**Why It Matters:**
**Typical Weekly Pattern:**
```
Day         Business Activity           MSU Pattern
Monday      Heavy (weekend catchup)     HIGH
Tuesday     Normal operations           MEDIUM
Wednesday   Normal operations           MEDIUM
Thursday    Normal operations           MEDIUM
Friday      Report generation           MEDIUM-HIGH
Saturday    Extended batch windows      HIGH (batch)
Sunday      Light processing            LOW
```

**Real-World Examples:**
- **Monday morning:** Weekend batch completion + online users returning = spike
- **Friday afternoon:** Weekly reports + month-end prep = higher MSU
- **Saturday 2am:** Long-running batch jobs = expected high R4HA
- **Sunday evening:** System quiet, low MSU baseline

**Month-End Interaction:**
```
If day_of_week = Friday AND is_month_end = True:
  ‚Üí Extra processing (weekly + monthly)
  ‚Üí Predict 20-30% higher R4HA than normal Friday
```

---

### 13. IS_BATCH_WINDOW
**Column Name:** `is_batch_window`  
**Data Type:** BOOLEAN (0 or 1)  
**Source:** Batch Schedule System  
**Frequency:** Every 15 minutes  
**Required:** Yes  

**How to Collect:**
```
Data Source: Job Scheduler (Control-M, TWS, CA-7)
Export: Daily batch schedule with time windows

Example Schedule File:
Window_Name, Start_Time, End_Time, Days
NIGHTLY_BATCH, 00:00, 06:00, Mon-Sun
MONTH_END, 22:00, 08:00, Last_Business_Day
EOD_PROCESSING, 18:00, 22:00, Mon-Fri

Logic:
If current_timestamp is within any batch window:
  is_batch_window = 1
Else:
  is_batch_window = 0
```

**Example Mapping:**
```python
def determine_batch_window(timestamp):
    hour = timestamp.hour
    day = timestamp.weekday()
    
    # Nightly batch: 00:00-06:00 every day
    if 0 <= hour < 6:
        return 1
    
    # EOD processing: 18:00-22:00 weekdays
    if day < 5 and 18 <= hour < 22:
        return 1
    
    # Otherwise, no batch window
    return 0

df['is_batch_window'] = df['timestamp'].apply(determine_batch_window)
```

**Relationship to R4HA:**
- **STRONG PREDICTOR OF SPIKES** (5-8% feature importance)
- Batch jobs consume 2-3x more CPU than online transactions
- Batch windows = planned high MSU periods
- Model learns: batch_window = expect higher R4HA

**Why It Matters:**
**Context Changes Everything:**
```
Scenario 1: Regular Hour
  MSU = 900
  is_batch_window = 0
  ‚Üí ALERT! Unexpected spike, investigate
  
Scenario 2: Batch Window
  MSU = 900
  is_batch_window = 1
  ‚Üí NORMAL, expected behavior, no alert
```

**Prediction Impact:**
```
Without is_batch_window:
  Model sees: "MSU suddenly jumped to 1000 at 2am"
  Prediction: Uncertain, could spike more or drop
  
With is_batch_window:
  Model sees: "MSU at 1000 at 2am + batch_window=True"
  Prediction: This is normal, will stay high until 6am, then drop
```

---

### 14. BATCH_JOBS_RUNNING
**Column Name:** `batch_jobs_running`  
**Data Type:** INTEGER  
**Source:** SMF 30 Records (aggregated)  
**Frequency:** Every 15 minutes  
**Required:** Yes  

**How to Collect:**
```
SMF Source: Type 30, Subtype 4/5 (Job Start/End)

Method 1: Real-time aggregation
  For each 15-minute interval:
    Count jobs with: start_time <= interval AND end_time >= interval
    
Method 2: Query from job tracking database
  SELECT COUNT(*) FROM active_jobs
  WHERE timestamp = @interval
  AND job_class IN ('A','B','C')  -- Batch classes only
  
Example SMF 30 Processing:
  Job ABC123: Started 14:15, Ended 16:45
  Intervals it appears in:
    14:15, 14:30, 14:45, 15:00, 15:15, ..., 16:30, 16:45
```

**Aggregation Logic:**
```python
# From SMF 30 job records
jobs = pd.read_csv('smf30_jobs.csv')
jobs['start'] = pd.to_datetime(jobs['job_start_time'])
jobs['end'] = pd.to_datetime(jobs['job_end_time'])

# Create 15-minute intervals
intervals = pd.date_range(start='2024-01-01', end='2024-12-31', freq='15min')

# Count running jobs per interval
for interval in intervals:
    count = len(jobs[(jobs['start'] <= interval) & (jobs['end'] >= interval)])
    df.loc[df['timestamp'] == interval, 'batch_jobs_running'] = count
```

**Relationship to R4HA:**
- **DIRECT DRIVER OF MSU** (8-12% feature importance)
- More jobs = more CPU consumption = higher MSU
- Each job consumes CPU, combined effect raises R4HA
- Non-linear relationship (100 jobs ‚â† 2x CPU of 50 jobs)

**Why It Matters:**
**MSU Consumption by Job Count:**
```
Jobs Running    Typical MSU    R4HA Impact
0-10           300-400        Baseline
10-20          450-600        Light batch
20-40          600-800        Normal batch
40-60          800-1000       Heavy batch
60+            1000-1200      Peak/month-end
```

**Prediction Scenarios:**
```
Scenario A:
  batch_jobs_running = 25
  hour = 14 (2pm)
  is_batch_window = 0
  ‚Üí UNUSUAL, predict spike, send alert
  
Scenario B:
  batch_jobs_running = 45
  hour = 2 (2am)
  is_batch_window = 1
  ‚Üí NORMAL, batch window in progress, expect continuation
```

**Combined with Time Features:**
```
Pattern 1: Growing batch load
  01:00 ‚Üí 15 jobs ‚Üí MSU 700
  02:00 ‚Üí 35 jobs ‚Üí MSU 850
  03:00 ‚Üí 50 jobs ‚Üí MSU 950
  Model learns: Jobs increasing ‚Üí Predict R4HA spike at 04:00
  
Pattern 2: Batch completion
  03:00 ‚Üí 60 jobs ‚Üí MSU 1100
  04:00 ‚Üí 45 jobs ‚Üí MSU 950
  05:00 ‚Üí 20 jobs ‚Üí MSU 750
  Model learns: Jobs decreasing ‚Üí Predict R4HA drop at 06:00
```

---

### 15. BATCH_CPU_SECONDS
**Column Name:** `batch_cpu_seconds`  
**Data Type:** DECIMAL(12,2)  
**Source:** SMF 30 Records (aggregated)  
**Frequency:** Every 15 minutes  
**Required:** Yes  

**How to Collect:**
```
SMF Source: Type 30, Field SMF30CPT (CPU Time)

Aggregation per 15-minute interval:
  SUM(cpu_seconds) for all batch jobs active during interval
  
Example:
  Interval: 14:00-14:15
  Job A: consumed 45 seconds
  Job B: consumed 120 seconds  
  Job C: consumed 87 seconds
  batch_cpu_seconds = 45 + 120 + 87 = 252 seconds
```

**Collection Query:**
```python
# Aggregate CPU consumption by interval
batch_cpu = smf30_data.groupby(['interval_15min', 'system_id']).agg({
    'cpu_seconds': 'sum',
    'job_name': 'count'  # This becomes batch_jobs_running
}).reset_index()

batch_cpu.columns = ['timestamp', 'system_id', 'batch_cpu_seconds', 'batch_jobs_running']
```

**Relationship to R4HA:**
- **INTENSITY MEASURE** - Not just how many jobs, but how hard they're working
- 10 CPU-intensive jobs ‚â† 10 light jobs
- Direct contributor to MSU calculation
- Complements batch_jobs_running

**Why It Matters:**
**Two Different Patterns:**
```
Pattern A: Many light jobs
  batch_jobs_running = 50
  batch_cpu_seconds = 200 (4 CPU-sec per job avg)
  MSU impact = Moderate
  
Pattern B: Few heavy jobs
  batch_jobs_running = 10
  batch_cpu_seconds = 400 (40 CPU-sec per job avg)
  MSU impact = High
```

**Relationship Between Columns:**
```
batch_jobs_running = Quantity (how many)
batch_cpu_seconds = Intensity (how much work)

Combined insight:
  High jobs + High CPU = Heavy processing ‚Üí High R4HA
  High jobs + Low CPU = Many idle/waiting jobs ‚Üí Lower R4HA
  Low jobs + High CPU = CPU-intensive work ‚Üí Medium-High R4HA
  Low jobs + Low CPU = Light workload ‚Üí Low R4HA
```

**Prediction Example:**
```
Current State:
  batch_jobs_running = 30
  batch_cpu_seconds = 450
  msu_current = 850
  
Model calculates average CPU per job:
  450 / 30 = 15 CPU-seconds per job
  
If this intensity continues for 4 hours:
  Predicted total CPU = 15 * 30 jobs * 16 intervals = 7,200 CPU-seconds
  Predicted MSU impact = HIGH
  Predicted R4HA = 900-950 range
```

---

## Data Collection Workflow

### Step 1: Extract SMF 70 Data (Every 15 minutes)
```
JCL/Batch Job: Run hourly, extract last hour of SMF 70 records
Output: smf70_YYYYMMDD_HH.csv

Columns extracted:
- SMF70DTE + SMF70TME ‚Üí timestamp
- SMF70SID ‚Üí system_id
- SMF70CPC ‚Üí msu_current
- SMF70CPT / SMF70TME ‚Üí cpu_utilization_pct
```

### Step 2: Extract SMF 30 Data (Hourly)
```
JCL/Batch Job: Run hourly, aggregate SMF 30 records
Output: smf30_batch_aggregated_YYYYMMDD_HH.csv

For each 15-minute interval:
- COUNT(active jobs) ‚Üí batch_jobs_running
- SUM(SMF30CPT) ‚Üí batch_cpu_seconds
```

### Step 3: Load Batch Schedule (Daily)
```
Extract from scheduler: Control-M, TWS, etc.
Output: batch_schedule_YYYYMMDD.csv

Contains: window definitions, start/end times
Used to populate: is_batch_window column
```

### Step 4: Feature Engineering (Python/SQL)
```python
# Load data
df = load_smf70_and_smf30_data()

# Calculate R4HA
df['r4ha_msu'] = df.groupby('system_id')['msu_current']\
    .rolling(16).mean().reset_index(0, drop=True)

# Create lags
df['r4ha_lag_1h'] = df.groupby('system_id')['r4ha_msu'].shift(4)
df['r4ha_lag_2h'] = df.groupby('system_id')['r4ha_msu'].shift(8)
df['msu_lag_1h'] = df.groupby('system_id')['msu_current'].shift(4)

# Create rolling means
df['msu_rolling_mean_2h'] = df.groupby('system_id')['msu_current']\
    .rolling(8).mean().reset_index(0, drop=True)
df['msu_rolling_mean_4h'] = df.groupby('system_id')['msu_current']\
    .rolling(16).mean().reset_index(0, drop=True)

# Extract temporal features
df['hour_of_day'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek + 1

# Map batch windows
df['is_batch_window'] = df['timestamp'].apply(map_to_batch_window)

# Create target (4 hours ahead)
df['target'] = df.groupby('system_id')['r4ha_msu'].shift(-16)
```

---

## Summary Table: All 15 Columns

| # | Column | Type | Source | Importance | Purpose |
|---|--------|------|--------|------------|---------|
| 1 | timestamp | TIMESTAMP | SMF 70 | Critical | Time alignment |
| 2 | system_id | VARCHAR(8) | SMF 70 | Critical | System identification |
| 3 | msu_current | DECIMAL(10,2) | SMF 70 | Critical | Raw MSU data |
| 4 | r4ha_msu | DECIMAL(10,2) | Calculated | Critical | Current R4HA (input) |
| 5 | cpu_utilization_pct | DECIMAL(5,2) | SMF 70 | High | Capacity indicator |
| 6 | r4ha_lag_1h | DECIMAL(10,2) | Calculated | Very High | Recent trend (25-30%) |
| 7 | r4ha_lag_2h | DECIMAL(10,2) | Calculated | High | Trend confirmation (15-20%) |
| 8 | msu_lag_1h | DECIMAL(10,2) | Calculated | Medium | Spike detection |
| 9 | msu_rolling_mean_2h | DECIMAL(10,2) | Calculated | High | Early trend (10-15%) |
| 10 | msu_rolling_mean_4h | DECIMAL(10,2) | Calculated | Medium | R4HA validation |
| 11 | hour_of_day | INTEGER | Derived | High | Daily patterns (6-10%) |
| 12 | day_of_week | INTEGER | Derived | Medium | Weekly patterns (3-5%) |
| 13 | is_batch_window | BOOLEAN | Schedule | High | Spike context (5-8%) |
| 14 | batch_jobs_running | INTEGER | SMF 30 | High | Workload driver (8-12%) |
| 15 | batch_cpu_seconds | DECIMAL(12,2) | SMF 30 | Medium | Intensity measure |

**TARGET VARIABLE (what we predict):**
- `target = r4ha_msu shifted 16 periods into the future (4 hours ahead)`

---

## Visual Relationship Map

```
PREDICTIVE CHAIN - How Features Connect to Target R4HA:

Past State (1-2 hours ago)
    r4ha_lag_1h ‚îÄ‚îÄ‚îê
    r4ha_lag_2h ‚îÄ‚îÄ‚îº‚îÄ‚Üí Show TREND direction
    msu_lag_1h ‚îÄ‚îÄ‚îÄ‚îò

Current Smoothed State
    r4ha_msu ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚Üí Current baseline
    msu_rolling_mean_2h ‚Üí Recent trend
    msu_rolling_mean_4h ‚Üí Stable average

Current Raw State
    msu_current ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚Üí Latest reading
    cpu_utilization_pct ‚Üí Capacity stress

Contextual State
    hour_of_day ‚îÄ‚îÄ‚îê
    day_of_week ‚îÄ‚îÄ‚îº‚îÄ‚îÄ‚Üí Expected PATTERN
    is_batch_window ‚îò

Workload Drivers
    batch_jobs_running ‚îÄ‚îÄ‚Üí Quantity of work
    batch_cpu_seconds ‚îÄ‚îÄ‚îÄ‚Üí Intensity of work

                ‚Üì
        [XGBoost Model]
                ‚Üì
        TARGET: R4HA in 4 hours
```

---

## Feature Correlation with R4HA (Expected)

**Strength of Relationship:**

```
Very Strong (r > 0.8):
‚úì r4ha_lag_1h         ‚Üí 0.85-0.90
‚úì r4ha_lag_2h         ‚Üí 0.80-0.85
‚úì msu_rolling_mean_4h ‚Üí 0.75-0.85

Strong (r > 0.6):
‚úì msu_rolling_mean_2h ‚Üí 0.70-0.75
‚úì r4ha_msu            ‚Üí 0.70-0.80
‚úì msu_current         ‚Üí 0.65-0.75

Moderate (r > 0.4):
‚úì batch_jobs_running  ‚Üí 0.50-0.65
‚úì msu_lag_1h          ‚Üí 0.55-0.65
‚úì cpu_utilization_pct ‚Üí 0.45-0.60
‚úì batch_cpu_seconds   ‚Üí 0.45-0.60

Contextual (non-linear):
‚úì hour_of_day         ‚Üí Pattern-based
‚úì day_of_week         ‚Üí Pattern-based
‚úì is_batch_window     ‚Üí Binary indicator

r = correlation coefficient
```

---

## Real-World Example: How Features Work Together

### Scenario: Predicting Month-End Spike

**Current Time:** Friday 18:00 (6pm), Last business day of month

**Feature Values:**
```
timestamp = 2025-01-31 18:00:00
system_id = PROD01
msu_current = 850
r4ha_msu = 780
cpu_utilization_pct = 72.5
r4ha_lag_1h = 750
r4ha_lag_2h = 720
msu_lag_1h = 800
msu_rolling_mean_2h = 825
msu_rolling_mean_4h = 775
hour_of_day = 18
day_of_week = 5 (Friday)
is_batch_window = 1 (EOD processing started)
batch_jobs_running = 35
batch_cpu_seconds = 420
```

**Model Analysis:**

1. **Trend Detection:**
   - r4ha_lag_2h (720) < r4ha_lag_1h (750) < r4ha_msu (780)
   - **Conclusion:** Steady upward trend

2. **Acceleration Check:**
   - 2h‚Üí1h increase: 750 - 720 = 30 MSU
   - 1h‚Üínow increase: 780 - 750 = 30 MSU
   - **Conclusion:** Consistent acceleration, not slowing down

3. **Recent Activity:**
   - msu_rolling_mean_2h (825) > r4ha_msu (780)
   - **Conclusion:** Last 2 hours were HIGH, R4HA still catching up

4. **Capacity Status:**
   - cpu_utilization_pct = 72.5% (comfortable, not maxed)
   - **Conclusion:** Room to grow further

5. **Context:**
   - hour_of_day = 18 (typical EOD batch start time)
   - day_of_week = 5 (Friday + month-end = extra processing)
   - is_batch_window = 1 (known spike period)
   - **Conclusion:** This is EXPECTED behavior for this time/day

6. **Workload Intensity:**
   - batch_jobs_running = 35 (moderate-high)
   - batch_cpu_seconds = 420 (heavy work)
   - Average: 420/35 = 12 CPU-sec per job (intensive jobs)
   - **Conclusion:** Jobs are CPU-heavy, will sustain high MSU

**MODEL PREDICTION:**
```
Predicted R4HA at 22:00 (4 hours from now): 920-950 MSU

Reasoning:
- Trend is UP (3 lag features confirm)
- Recent activity HIGH (2h mean > current R4HA)
- Context supports spike (month-end Friday EOD)
- Workload will continue (batch window runs until 22:00)
- Capacity available (not maxed, can grow)

Confidence: HIGH (85%)
Alert: No (this is expected for month-end)
```

---

## Data Quality Checks

Before training model, validate your data:

### 1. Completeness Check
```python
# Check for missing 15-minute intervals
df_check = df.groupby('system_id').apply(
    lambda x: x.set_index('timestamp').resample('15min').asfreq()
)

missing_intervals = df_check[df_check['msu_current'].isna()]
print(f"Missing intervals: {len(missing_intervals)}")
# Should be 0 or very few
```

### 2. R4HA Calculation Validation
```python
# Verify R4HA matches manual calculation
manual_r4ha = df.groupby('system_id')['msu_current']\
    .rolling(16).mean().reset_index(0, drop=True)

difference = abs(df['r4ha_msu'] - manual_r4ha)
print(f"Max difference: {difference.max()}")
# Should be < 0.01
```

### 3. Lag Feature Validation
```python
# Verify lag is correct
for i in range(1, len(df)):
    if df.loc[i, 'system_id'] == df.loc[i-4, 'system_id']:
        expected_lag = df.loc[i-4, 'r4ha_msu']
        actual_lag = df.loc[i, 'r4ha_lag_1h']
        
        if abs(expected_lag - actual_lag) > 0.01:
            print(f"Lag mismatch at row {i}")
```

### 4. Value Range Validation
```python
# Check for unrealistic values
print(df[['msu_current', 'r4ha_msu', 'cpu_utilization_pct']].describe())

# Flags:
# - MSU < 0 or > 5000 ‚Üí Error
# - CPU% < 0 or > 100 ‚Üí Error
# - R4HA wildly different from MSU ‚Üí Check calculation
```

### 5. Temporal Consistency
```python
# Verify timestamps are sequential
df_sorted = df.sort_values(['system_id', 'timestamp'])
time_diffs = df_sorted.groupby('system_id')['timestamp'].diff()

# Should be exactly 15 minutes (or 900 seconds)
unexpected = time_diffs[time_diffs != pd.Timedelta(minutes=15)]
print(f"Unexpected time gaps: {len(unexpected)}")
```

---

## Sample Data Extract

### Example: 2 Hours of Data for PROD01 System

```
timestamp           sys_id  msu_cur r4ha  cpu%  r4ha_1h r4ha_2h msu_1h  mean_2h mean_4h hr dow btch jobs cpu_sec
2025-01-15 14:00:00 PROD01  720     705   68.5  695     680     710     715     708     14  3   0    12   145
2025-01-15 14:15:00 PROD01  735     708   70.2  698     682     715     720     710     14  3   0    14   158
2025-01-15 14:30:00 PROD01  750     712   72.8  702     685     720     728     715     14  3   0    15   162
2025-01-15 14:45:00 PROD01  760     715   74.1  705     688     735     735     718     14  3   0    16   178
2025-01-15 15:00:00 PROD01  775     720   75.5  708     695     750     743     722     15  3   0    18   195
2025-01-15 15:15:00 PROD01  790     725   77.2  712     698     760     750     728     15  3   0    19   208
2025-01-15 15:30:00 PROD01  805     732   78.8  715     702     775     758     735     15  3   0    21   225
2025-01-15 15:45:00 PROD01  820     738   80.5  720     705     790     768     742     15  3   0    23   242
```

**Column Legend:**
- sys_id = system_id
- msu_cur = msu_current
- r4ha = r4ha_msu
- cpu% = cpu_utilization_pct
- r4ha_1h = r4ha_lag_1h
- r4ha_2h = r4ha_lag_2h
- msu_1h = msu_lag_1h
- mean_2h = msu_rolling_mean_2h
- mean_4h = msu_rolling_mean_4h
- hr = hour_of_day
- dow = day_of_week
- btch = is_batch_window
- jobs = batch_jobs_running
- cpu_sec = batch_cpu_seconds

**Observations from this sample:**
- Steady upward trend: MSU growing from 720‚Üí820
- R4HA smoothly following: 705‚Üí738
- Lag features show consistent increase
- Time: 2pm-4pm on Wednesday (normal business hours)
- No batch window active (btch=0)
- Moderate job count (12-23 jobs)

---

## Implementation Checklist

### Week 1: Data Collection Setup
- [ ] Configure SMF 70 data collection (15-min intervals)
- [ ] Set up SMF 30 extraction job (hourly)
- [ ] Export batch schedule from scheduler
- [ ] Create data storage tables/files
- [ ] Verify 6+ months of historical data available

### Week 2: Data Processing
- [ ] Parse SMF 70 records ‚Üí extract 5 base columns
- [ ] Parse SMF 30 records ‚Üí aggregate batch metrics
- [ ] Merge SMF 70 + SMF 30 by timestamp + system_id
- [ ] Calculate R4HA from msu_current
- [ ] Create lag features (shift operations)
- [ ] Create rolling mean features
- [ ] Extract temporal features from timestamp
- [ ] Map batch windows from schedule
- [ ] Validate data quality (completeness, ranges)

### Week 3: Model Development
- [ ] Create target variable (R4HA 4 hours ahead)
- [ ] Remove rows with missing target
- [ ] Split data: 75% train, 10% validation, 15% test (time-based)
- [ ] Train initial XGBoost model
- [ ] Evaluate on validation set (RMSE, MAE, R¬≤)
- [ ] Analyze feature importance
- [ ] Tune hyperparameters
- [ ] Test on holdout test set

### Week 4: Validation & Deployment
- [ ] Backtest predictions on recent data
- [ ] Compare predictions to actual R4HA spikes
- [ ] Calculate alert accuracy metrics
- [ ] Document model performance
- [ ] Create prediction pipeline
- [ ] Set up monitoring dashboard
- [ ] Plan monthly retraining schedule

---

## Minimum Viable Dataset Size

**For Training:**
- **Minimum:** 3 months of data (~8,600 records)
- **Recommended:** 6 months of data (~17,000 records)
- **Ideal:** 12 months of data (~35,000 records)

**Why 12 months?**
- Captures full yearly cycle (seasonal patterns)
- Includes all quarterly and year-end processing
- Multiple instances of month-end patterns
- Better model generalization

**Storage Requirements:**
- 15 columns √ó 35,000 rows √ó 8 bytes ‚âà 4 MB
- Very manageable, fits in memory easily

---

## Expected Model Performance

With TIER 1 features (15 columns):

**Accuracy Metrics:**
- **RMSE:** 40-60 MSU (typical)
- **MAE:** 30-45 MSU
- **R¬≤:** 0.75-0.85
- **MAPE:** 4-8%

**Practical Performance:**
- **Spike Detection:** 70-80% of R4HA spikes predicted correctly
- **False Positives:** 15-25% (predicted spike, didn't happen)
- **False Negatives:** 10-20% (missed spike)
- **Lead Time:** 4 hours advance warning

**Example Results:**
```
Actual R4HA: 950 MSU (spike)
Predicted R4HA: 920 MSU
Error: 30 MSU (3.2%)
Result: ‚úì Spike correctly identified 4 hours in advance

Actual R4HA: 650 MSU (normal)
Predicted R4HA: 680 MSU
Error: 30 MSU (4.6%)
Result: ‚úì No false alarm
```

---

## Next Steps After POC

Once you have working model with TIER 1:

**Phase 2 Enhancements (Add TIER 2 features):**
1. Add r4ha_lag_168h (same time last week)
2. Add is_month_end flag from business calendar
3. Add critical_batch_running count
4. Add msu_rate_change_1h (growth rate)
5. Add service_class metrics from SMF 72

**Expected Improvement:**
- TIER 1 alone: 70-75% accuracy
- TIER 1 + TIER 2: 85-90% accuracy
- Incremental gain: 10-15 percentage points

---

## Support Resources

### SMF Documentation
- **IBM Manual:** z/OS MVS System Management Facilities (SMF)
- **SMF 70:** RMF CPU Activity Record
- **SMF 72:** RMF Workload Activity Record  
- **SMF 30:** Job/Task Accounting Record

### Tools for SMF Parsing
- **MXG:** Commercial SMF parsing software
- **IBM's SMF Dump Utilities:** IFASMFDP, IFASMFDL
- **Custom COBOL/Rexx:** Write your own parser

### Python Libraries for ML
- **pandas:** Data manipulation
- **xgboost:** Gradient boosting model
- **scikit-learn:** Preprocessing, metrics
- **matplotlib/seaborn:** Visualization

---

## FAQ

**Q: What if I can't get SMF 30 data?**
A: You lose batch_jobs_running and batch_cpu_seconds. Still have 13 features, ~65-70% accuracy possible.

**Q: Can I use hourly data instead of 15-minute?**
A: Yes, but change window sizes: R4HA = 4 periods (hourly), lag_1h = 1 period. Slightly lower accuracy.

**Q: What if historical data has gaps?**
A: Forward-fill small gaps (<1 hour). If gaps >4 hours, exclude that period from training.

**Q: Do I need separate models per system?**
A: Recommended if systems have very different capacities/workloads. Otherwise, system_id as feature works.

**Q: How often retrain the model?**
A: Monthly recommended. Workload patterns evolve (new applications, changed batch schedules).

---

## Conclusion

These **15 TIER 1 columns** provide everything needed for a successful R4HA prediction POC:

**Core Data (5):** timestamp, system_id, msu_current, r4ha_msu, cpu_utilization_pct
**History (4):** r4ha_lag_1h, r4ha_lag_2h, msu_lag_1h, msu_rolling_mean_2h/4h  
**Context (3):** hour_of_day, day_of_week, is_batch_window
**Workload (2):** batch_jobs_running, batch_cpu_seconds

**Total implementation time:** 2-3 weeks
**Expected accuracy:** 70-75%
**Good enough to:** Demonstrate value, get stakeholder buy-in, plan production deployment

Start simple, prove the concept, then enhance!

