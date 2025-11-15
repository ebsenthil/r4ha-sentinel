



```
{
  "timestamp": "2025-01-15 21:30:00",
  "system_id": "PROD01",
  "description": "High MSU scenario during business hours",
  
  "prediction_features": {
    "msu_current": 690.0,
    "r4ha_msu": 710.0,
    "cpu_utilization_pct":92.5,
    "r4ha_lag_1h": 900.0,
    "r4ha_lag_2h": 880.0,
    "msu_lag_1h": 930.0,
    "msu_rolling_mean_2h": 940.0,
    "msu_rolling_mean_4h": 925.0,
    "hour_of_day": 14,
    "day_of_week": 3,
    "is_batch_window": 0,
    "batch_jobs_running": 22,
    "batch_cpu_seconds": 285.5
  },
  
  "threshold": 800,
  ---
```
```
  "wlm_context": {
    "source": "SMF Type 72 (Subtype 3)",
    "collection_time": "2025-01-15 14:30:00",
    "service_classes": [
      {
        "name": "ONLINE",
        "type": "transaction",
        "importance": 1,
        "goal_type": "response_time",
        "goal_value_ms": 500,
        "current_performance_ms": 420,
        "goal_achievement_pct": 95.2,
        "msu_consumption": 380.0,
        "cpu_percent": 40.0,
        "transactions_per_sec": 2500,
        "active_users": 450,
        "status": "meeting_goal",
        "can_reduce": false,
        "reason": "Critical online transactions - must maintain service levels"
      },
      {
        "name": "BATCH",
        "type": "batch",
        "importance": 3,
        "goal_type": "execution_velocity",
        "goal_value": 50,
        "current_performance": 52,
        "goal_achievement_pct": 104.0,
        "msu_consumption": 332.5,
        "cpu_percent": 35.0,
        "active_jobs": 28,
        "status": "exceeding_goal",
        "can_reduce": true,
        "potential_savings_msu": 66.5,
        "reason": "Currently exceeding goals - can throttle without impact"
      },
      {
        "name": "REPORTS",
        "type": "batch",
        "importance": 2,
        "goal_type": "execution_velocity",
        "goal_value": 30,
        "current_performance": 25,
        "goal_achievement_pct": 83.3,
        "msu_consumption": 142.5,
        "cpu_percent": 15.0,
        "active_jobs": 12,
        "status": "below_goal",
        "can_reduce": true,
        "potential_savings_msu": 71.25,
        "reason": "Already below goals - can defer to off-peak hours"
      },
      {
        "name": "DISCRETIONARY",
        "type": "discretionary",
        "importance": 5,
        "goal_type": "discretionary",
        "goal_value": 10,
        "current_performance": 8,
        "msu_consumption": 95.0,
        "cpu_percent": 10.0,
        "active_jobs": 5,
        "status": "running",
        "can_reduce": true,
        "potential_savings_msu": 95.0,
        "reason": "Low priority work - can stop completely"
      }
    ],
    "summary": {
      "total_service_classes": 4,
      "classes_meeting_goals": 2,
      "classes_below_goals": 1,
      "total_msu_consumption": 950.0,
      "total_potential_savings": 232.75,
      "capping_active": false,
      "resource_group": "PRODUCTION"
    }
  },
----
```

```
  
  "active_jobs": {
    "source": "SMF Type 30 (Active Jobs)",
    "collection_time": "2025-01-15 14:30:00",
    "jobs": [
      {
        "job_name": "PAYROLL01",
        "job_id": "JOB00123",
        "service_class": "BATCH",
        "priority": "HIGH",
        "start_time": "2025-01-15 13:45:00",
        "elapsed_minutes": 45,
        "estimated_remaining_minutes": 30,
        "cpu_seconds_consumed": 450.0,
        "msu_consumption": 75.0,
        "is_critical": true,
        "can_defer": false,
        "reason": "Payroll processing - business critical"
      },
      {
        "job_name": "CUST_AGING_RPT",
        "job_id": "JOB00124",
        "service_class": "REPORTS",
        "priority": "MEDIUM",
        "start_time": "2025-01-15 14:00:00",
        "elapsed_minutes": 30,
        "estimated_remaining_minutes": 45,
        "cpu_seconds_consumed": 180.0,
        "msu_consumption": 65.0,
        "is_critical": false,
        "can_defer": true,
        "potential_savings_msu": 65.0,
        "reason": "Monthly report - can defer to evening"
      },
      {
        "job_name": "DATA_ARCHIVE",
        "job_id": "JOB00125",
        "service_class": "BATCH",
        "priority": "LOW",
        "start_time": "2025-01-15 14:15:00",
        "elapsed_minutes": 15,
        "estimated_remaining_minutes": 60,
        "cpu_seconds_consumed": 95.0,
        "msu_consumption": 95.0,
        "is_critical": false,
        "can_defer": true,
        "potential_savings_msu": 95.0,
        "reason": "Archive job - not time sensitive"
      },
      {
        "job_name": "ADHOC_QUERY_001",
        "job_id": "JOB00126",
        "service_class": "DISCRETIONARY",
        "priority": "LOW",
        "start_time": "2025-01-15 14:20:00",
        "elapsed_minutes": 10,
        "estimated_remaining_minutes": 20,
        "cpu_seconds_consumed": 25.0,
        "msu_consumption": 25.0,
        "is_critical": false,
        "can_defer": true,
        "can_cancel": true,
        "potential_savings_msu": 25.0,
        "reason": "Ad-hoc user query - can cancel"
      },
      {
        "job_name": "USER_REPORT_GEN",
        "job_id": "JOB00127",
        "service_class": "DISCRETIONARY",
        "priority": "LOW",
        "start_time": "2025-01-15 14:25:00",
        "elapsed_minutes": 5,
        "estimated_remaining_minutes": 25,
        "cpu_seconds_consumed": 15.0,
        "msu_consumption": 35.0,
        "is_critical": false,
        "can_defer": true,
        "can_cancel": true,
        "potential_savings_msu": 35.0,
        "reason": "User-requested report - can cancel"
      }
    ],
    "summary": {
      "total_jobs": 5,
      "critical_jobs": 1,
      "deferrable_jobs": 4,
      "cancellable_jobs": 2,
      "total_msu_consumption": 295.0,
      "potential_savings_defer": 220.0,
      "potential_savings_cancel": 60.0
    }
  },
```


```
  "system_capacity": {
    "source": "SMF Type 70 (System Data)",
    "collection_time": "2025-01-15 14:30:00",
    "capacity": {
      "max_msu": 1200,
      "licensed_msu": 1000,
      "defined_capacity": 1200,
      "msu_soft_cap": 1000,
      "msu_hard_cap": 1200
    },
    "current_utilization": {
      "msu_usage": 950.0,
      "percent_of_license": 95.0,
      "percent_of_capacity": 79.2,
      "headroom_to_soft_cap": 50.0,
      "headroom_to_hard_cap": 250.0
    },
    "processor_info": {
      "logical_cpus": 8,
      "cpu_model": "z15",
      "total_cpu_capacity": 1200,
      "available_engines": 8
    },
    "thresholds": {
      "warning_level": 900,
      "critical_level": 1000,
      "emergency_level": 1100
    }
  },
```

```
  "batch_schedule": {
    "source": "Job Scheduler (Control-M)",
    "collection_time": "2025-01-15 14:30:00",
    "current_window": "Business Hours",
    "upcoming_jobs": [
      {
        "job_name": "EOD_PROCESSING",
        "scheduled_start": "18:00",
        "estimated_duration_min": 90,
        "estimated_msu": 150,
        "priority": "HIGH",
        "service_class": "BATCH",
        "can_defer": false,
        "reason": "End-of-day mandatory processing"
      },
      {
        "job_name": "DAILY_REPORTS",
        "scheduled_start": "19:00",
        "estimated_duration_min": 60,
        "estimated_msu": 70,
        "priority": "MEDIUM",
        "service_class": "REPORTS",
        "can_defer": true,
        "defer_until": "22:00",
        "reason": "Daily reports - can run later"
      }
    ],
    "summary": {
      "next_4_hours_estimated_msu": 220,
      "peak_window_start": "18:00",
      "deferrable_jobs_count": 1,
      "critical_jobs_count": 1
    }
  },
```

```
  "historical_context": {
    "source": "Historical Analysis (30-day average)",
    "typical_for_this_time": {
      "day_of_week": "Wednesday",
      "hour": 14,
      "typical_r4ha_baseline": 810,
      "typical_range_min": 750,
      "typical_range_max": 870,
      "spike_threshold": 910,
      "current_vs_typical": "+140 MSU (17% above normal)"
    },
    "recent_trends": {
      "past_24h_avg": 780,
      "past_week_avg": 795,
      "trend": "increasing",
      "volatility": "moderate"
    }
  },

```


```
  
  "business_context": {
    "month_end": false,
    "quarter_end": false,
    "year_end": false,
    "holiday": false,
    "special_event": false,
    "peak_season": false,
    "notes": "Normal business operations"
  }
}


```
