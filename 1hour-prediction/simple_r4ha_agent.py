"""
Simple R4HA LangGraph Agent - Command Line Version
Clean, focused implementation that:
1. Reads JSON input file
2. Uses trained model to predict
3. LLM provides analysis

Usage:
    python simple_r4ha_agent.py --file input_normal.json
    python simple_r4ha_agent.py --file input_high.json --output report.json
    python simple_r4ha_agent.py --monitor --file input_normal.json --interval 15
"""

import os
import json
import argparse
import pandas as pd
import xgboost as xgb
import pickle
from datetime import datetime
from typing import TypedDict, Annotated
import operator
import time

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool

# ============================================================
# Configuration
# ============================================================

# Check for API key
if "OPENAI_API_KEY" not in os.environ:
    from dotenv import load_dotenv
    load_dotenv()
    
    if "OPENAI_API_KEY" not in os.environ:
        print("ERROR: OpenAI API key not found!")
        print("Set OPENAI_API_KEY environment variable or create .env file")
        exit(1)

# ============================================================
# Load Trained Model
# ============================================================

print("Loading R4HA prediction model...")

try:
    # Load XGBoost model
    model = xgb.XGBRegressor()
    model.load_model('r4ha_model_1h.json')
    
    # Load metadata
    with open('r4ha_model_1h_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    feature_columns = metadata['feature_columns']
    
    print(f"✓ Model loaded successfully ({len(feature_columns)} features)")

except FileNotFoundError as e:
    print(f"❌ ERROR: Model files not found!")
    print(f"   Make sure r4ha_model_1h.json and r4ha_model_1h_metadata.pkl exist")
    print(f"   Run r4ha_train_model_1h.py first to train the model")
    exit(1)

# ============================================================
# Prediction Function
# ============================================================

def predict_r4ha(features_dict, threshold=1000):
    """
    Predict R4HA 1 hour ahead and check alerts
    
    Args:
        features_dict: Dictionary with 13 feature values
        threshold: Alert threshold in MSU
    
    Returns:
        Dictionary with prediction and analysis
    """
    # Prepare features in correct order
    features_df = pd.DataFrame([features_dict])[feature_columns]
    
    # Make prediction
    predicted_r4ha = model.predict(features_df)[0]
    
    # Extract current values
    current_r4ha = features_dict['r4ha_msu']
    current_msu = features_dict['msu_current']
    r4ha_lag_1h = features_dict['r4ha_lag_1h']
    r4ha_lag_2h = features_dict['r4ha_lag_2h']
    
    # Calculate HISTORICAL trend (what has been happening)
    historical_change_1h = current_r4ha - r4ha_lag_1h  # Change in last hour
    historical_change_2h = current_r4ha - r4ha_lag_2h  # Change over 2 hours
    
    if historical_change_1h > 5:
        historical_trend = "INCREASING"
    elif historical_change_1h < -5:
        historical_trend = "DECREASING"
    else:
        historical_trend = "STABLE"
    
    # Calculate PREDICTED trend (what will happen)
    predicted_change = predicted_r4ha - current_r4ha
    
    if predicted_change > 5:
        predicted_trend = "INCREASING"
    elif predicted_change < -5:
        predicted_trend = "DECREASING"
    else:
        predicted_trend = "STABLE"
    
    # Check alerts
    current_alert = current_r4ha >= threshold
    future_alert = predicted_r4ha >= threshold
    
    # Generate alert message
    if current_alert and future_alert:
        alert_msg = f"🔴 CRITICAL: Current R4HA ({current_r4ha:.0f}) EXCEEDS threshold ({threshold})! Will remain HIGH ({predicted_r4ha:.0f}) in 1 hour."
    elif current_alert and not future_alert:
        alert_msg = f"🟡 WARNING: Current R4HA ({current_r4ha:.0f}) EXCEEDS threshold, but will drop to {predicted_r4ha:.0f} in 1 hour."
    elif not current_alert and future_alert:
        alert_msg = f"🟠 CAUTION: Current R4HA ({current_r4ha:.0f}) is OK, but will EXCEED threshold in 1 hour ({predicted_r4ha:.0f})."
    else:
        alert_msg = f"✅ OK: Current R4HA ({current_r4ha:.0f}) and predicted ({predicted_r4ha:.0f}) are both below threshold ({threshold})."
    
    return {
        "current_r4ha": round(current_r4ha, 2),
        "current_msu": round(current_msu, 2),
        "predicted_r4ha_1h": round(predicted_r4ha, 2),
        "predicted_change": round(predicted_change, 2),
        "predicted_trend": predicted_trend,
        "historical_trend": historical_trend,
        "historical_values": {
            "2_hours_ago": round(r4ha_lag_2h, 2),
            "1_hour_ago": round(r4ha_lag_1h, 2),
            "now": round(current_r4ha, 2)
        },
        "historical_change_1h": round(historical_change_1h, 2),
        "historical_change_2h": round(historical_change_2h, 2),
        "threshold": threshold,
        "current_alert": current_alert,
        "future_alert": future_alert,
        "alert_message": alert_msg,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

# ============================================================
# LangGraph Tool
# ============================================================

@tool
def get_r4ha_prediction(input_json: str) -> dict:
    """
    Get R4HA prediction for 1 hour ahead using the trained XGBoost model.
    
    Args:
        input_json: JSON string containing features and threshold.
                   Example: '{"features": {...}, "threshold": 1000}'
    
    Returns:
        Prediction results including current R4HA, predicted R4HA, trend, and alerts
    """
    try:
        # Parse input
        data = json.loads(input_json)
        features = data.get('features', data)
        threshold = data.get('threshold', 1000)
        
        # Make prediction
        result = predict_r4ha(features, threshold)
        
        return {
            "status": "success",
            "prediction": result
        }
    
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "Failed to get prediction. Check input format."
        }

tools = [get_r4ha_prediction]

# ============================================================
# Agent State
# ============================================================

class AgentState(TypedDict):
    """State that the agent maintains"""
    messages: Annotated[list[BaseMessage], operator.add]
    input_file: str
    prediction_result: dict

# ============================================================
# Initialize LLM
# ============================================================

llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# ============================================================
# Agent Nodes
# ============================================================

def agent_node(state: AgentState) -> dict:
    """Main agent node - decides what to do"""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Decide if agent should continue or end"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If LLM called a tool, continue to tool execution
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "continue"
    
    # Otherwise, we're done
    return "end"

# ============================================================
# Build LangGraph Workflow
# ============================================================

def create_r4ha_agent():
    """Create the LangGraph agent"""
    
    # Create graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END
        }
    )
    
    # Tools go back to agent
    workflow.add_edge("tools", "agent")
    
    # Compile
    return workflow.compile()


# Create the agent
agent = create_r4ha_agent()

# ============================================================
# System Prompt
# ============================================================

SYSTEM_PROMPT = """You are an IBM Z Mainframe Capacity Planning Expert specializing in R4HA (Rolling 4-Hour Average) analysis.

Your job:
1. Use the get_r4ha_prediction tool to analyze current mainframe metrics
2. Interpret the prediction results from the trained XGBoost model
3. Provide clear, actionable recommendations

CRITICAL: Pay close attention to trends!
- **historical_trend**: What HAS been happening (past 2 hours)
- **predicted_trend**: What WILL happen (next 1 hour)
- **historical_values**: Shows the actual progression (2h ago → 1h ago → now)

When analyzing R4HA predictions:
- Review historical_values to understand the ACTUAL trend direction
- If values are INCREASING (950 → 980 → 1020), say "increasing"
- If values are DECREASING (1020 → 980 → 950), say "decreasing"
- Compare historical trend vs predicted trend
- Check if thresholds are exceeded
- Assess the urgency level

Provide recommendations in this format:
1. **Current Situation**: Summarize current R4HA and system state
2. **Historical Trend**: What has been happening (use historical_values to be precise)
3. **Prediction**: What the model predicts for 1 hour ahead
4. **Trend Analysis**: Compare historical vs predicted trend
5. **Risk Assessment**: LOW/MEDIUM/HIGH/CRITICAL
6. **Immediate Actions**: What to do now (if anything)
7. **Explanation**: Why the prediction makes sense based on the metrics

Be precise with numbers. Don't confuse increasing with decreasing trends!"""

# ============================================================
# Main Functions
# ============================================================

def load_input_file(file_path):
    """Load metrics from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def run_r4ha_agent(input_file_path):
    """Run the R4HA agent with input from JSON file"""
    
    # Load input
    input_data = load_input_file(input_file_path)
    input_json_str = json.dumps(input_data)
    
    # Create initial message
    user_message = f"""Analyze the current mainframe R4HA situation using the provided metrics.

Input data from file: {input_file_path}
{input_json_str}

Please:
1. Use the get_r4ha_prediction tool to analyze these metrics
2. Provide a comprehensive R4HA analysis and recommendations
"""
    
    # Initialize state
    initial_state = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_message)
        ],
        "input_file": input_file_path,
        "prediction_result": {}
    }
    
    # Run agent
    result = agent.invoke(initial_state)
    
    # Get final answer
    final_message = result["messages"][-1]
    
    return final_message.content


def save_analysis(input_file, analysis, output_file=None):
    """Save analysis results to JSON file"""
    
    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'r4ha_analysis_{timestamp}.json'
    
    # Load original input
    input_data = load_input_file(input_file)
    
    # Create output
    output_data = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "input_file": input_file,
        "input_data": input_data,
        "analysis": analysis
    }
    
    # Save
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Analysis saved to: {output_file}")
    return output_file


def monitor_periodic(input_file, interval_minutes=15, max_iterations=None):
    """
    Run periodic monitoring
    
    Args:
        input_file: Path to JSON input file
        interval_minutes: How often to check
        max_iterations: Max number of checks (None = infinite)
    """
    
    print(f"\n{'='*70}")
    print(f"STARTING PERIODIC R4HA MONITORING")
    print(f"File: {input_file}")
    print(f"Interval: {interval_minutes} minutes")
    print(f"Press Ctrl+C to stop")
    print("="*70)
    
    iteration = 0
    
    try:
        while max_iterations is None or iteration < max_iterations:
            iteration += 1
            
            print(f"\n\n🕐 Check #{iteration} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("-" * 70)
            
            # Run analysis
            analysis = run_r4ha_agent(input_file)
            print(analysis)
            
            # Save to timestamped file
            save_analysis(input_file, analysis)
            
            # Wait for next interval
            if max_iterations is None or iteration < max_iterations:
                print(f"\n⏳ Waiting {interval_minutes} minutes until next check...")
                time.sleep(interval_minutes * 60)
    
    except KeyboardInterrupt:
        print(f"\n\n{'='*70}")
        print(f"MONITORING STOPPED (Total checks: {iteration})")
        print("="*70)


def main():
    """Main CLI"""
    
    parser = argparse.ArgumentParser(
        description='Simple R4HA LangGraph Agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single analysis
  python simple_r4ha_agent.py --file input_normal.json
  
  # Save output
  python simple_r4ha_agent.py --file input_high.json --output report.json
  
  # Periodic monitoring (every 15 minutes)
  python simple_r4ha_agent.py --monitor --file input_normal.json --interval 15
  
  # Test run (3 checks with 1 minute intervals)
  python simple_r4ha_agent.py --monitor --file input_normal.json --interval 1 --max-checks 3
        """
    )
    
    parser.add_argument('--file', '-f', required=True, help='Input JSON file')
    parser.add_argument('--output', '-o', help='Save analysis to JSON file')
    parser.add_argument('--monitor', '-m', action='store_true', help='Run periodic monitoring')
    parser.add_argument('--interval', '-i', type=int, default=15, help='Monitoring interval in minutes (default: 15)')
    parser.add_argument('--max-checks', type=int, help='Max number of checks (for testing)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.file):
        print(f"❌ ERROR: Input file '{args.file}' not found!")
        print("\nCreate input files using: python r4ha_input_examples.py")
        exit(1)
    
    # Periodic monitoring mode
    if args.monitor:
        monitor_periodic(args.file, args.interval, args.max_checks)
        return
    
    # Single analysis mode
    print("\n" + "="*70)
    print("🤖 R4HA AGENT ANALYZING...")
    print("="*70)
    
    analysis = run_r4ha_agent(args.file)
    
    print("\n" + "="*70)
    print("📊 R4HA ANALYSIS REPORT")
    print("="*70)
    print(analysis)
    print("="*70)
    
    # Save if requested
    if args.output:
        save_analysis(args.file, analysis, args.output)


if __name__ == "__main__":
    main()
