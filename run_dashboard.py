#!/usr/bin/env python3
"""
Dashboard Launcher
==================
Launches the Streamlit dashboard for the Convexity Sensitivity AI Agent.

Usage:
    python run_dashboard.py
    OR
    streamlit run dashboard/app.py
"""

import subprocess
import sys
import os

if __name__ == "__main__":
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard", "app.py")
    print("🚀 Launching Convexity Sensitivity AI Agent Dashboard...")
    print(f"   Dashboard: {dashboard_path}")
    print("   Press Ctrl+C to stop\n")
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", dashboard_path,
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ])
