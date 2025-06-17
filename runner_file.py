#!/usr/bin/env python3
"""
Stress Test Dashboard Runner
This script helps avoid PyTorch import issues with Streamlit
"""

import os
import sys
import subprocess

def main():
    # Set environment variables to help with PyTorch compatibility
    os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_file = os.path.join(script_dir, "Stress_code_V2.py")
    
    # Check if the dashboard file exists
    if not os.path.exists(app_file):
        print(f"Error: {app_file} not found!")
        print("Please make sure the stress_test_dashboard.py file is in the same directory.")
        return 1
    
    try:
        # Run streamlit with the app
        cmd = [sys.executable, "-m", "streamlit", "run", app_file, "--server.fileWatcherType", "none"]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
        return 0

if __name__ == "__main__":
    exit(main())