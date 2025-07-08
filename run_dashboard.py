#!/usr/bin/env python3
"""
Healthcare Dashboard Launcher
Simple script to run the healthcare data visualization dashboard
"""

import sys
import subprocess
import os

def check_requirements():
    """Check if required packages are installed"""
    required_packages = ['streamlit', 'pandas', 'plotly', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing_packages)

def check_dataset():
    """Check if dataset.csv exists"""
    if not os.path.exists('dataset.csv'):
        print("⚠️  Warning: dataset.csv not found in current directory")
        print("Please ensure your dataset file is named 'dataset.csv' and placed in the same directory")
        return False
    return True

def run_dashboard():
    """Run the Streamlit dashboard"""
    print("🏥 Healthcare Data Visualization Dashboard")
    print("=" * 50)
    
    # Check requirements
    print("📦 Checking requirements...")
    check_requirements()
    
    # Check dataset
    print("📊 Checking dataset...")
    if not check_dataset():
        print("❌ Cannot proceed without dataset.csv")
        return
    
    print("✅ All checks passed!")
    print("🚀 Starting dashboard...")
    print("📱 Dashboard will open in your default browser")
    print("🛑 Press Ctrl+C to stop the dashboard")
    print("=" * 50)
    
    # Run Streamlit
    subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'app.py'])

if __name__ == "__main__":
    run_dashboard() 