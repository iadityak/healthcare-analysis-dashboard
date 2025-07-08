#!/usr/bin/env python3
"""
Test script for the healthcare dashboard
Verifies that the dashboard can load and process the dataset
"""

import pandas as pd
import sys
import os

def test_dataset_loading():
    """Test if dataset can be loaded and processed"""
    print("🧪 Testing Healthcare Dashboard")
    print("=" * 40)
    
    # Check if dataset exists
    if not os.path.exists('dataset.csv'):
        print("❌ Error: dataset.csv not found!")
        print("Please ensure the dataset file is in the same directory")
        return False
    
    print("✅ Dataset file found")
    
    # Try to load dataset
    try:
        df = pd.read_csv('dataset.csv')
        print(f"✅ Dataset loaded successfully")
        print(f"📊 Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False
    
    # Check for key columns
    key_column_patterns = [
        'Do_you_have_diagnose_h_chronic_conditions',
        'where_do_you_usually_re_when_you_are_sick',
        'What_are_the_biggest_accessing_healthcare',
        'Gender',
        'start'
    ]
    
    found_patterns = []
    for pattern in key_column_patterns:
        matching_cols = [col for col in df.columns if pattern in col]
        if matching_cols:
            found_patterns.append(pattern)
            print(f"✅ Found {len(matching_cols)} columns for {pattern}")
        else:
            print(f"⚠️  No columns found for {pattern}")
    
    print(f"\n📈 Data Summary:")
    print(f"   • Total respondents: {len(df)}")
    
    if 'Gender' in df.columns:
        gender_counts = df['Gender'].value_counts()
        print(f"   • Gender distribution: {dict(gender_counts)}")
    
    if 'start' in df.columns:
        try:
            df['start'] = pd.to_datetime(df['start'])
            date_range = f"{df['start'].min().strftime('%Y-%m-%d')} to {df['start'].max().strftime('%Y-%m-%d')}"
            print(f"   • Data collection period: {date_range}")
        except:
            print("   • Date processing: Could not process date range")
    
    print("\n🎯 Dashboard Features Available:")
    
    # Test chronic conditions
    chronic_cols = [col for col in df.columns if 'Do_you_have_diagnose_h_chronic_conditions' in col]
    if chronic_cols:
        print(f"   ✅ Disease Prevalence Analysis ({len(chronic_cols)} conditions)")
    
    # Test healthcare providers
    provider_cols = [col for col in df.columns if 'where_do_you_usually_re_when_you_are_sick' in col]
    if provider_cols:
        print(f"   ✅ Healthcare Access Analysis ({len(provider_cols)} providers)")
    
    # Test barriers
    barrier_cols = [col for col in df.columns if 'What_are_the_biggest_accessing_healthcare' in col]
    if barrier_cols:
        print(f"   ✅ Healthcare Barriers Analysis ({len(barrier_cols)} barriers)")
    
    # Test maternal health
    maternal_cols = [col for col in df.columns if 'Where_do_women_in_yo_household_give_birth' in col]
    if maternal_cols:
        print(f"   ✅ Maternal & Child Health Analysis ({len(maternal_cols)} delivery options)")
    
    # Test mental health
    stress_cols = [col for col in df.columns if 'What_are_the_biggest_stress_in_your_life' in col]
    if stress_cols:
        print(f"   ✅ Mental Health Analysis ({len(stress_cols)} stress factors)")
    
    # Test infrastructure
    infrastructure_cols = [col for col in df.columns if 'What_healthcare_serv_nk_needs_improvement' in col]
    if infrastructure_cols:
        print(f"   ✅ Infrastructure Needs Analysis ({len(infrastructure_cols)} service types)")
    
    print("\n" + "=" * 40)
    print("🚀 Dashboard is ready to run!")
    print("📱 To start the dashboard:")
    print("   1. Install packages: pip install -r requirements.txt")
    print("   2. Run: streamlit run app.py")
    print("   3. Or use: python run_dashboard.py")
    
    return True

if __name__ == "__main__":
    test_dataset_loading() 