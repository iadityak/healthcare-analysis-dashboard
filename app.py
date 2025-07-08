import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure the page
st.set_page_config(
    page_title="Healthcare Access & Disease Prevalence Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c5aa0;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .sidebar-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
        color: black;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try:
        df = pd.read_csv('dataset.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset file not found. Please ensure 'dataset.csv' is in the same directory as this app.")
        return None
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

def preprocess_data(df):
    """Preprocess the data for analysis"""
    if df is None:
        return None
    
    # Convert date columns
    date_columns = ['start', 'end', 'DOB', '_submission_time']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Clean numeric columns
    numeric_columns = ['Number_of_Household_members', 'Number_of_Children_Under_5_Years', 
                      'Number_of_Elderly_65_Years', 'How_many_meals_do_you_eat_per_day']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def get_chronic_conditions_data(df):
    """Extract chronic conditions data"""
    chronic_conditions = [
        'Do_you_have_diagnose_h_chronic_conditions/hypertension',
        'Do_you_have_diagnose_h_chronic_conditions/diabetes',
        'Do_you_have_diagnose_h_chronic_conditions/asthma',
        'Do_you_have_diagnose_h_chronic_conditions/arthritis',
        'Do_you_have_diagnose_h_chronic_conditions/tuberclosis',
        'Do_you_have_diagnose_h_chronic_conditions/kidney_disease',
        'Do_you_have_diagnose_h_chronic_conditions/heart_disease',
        'Do_you_have_diagnose_h_chronic_conditions/mental_health_conditions'
    ]
    
    condition_data = {}
    for condition in chronic_conditions:
        if condition in df.columns:
            condition_name = condition.split('/')[-1].replace('_', ' ').title()
            condition_data[condition_name] = df[condition].sum() if df[condition].dtype in ['int64', 'float64'] else df[condition].value_counts().get(1, 0)
    
    return condition_data

def get_healthcare_providers_data(df):
    """Extract healthcare provider preference data"""
    provider_columns = [
        'where_do_you_usually_re_when_you_are_sick/government_hospital_health_post',
        'where_do_you_usually_re_when_you_are_sick/private_clinic',
        'where_do_you_usually_re_when_you_are_sick/pharmacy',
        'where_do_you_usually_re_when_you_are_sick/traditional_healer',
        'where_do_you_usually_re_when_you_are_sick/home_remedies',
        'where_do_you_usually_re_when_you_are_sick/i_do_not_seek_healthcare'
    ]
    
    provider_data = {}
    for provider in provider_columns:
        if provider in df.columns:
            provider_name = provider.split('/')[-1].replace('_', ' ').title()
            provider_data[provider_name] = df[provider].sum() if df[provider].dtype in ['int64', 'float64'] else df[provider].value_counts().get(1, 0)
    
    return provider_data

def get_barriers_data(df):
    """Extract healthcare barriers data"""
    barrier_columns = [
        'What_are_the_biggest_accessing_healthcare/distance',
        'What_are_the_biggest_accessing_healthcare/cost',
        'What_are_the_biggest_accessing_healthcare/poor_service_quality',
        'What_are_the_biggest_accessing_healthcare/cultural_beliefs',
        'What_are_the_biggest_accessing_healthcare/lack_of_transport'
    ]
    
    barrier_data = {}
    for barrier in barrier_columns:
        if barrier in df.columns:
            barrier_name = barrier.split('/')[-1].replace('_', ' ').title()
            barrier_data[barrier_name] = df[barrier].sum() if df[barrier].dtype in ['int64', 'float64'] else df[barrier].value_counts().get(1, 0)
    
    return barrier_data



def clean_village_name(village):
    """Clean and standardize village names"""
    if pd.isna(village) or village == '':
        return 'Unknown'
    
    # Convert to string and strip whitespace
    village = str(village).strip()
    
    # Common cleaning patterns
    village = village.replace(' ai village', '').replace(' village', '')
    village = village.replace(',', '').replace('  ', ' ')
    
    # Standardize common variations
    village_mappings = {
        'choglamsar': 'Choglamsar',
        'leh': 'Leh',
        'thiksey': 'Thiksey',
        'chuchot': 'Chuchot',
        'basgo': 'Basgo',
        'igoo': 'Igoo',
        'sakti': 'Sakti',
        'shey': 'Shey',
        'kargil': 'Kargil',
        'domkhar': 'Domkhar',
        'zanskar': 'Zanskar',
        'nubra': 'Nubra',
        'wakha': 'Wakha',
        'skampari': 'Skampari',
        'saboo': 'Saboo',
        'phey': 'Phey',
        'markha': 'Markha',
        'wanla': 'Wanla',
        'timosgum': 'Timosgum',
        'tukla': 'Tukla',
        'mulbekh': 'Mulbekh',
        'lamayuru': 'Lamayuru',
        'chemday': 'Chemday',
        'chemray': 'Chemray',
        'diskit': 'Diskit',
        'sumoor': 'Sumoor',
        'tia': 'Tia',
        'hanu': 'Hanu'
    }
    
    # Check for mapping
    village_lower = village.lower()
    for key, value in village_mappings.items():
        if key in village_lower:
            return value
    
    # Title case for unmapped villages
    return village.title()

def get_village_disease_data(df):
    """Get disease prevalence by village"""
    if 'Address' not in df.columns:
        return pd.DataFrame()
    
    # Clean village names
    df['Village'] = df['Address'].apply(clean_village_name)
    
    # Get chronic conditions
    chronic_columns = [
        'Do_you_have_diagnose_h_chronic_conditions/hypertension',
        'Do_you_have_diagnose_h_chronic_conditions/diabetes',
        'Do_you_have_diagnose_h_chronic_conditions/asthma',
        'Do_you_have_diagnose_h_chronic_conditions/arthritis',
        'Do_you_have_diagnose_h_chronic_conditions/heart_disease',
        'Do_you_have_diagnose_h_chronic_conditions/mental_health_conditions'
    ]
    
    village_disease_data = []
    for village in df['Village'].unique():
        if village == 'Unknown':
            continue
        village_data = df[df['Village'] == village]
        
        for condition in chronic_columns:
            if condition in df.columns:
                condition_name = condition.split('/')[-1].replace('_', ' ').title()
                count = village_data[condition].sum() if village_data[condition].dtype in ['int64', 'float64'] else village_data[condition].value_counts().get(1, 0)
                total = len(village_data)
                
                village_disease_data.append({
                    'Village': village,
                    'Condition': condition_name,
                    'Count': count,
                    'Total': total,
                    'Prevalence': (count / total * 100) if total > 0 else 0
                })
    
    return pd.DataFrame(village_disease_data)

def get_village_distance_data(df):
    """Get healthcare access distance by village"""
    if 'Address' not in df.columns:
        return pd.DataFrame()
    
    df['Village'] = df['Address'].apply(clean_village_name)
    
    if 'How_far_is_the_neare_healthcare_facility' not in df.columns:
        return pd.DataFrame()
    
    village_distance_data = []
    for village in df['Village'].unique():
        if village == 'Unknown':
            continue
        village_data = df[df['Village'] == village]
        
        # Get distance distribution
        distances = village_data['How_far_is_the_neare_healthcare_facility'].value_counts()
        total = len(village_data)
        
        for distance, count in distances.items():
            if pd.notna(distance):
                village_distance_data.append({
                    'Village': village,
                    'Distance': distance,
                    'Count': count,
                    'Total': total,
                    'Percentage': (count / total * 100) if total > 0 else 0
                })
    
    return pd.DataFrame(village_distance_data)

def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Naropa Healthcare Survey Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    df = preprocess_data(df)
    
    # Sidebar
    st.sidebar.markdown('<div class="sidebar-info"><h3>📊 Dashboard Overview</h3><p>This dashboard provides comprehensive insights into healthcare access, disease prevalence, and service utilization patterns across high-altitude communities.</p></div>', unsafe_allow_html=True)
    
    # Data summary
    st.sidebar.markdown("### 📈 Data Summary")
    st.sidebar.metric("Total Respondents", len(df))
    st.sidebar.metric("Data Collection Period", f"{df['start'].min().strftime('%Y-%m-%d')} to {df['start'].max().strftime('%Y-%m-%d')}")
    
    # Filter options
    st.sidebar.markdown("### 🔍 Filter Options")
    
    # Gender filter
    if 'Gender' in df.columns:
        gender_options = ['All'] + list(df['Gender'].dropna().unique())
        selected_gender = st.sidebar.selectbox("Filter by Gender", gender_options)
        if selected_gender != 'All':
            df = df[df['Gender'] == selected_gender]
    
    # Age group filter (based on age ranges)
    age_options = ['All', 'Under 18', '18-59', '60+']
    selected_age = st.sidebar.selectbox("Filter by Age Group", age_options)
    
    # Income filter
    if 'Total_number_of_hous_ncome_monthly_Approx' in df.columns:
        income_options = ['All'] + list(df['Total_number_of_hous_ncome_monthly_Approx'].dropna().unique())
        selected_income = st.sidebar.selectbox("Filter by Income Level", income_options)
        if selected_income != 'All':
            df = df[df['Total_number_of_hous_ncome_monthly_Approx'] == selected_income]
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🦠 Disease Prevalence", 
        "🏥 Healthcare Access", 
        "🎯 Health Seeking Behavior", 
        "👶 Maternal & Child Health", 
        "🧠 Mental Health & Substance Use",
        "🏗️ Infrastructure Needs",
        "🏘️ Village-Level Analysis"
    ])
    
    with tab1:
        st.markdown('<h2 class="section-header">Disease Prevalence Analysis</h2>', unsafe_allow_html=True)
        
        # Chronic conditions analysis
        col1, col2 = st.columns(2)
        
        with col1:
            chronic_conditions = get_chronic_conditions_data(df)
            if chronic_conditions:
                fig = px.bar(
                    x=list(chronic_conditions.keys()),
                    y=list(chronic_conditions.values()),
                    title="Chronic Conditions Prevalence",
                    labels={'x': 'Condition', 'y': 'Number of Cases'},
                    color=list(chronic_conditions.values()),
                    color_continuous_scale='Reds'
                )
                fig.update_layout(xaxis_tickangle=-45, height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Recent health problems in past 6 months
            recent_health_cols = [col for col in df.columns if 'In_the_past_6_months_any_of_the_following' in col and '/' in col]
            if recent_health_cols:
                recent_health_data = {}
                for col in recent_health_cols:
                    condition_name = col.split('/')[-1].replace('_', ' ').title()
                    recent_health_data[condition_name] = df[col].sum() if df[col].dtype in ['int64', 'float64'] else df[col].value_counts().get(1, 0)
                
                # Filter out zero values and invalid entries
                recent_health_data = {k: v for k, v in recent_health_data.items() if v > 0 and k.lower() not in ['in the past 6 months any of the following', 'nan', '']}
                
                if recent_health_data:
                    fig = px.pie(
                        values=list(recent_health_data.values()),
                        names=list(recent_health_data.keys()),
                        title="Recent Health Problems (Past 6 Months)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Community health issues
        st.markdown("### Community Health Issues")
        community_health_cols = [col for col in df.columns if 'What_are_the_most_co_es_in_your_community' in col and '/' in col]
        if community_health_cols:
            community_data = {}
            for col in community_health_cols:
                issue_name = col.split('/')[-1].replace('_', ' ').title()
                community_data[issue_name] = df[col].sum() if df[col].dtype in ['int64', 'float64'] else df[col].value_counts().get(1, 0)
            
            # Filter out zero values and invalid entries
            community_data = {k: v for k, v in community_data.items() if v > 0 and k.lower() not in ['what are the most co es in your community', 'nan', '']}
            
            if community_data:
                fig = px.bar(
                    x=list(community_data.values()),
                    y=list(community_data.keys()),
                    orientation='h',
                    title="Most Common Community Health Issues",
                    labels={'x': 'Number of Reports', 'y': 'Health Issue'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown('<h2 class="section-header">Healthcare Access Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Healthcare provider preferences
            provider_data = get_healthcare_providers_data(df)
            if provider_data:
                fig = px.bar(
                    x=list(provider_data.keys()),
                    y=list(provider_data.values()),
                    title="Healthcare Provider Preferences",
                    labels={'x': 'Provider Type', 'y': 'Number of Users'},
                    color=list(provider_data.values()),
                    color_continuous_scale='Blues'
                )
                fig.update_layout(xaxis_tickangle=-45, height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Distance to healthcare facility
            if 'How_far_is_the_neare_healthcare_facility' in df.columns:
                distance_data = df['How_far_is_the_neare_healthcare_facility'].value_counts()
                fig = px.pie(
                    values=distance_data.values,
                    names=distance_data.index,
                    title="Distance to Nearest Healthcare Facility"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Healthcare barriers
        st.markdown("### Healthcare Access Barriers")
        barriers_data = get_barriers_data(df)
        if barriers_data:
            fig = px.bar(
                x=list(barriers_data.keys()),
                y=list(barriers_data.values()),
                title="Major Barriers to Healthcare Access",
                labels={'x': 'Barrier Type', 'y': 'Number of Reports'},
                color=list(barriers_data.values()),
                color_continuous_scale='Reds'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown('<h2 class="section-header">Health Seeking Behavior</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Reasons for choosing healthcare provider
            reason_cols = [col for col in df.columns if 'What_is_the_primary_choosing_this_option' in col and '/' in col]
            if reason_cols:
                reason_data = {}
                for col in reason_cols:
                    reason_name = col.split('/')[-1].replace('_', ' ').title()
                    reason_data[reason_name] = df[col].sum() if df[col].dtype in ['int64', 'float64'] else df[col].value_counts().get(1, 0)
                
                # Filter out zero values and invalid entries
                reason_data = {k: v for k, v in reason_data.items() if v > 0 and k.lower() not in ['what is the primary choosing this option', 'nan', '']}
                
                if reason_data:
                    fig = px.bar(
                        x=list(reason_data.keys()),
                        y=list(reason_data.values()),
                        title="Reasons for Healthcare Provider Choice",
                        labels={'x': 'Reason', 'y': 'Number of Responses'}
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Preventive services utilization
            preventive_cols = [col for col in df.columns if 'If_Yes_What_type' in col and '/' in col]
            if preventive_cols:
                preventive_data = {}
                for col in preventive_cols:
                    service_name = col.split('/')[-1].replace('_', ' ').title()
                    preventive_data[service_name] = df[col].sum() if df[col].dtype in ['int64', 'float64'] else df[col].value_counts().get(1, 0)
                
                # Filter out zero values and invalid entries
                preventive_data = {k: v for k, v in preventive_data.items() if v > 0 and k.lower() not in ['if yes what type', 'nan', '']}
                
                if preventive_data:
                    fig = px.pie(
                        values=list(preventive_data.values()),
                        names=list(preventive_data.keys()),
                        title="Preventive Services Utilization"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown('<h2 class="section-header">Maternal & Child Health</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Place of delivery
            delivery_cols = [col for col in df.columns if 'Where_do_women_in_yo_household_give_birth' in col and '/' in col]
            if delivery_cols:
                delivery_data = {}
                for col in delivery_cols:
                    place_name = col.split('/')[-1].replace('_', ' ').title()
                    delivery_data[place_name] = df[col].sum() if df[col].dtype in ['int64', 'float64'] else df[col].value_counts().get(1, 0)
                
                # Filter out zero values and invalid entries
                delivery_data = {k: v for k, v in delivery_data.items() if v > 0 and k.lower() not in ['where do women in yo household give birth', 'nan', '']}
                
                if delivery_data:
                    fig = px.pie(
                        values=list(delivery_data.values()),
                        names=list(delivery_data.keys()),
                        title="Place of Delivery"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Reasons for home delivery
            home_delivery_cols = [col for col in df.columns if 'What_are_the_main_reason_for_homebirth' in col and '/' in col]
            if home_delivery_cols:
                home_delivery_data = {}
                for col in home_delivery_cols:
                    reason_name = col.split('/')[-1].replace('_', ' ').title()
                    home_delivery_data[reason_name] = df[col].sum() if df[col].dtype in ['int64', 'float64'] else df[col].value_counts().get(1, 0)
                
                # Filter out zero values and invalid entries
                home_delivery_data = {k: v for k, v in home_delivery_data.items() if v > 0 and k.lower() not in ['what are the main reason for homebirth', 'nan', '']}
                
                if home_delivery_data:
                    fig = px.bar(
                        x=list(home_delivery_data.keys()),
                        y=list(home_delivery_data.values()),
                        title="Reasons for Home Delivery",
                        labels={'x': 'Reason', 'y': 'Number of Cases'}
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Child nutrition
        st.markdown("### Child Nutrition")
        nutrition_cols = [col for col in df.columns if 'What_is_their_main_source_of_nutrition' in col and '/' in col]
        if nutrition_cols:
            nutrition_data = {}
            for col in nutrition_cols:
                nutrition_name = col.split('/')[-1].replace('_', ' ').title()
                nutrition_data[nutrition_name] = df[col].sum() if df[col].dtype in ['int64', 'float64'] else df[col].value_counts().get(1, 0)
            
            # Filter out zero values and invalid entries
            nutrition_data = {k: v for k, v in nutrition_data.items() if v > 0 and k.lower() not in ['what is their main source of nutrition', 'nan', '']}
            
            if nutrition_data:
                fig = px.bar(
                    x=list(nutrition_data.keys()),
                    y=list(nutrition_data.values()),
                    title="Main Sources of Child Nutrition",
                    labels={'x': 'Nutrition Source', 'y': 'Number of Cases'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.markdown('<h2 class="section-header">Mental Health & Substance Use</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Stress sources
            stress_cols = [col for col in df.columns if 'What_are_the_biggest_stress_in_your_life' in col and '/' in col]
            if stress_cols:
                stress_data = {}
                for col in stress_cols:
                    stress_name = col.split('/')[-1].replace('_', ' ').title()
                    stress_data[stress_name] = df[col].sum() if df[col].dtype in ['int64', 'float64'] else df[col].value_counts().get(1, 0)
                
                # Filter out zero values and invalid entries
                stress_data = {k: v for k, v in stress_data.items() if v > 0 and k.lower() not in ['what are the biggest stress in your life', 'nan', '']}
                
                if stress_data:
                    fig = px.bar(
                        x=list(stress_data.keys()),
                        y=list(stress_data.values()),
                        title="Major Sources of Stress",
                        labels={'x': 'Stress Source', 'y': 'Number of Cases'}
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Substance use
            substance_cols = ['Tobacco', 'Alcohol', 'Chewing_betel_nut_Pan_Gutka_etc']
            substance_data = {}
            for col in substance_cols:
                if col in df.columns:
                    substance_name = col.replace('_', ' ').title()
                    substance_data[substance_name] = df[col].value_counts().get('yes', 0) + df[col].value_counts().get('yes__frequently', 0) + df[col].value_counts().get('yes__occasionally', 0)
            
            if substance_data:
                fig = px.bar(
                    x=list(substance_data.keys()),
                    y=list(substance_data.values()),
                    title="Substance Use Patterns",
                    labels={'x': 'Substance', 'y': 'Number of Users'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab6:
        st.markdown('<h2 class="section-header">Infrastructure & Service Needs</h2>', unsafe_allow_html=True)
        
        # Healthcare service improvement needs
        improvement_cols = [col for col in df.columns if 'What_healthcare_serv_nk_needs_improvement' in col and '/' in col]
        if improvement_cols:
            improvement_data = {}
            for col in improvement_cols:
                service_name = col.split('/')[-1].replace('_', ' ').title()
                improvement_data[service_name] = df[col].sum() if df[col].dtype in ['int64', 'float64'] else df[col].value_counts().get(1, 0)
            
            # Filter out zero values and invalid entries
            improvement_data = {k: v for k, v in improvement_data.items() if v > 0 and k.lower() not in ['what healthcare serv nk needs improvement', 'nan', '']}
            
            if improvement_data:
                fig = px.bar(
                    x=list(improvement_data.keys()),
                    y=list(improvement_data.values()),
                    title="Healthcare Services Needing Improvement",
                    labels={'x': 'Service Type', 'y': 'Number of Requests'},
                    color=list(improvement_data.values()),
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        
        # Water and sanitation
        col1, col2 = st.columns(2)
        
        with col1:
            # Water sources
            water_cols = [col for col in df.columns if 'What_is_your_primary_rinking_water_source' in col and '/' in col]
            if water_cols:
                water_data = {}
                for col in water_cols:
                    water_name = col.split('/')[-1].replace('_', ' ').title()
                    water_data[water_name] = df[col].sum() if df[col].dtype in ['int64', 'float64'] else df[col].value_counts().get(1, 0)
                
                # Filter out zero values and invalid entries
                water_data = {k: v for k, v in water_data.items() if v > 0 and k.lower() not in ['what is your primary rinking water source', 'nan', '']}
                
                if water_data:
                    fig = px.pie(
                        values=list(water_data.values()),
                        names=list(water_data.keys()),
                        title="Primary Water Sources"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Toilet facilities
            toilet_cols = [col for col in df.columns if 'What_type_of_toilet_facility_do_you_use' in col and '/' in col]
            if toilet_cols:
                toilet_data = {}
                for col in toilet_cols:
                    toilet_name = col.split('/')[-1].replace('_', ' ').title()
                    toilet_data[toilet_name] = df[col].sum() if df[col].dtype in ['int64', 'float64'] else df[col].value_counts().get(1, 0)
                
                # Filter out zero values and invalid entries
                toilet_data = {k: v for k, v in toilet_data.items() if v > 0 and k.lower() not in ['what type of toilet facility do you use', 'nan', '']}
                
                if toilet_data:
                    fig = px.pie(
                        values=list(toilet_data.values()),
                        names=list(toilet_data.keys()),
                        title="Toilet Facility Types"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab7:
        st.markdown('<h2 class="section-header">Village-Level Healthcare Analysis</h2>', unsafe_allow_html=True)
        
        # Village selector
        if 'Address' in df.columns:
            df['Village'] = df['Address'].apply(clean_village_name)
            villages = sorted([v for v in df['Village'].unique() if v != 'Unknown'])
            selected_villages = st.multiselect(
                "Select villages to analyze (leave empty for all villages):",
                villages,
                default=villages[:10] if len(villages) > 10 else villages
            )
            
            if not selected_villages:
                selected_villages = villages
            
            # Filter data for selected villages
            village_df = df[df['Village'].isin(selected_villages)]
            
            # 1. Disease Prevalence by Village
            st.markdown("### 🦠 Disease Prevalence by Village")
            village_disease_data = get_village_disease_data(village_df)
            
            if not village_disease_data.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Heatmap of disease prevalence
                    pivot_disease = village_disease_data.pivot(index='Village', columns='Condition', values='Prevalence')
                    fig = px.imshow(
                        pivot_disease,
                        title="Disease Prevalence Heatmap by Village (%)",
                        labels={'color': 'Prevalence (%)'},
                        aspect="auto",
                        color_continuous_scale="Reds"
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Top villages by specific condition
                    condition_choice = st.selectbox(
                        "Select condition to analyze:",
                        village_disease_data['Condition'].unique()
                    )
                    
                    condition_data = village_disease_data[village_disease_data['Condition'] == condition_choice]
                    condition_data = condition_data.sort_values('Prevalence', ascending=False).head(10)
                    
                    fig = px.bar(
                        condition_data,
                        x='Prevalence',
                        y='Village',
                        title=f"Top Villages by {condition_choice} Prevalence",
                        labels={'Prevalence': 'Prevalence (%)', 'Village': 'Village'},
                        orientation='h'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # 2. Healthcare Access Distance by Village
            st.markdown("### 🏥 Healthcare Access Distance by Village")
            village_distance_data = get_village_distance_data(village_df)
            
            if not village_distance_data.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Average distance by village
                    distance_mapping = {
                        'Less than 1 km': 0.5,
                        '1-5 km': 3,
                        '5-10 km': 7.5,
                        'More than 10 km': 15
                    }
                    
                    village_avg_distance = []
                    for village in village_distance_data['Village'].unique():
                        village_data = village_distance_data[village_distance_data['Village'] == village]
                        weighted_distance = 0
                        total_responses = 0
                        
                        for _, row in village_data.iterrows():
                            if row['Distance'] in distance_mapping:
                                weighted_distance += distance_mapping[row['Distance']] * row['Count']
                                total_responses += row['Count']
                        
                        if total_responses > 0:
                            avg_distance = weighted_distance / total_responses
                            village_avg_distance.append({
                                'Village': village,
                                'Average_Distance': avg_distance,
                                'Total_Responses': total_responses
                            })
                    
                    if village_avg_distance:
                        distance_df = pd.DataFrame(village_avg_distance)
                        distance_df = distance_df.sort_values('Average_Distance', ascending=False)
                        
                        fig = px.bar(
                            distance_df,
                            x='Average_Distance',
                            y='Village',
                            title="Average Distance to Healthcare Facility by Village",
                            labels={'Average_Distance': 'Average Distance (km)', 'Village': 'Village'},
                            orientation='h',
                            color='Average_Distance',
                            color_continuous_scale='Reds'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Distance distribution
                    fig = px.sunburst(
                        village_distance_data,
                        path=['Distance', 'Village'],
                        values='Count',
                        title="Healthcare Distance Distribution by Village"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # 3. Healthcare Provider Preferences by Village
            st.markdown("### 🎯 Healthcare Provider Preferences by Village")
            
            provider_columns = [
                'where_do_you_usually_re_when_you_are_sick/government_hospital_health_post',
                'where_do_you_usually_re_when_you_are_sick/private_clinic',
                'where_do_you_usually_re_when_you_are_sick/pharmacy',
                'where_do_you_usually_re_when_you_are_sick/traditional_healer',
                'where_do_you_usually_re_when_you_are_sick/home_remedies'
            ]
            
            village_provider_data = []
            for village in selected_villages:
                village_data = village_df[village_df['Village'] == village]
                
                for provider in provider_columns:
                    if provider in village_df.columns:
                        provider_name = provider.split('/')[-1].replace('_', ' ').title()
                        count = village_data[provider].sum() if village_data[provider].dtype in ['int64', 'float64'] else village_data[provider].value_counts().get(1, 0)
                        total = len(village_data)
                        
                        village_provider_data.append({
                            'Village': village,
                            'Provider': provider_name,
                            'Count': count,
                            'Total': total,
                            'Percentage': (count / total * 100) if total > 0 else 0
                        })
            
            if village_provider_data:
                provider_df = pd.DataFrame(village_provider_data)
                
                fig = px.bar(
                    provider_df,
                    x='Village',
                    y='Percentage',
                    color='Provider',
                    title="Healthcare Provider Preferences by Village",
                    labels={'Percentage': 'Usage (%)', 'Village': 'Village'},
                    barmode='group'
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            # 4. Water Sources and Sanitation by Village
            st.markdown("### 💧 Water Sources and Sanitation by Village")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Water sources by village
                water_columns = [
                    'What_is_your_primary_rinking_water_source/piped_water',
                    'What_is_your_primary_rinking_water_source/well',
                    'What_is_your_primary_rinking_water_source/river_lake',
                    'What_is_your_primary_rinking_water_source/spring_water__chumik'
                ]
                
                village_water_data = []
                for village in selected_villages:
                    village_data = village_df[village_df['Village'] == village]
                    
                    for water_source in water_columns:
                        if water_source in village_df.columns:
                            source_name = water_source.split('/')[-1].replace('_', ' ').title()
                            count = village_data[water_source].sum() if village_data[water_source].dtype in ['int64', 'float64'] else village_data[water_source].value_counts().get(1, 0)
                            total = len(village_data)
                            
                            village_water_data.append({
                                'Village': village,
                                'Water_Source': source_name,
                                'Count': count,
                                'Total': total,
                                'Percentage': (count / total * 100) if total > 0 else 0
                            })
                
                if village_water_data:
                    water_df = pd.DataFrame(village_water_data)
                    
                    fig = px.bar(
                        water_df,
                        x='Village',
                        y='Percentage',
                        color='Water_Source',
                        title="Primary Water Sources by Village",
                        labels={'Percentage': 'Usage (%)', 'Village': 'Village'},
                        barmode='stack'
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Toilet facilities by village
                toilet_columns = [
                    'What_type_of_toilet_facility_do_you_use/open_defecation',
                    'What_type_of_toilet_facility_do_you_use/pit_latrine',
                    'What_type_of_toilet_facility_do_you_use/flush_toilet'
                ]
                
                village_toilet_data = []
                for village in selected_villages:
                    village_data = village_df[village_df['Village'] == village]
                    
                    for toilet_type in toilet_columns:
                        if toilet_type in village_df.columns:
                            toilet_name = toilet_type.split('/')[-1].replace('_', ' ').title()
                            count = village_data[toilet_type].sum() if village_data[toilet_type].dtype in ['int64', 'float64'] else village_data[toilet_type].value_counts().get(1, 0)
                            total = len(village_data)
                            
                            village_toilet_data.append({
                                'Village': village,
                                'Toilet_Type': toilet_name,
                                'Count': count,
                                'Total': total,
                                'Percentage': (count / total * 100) if total > 0 else 0
                            })
                
                if village_toilet_data:
                    toilet_df = pd.DataFrame(village_toilet_data)
                    
                    fig = px.bar(
                        toilet_df,
                        x='Village',
                        y='Percentage',
                        color='Toilet_Type',
                        title="Toilet Facilities by Village",
                        labels={'Percentage': 'Usage (%)', 'Village': 'Village'},
                        barmode='stack'
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
            
            # 5. Mental Health and Substance Use by Village
            st.markdown("### 🧠 Mental Health and Substance Use by Village")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Stress levels by village
                stress_columns = [
                    'What_are_the_biggest_stress_in_your_life/financial_problems',
                    'What_are_the_biggest_stress_in_your_life/health_issues',
                    'What_are_the_biggest_stress_in_your_life/family_conflict',
                    'What_are_the_biggest_stress_in_your_life/work_related_stress'
                ]
                
                village_stress_data = []
                for village in selected_villages:
                    village_data = village_df[village_df['Village'] == village]
                    
                    for stress_type in stress_columns:
                        if stress_type in village_df.columns and '/' in stress_type:
                            stress_name = stress_type.split('/')[-1].replace('_', ' ').title()
                            count = village_data[stress_type].sum() if village_data[stress_type].dtype in ['int64', 'float64'] else village_data[stress_type].value_counts().get(1, 0)
                            total = len(village_data)
                            
                            village_stress_data.append({
                                'Village': village,
                                'Stress_Type': stress_name,
                                'Count': count,
                                'Total': total,
                                'Percentage': (count / total * 100) if total > 0 else 0
                            })
                
                if village_stress_data:
                    stress_df = pd.DataFrame(village_stress_data)
                    
                    fig = px.bar(
                        stress_df,
                        x='Village',
                        y='Percentage',
                        color='Stress_Type',
                        title="Major Stress Factors by Village",
                        labels={'Percentage': 'Prevalence (%)', 'Village': 'Village'},
                        barmode='group'
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Substance use by village
                substance_columns = ['Tobacco', 'Alcohol', 'Chewing_betel_nut_Pan_Gutka_etc']
                
                village_substance_data = []
                for village in selected_villages:
                    village_data = village_df[village_df['Village'] == village]
                    
                    for substance in substance_columns:
                        if substance in village_df.columns:
                            substance_name = substance.replace('_', ' ').title()
                            # Count various forms of "yes" responses
                            yes_count = 0
                            if village_data[substance].dtype == 'object':
                                yes_count = village_data[substance].str.contains('yes', case=False, na=False).sum()
                            else:
                                yes_count = village_data[substance].sum()
                            
                            total = len(village_data)
                            
                            village_substance_data.append({
                                'Village': village,
                                'Substance': substance_name,
                                'Count': yes_count,
                                'Total': total,
                                'Percentage': (yes_count / total * 100) if total > 0 else 0
                            })
                
                if village_substance_data:
                    substance_df = pd.DataFrame(village_substance_data)
                    
                    fig = px.bar(
                        substance_df,
                        x='Village',
                        y='Percentage',
                        color='Substance',
                        title="Substance Use Patterns by Village",
                        labels={'Percentage': 'Usage (%)', 'Village': 'Village'},
                        barmode='group'
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
            
            # 6. Healthcare Barriers by Village
            st.markdown("### 🚧 Healthcare Barriers by Village")
            
            barrier_columns = [
                'What_are_the_biggest_accessing_healthcare/distance',
                'What_are_the_biggest_accessing_healthcare/cost',
                'What_are_the_biggest_accessing_healthcare/poor_service_quality',
                'What_are_the_biggest_accessing_healthcare/cultural_beliefs',
                'What_are_the_biggest_accessing_healthcare/lack_of_transport'
            ]
            
            village_barrier_data = []
            for village in selected_villages:
                village_data = village_df[village_df['Village'] == village]
                
                for barrier in barrier_columns:
                    if barrier in village_df.columns and '/' in barrier:
                        barrier_name = barrier.split('/')[-1].replace('_', ' ').title()
                        count = village_data[barrier].sum() if village_data[barrier].dtype in ['int64', 'float64'] else village_data[barrier].value_counts().get(1, 0)
                        total = len(village_data)
                        
                        village_barrier_data.append({
                            'Village': village,
                            'Barrier': barrier_name,
                            'Count': count,
                            'Total': total,
                            'Percentage': (count / total * 100) if total > 0 else 0
                        })
            
            if village_barrier_data:
                barrier_df = pd.DataFrame(village_barrier_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(
                        barrier_df,
                        x='Village',
                        y='Percentage',
                        color='Barrier',
                        title="Healthcare Access Barriers by Village",
                        labels={'Percentage': 'Reports (%)', 'Village': 'Village'},
                        barmode='group'
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Heatmap of barriers
                    pivot_barriers = barrier_df.pivot(index='Village', columns='Barrier', values='Percentage')
                    fig = px.imshow(
                        pivot_barriers,
                        title="Healthcare Barriers Heatmap by Village (%)",
                        labels={'color': 'Reports (%)'},
                        aspect="auto",
                        color_continuous_scale="Oranges"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # 7. Infrastructure Needs by Village
            st.markdown("### 🏗️ Infrastructure Needs by Village")
            
            infrastructure_columns = [
                'What_healthcare_serv_nk_needs_improvement/more_doctors_nurses',
                'What_healthcare_serv_nk_needs_improvement/better_facilities',
                'What_healthcare_serv_nk_needs_improvement/more_medicines',
                'What_healthcare_serv_nk_needs_improvement/health_education_programs',
                'What_healthcare_serv_nk_needs_improvement/emergency_services'
            ]
            
            village_infrastructure_data = []
            for village in selected_villages:
                village_data = village_df[village_df['Village'] == village]
                
                for infrastructure in infrastructure_columns:
                    if infrastructure in village_df.columns and '/' in infrastructure:
                        infra_name = infrastructure.split('/')[-1].replace('_', ' ').title()
                        count = village_data[infrastructure].sum() if village_data[infrastructure].dtype in ['int64', 'float64'] else village_data[infrastructure].value_counts().get(1, 0)
                        total = len(village_data)
                        
                        village_infrastructure_data.append({
                            'Village': village,
                            'Infrastructure_Need': infra_name,
                            'Count': count,
                            'Total': total,
                            'Percentage': (count / total * 100) if total > 0 else 0
                        })
            
            if village_infrastructure_data:
                infra_df = pd.DataFrame(village_infrastructure_data)
                
                fig = px.bar(
                    infra_df,
                    x='Village',
                    y='Percentage',
                    color='Infrastructure_Need',
                    title="Infrastructure Needs by Village",
                    labels={'Percentage': 'Demand (%)', 'Village': 'Village'},
                    barmode='group'
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            # 8. Summary Statistics
            st.markdown("### 📊 Village Summary Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Villages Analyzed", len(selected_villages))
            
            with col2:
                st.metric("Total Respondents", len(village_df))
            
            with col3:
                avg_respondents = len(village_df) / len(selected_villages) if selected_villages else 0
                st.metric("Average Respondents per Village", f"{avg_respondents:.1f}")
            
            # Village comparison table
            if len(selected_villages) > 1:
                st.markdown("### 📋 Village Comparison Summary")
                
                village_summary = []
                for village in selected_villages:
                    village_data = village_df[village_df['Village'] == village]
                    
                    # Calculate key metrics
                    total_respondents = len(village_data)
                    
                    # Average distance to healthcare
                    avg_distance = "N/A"
                    if 'How_far_is_the_neare_healthcare_facility' in village_data.columns:
                        distance_mode = village_data['How_far_is_the_neare_healthcare_facility'].mode()
                        if len(distance_mode) > 0:
                            avg_distance = distance_mode.iloc[0]
                    
                    # Most common health issue
                    chronic_conditions = [
                        'Do_you_have_diagnose_h_chronic_conditions/hypertension',
                        'Do_you_have_diagnose_h_chronic_conditions/diabetes',
                        'Do_you_have_diagnose_h_chronic_conditions/asthma',
                        'Do_you_have_diagnose_h_chronic_conditions/arthritis'
                    ]
                    
                    most_common_condition = "N/A"
                    max_count = 0
                    for condition in chronic_conditions:
                        if condition in village_data.columns:
                            count = village_data[condition].sum() if village_data[condition].dtype in ['int64', 'float64'] else village_data[condition].value_counts().get(1, 0)
                            if count > max_count:
                                max_count = count
                                most_common_condition = condition.split('/')[-1].replace('_', ' ').title()
                    
                    # Most common barrier
                    most_common_barrier = "N/A"
                    max_barrier_count = 0
                    for barrier in barrier_columns:
                        if barrier in village_data.columns:
                            count = village_data[barrier].sum() if village_data[barrier].dtype in ['int64', 'float64'] else village_data[barrier].value_counts().get(1, 0)
                            if count > max_barrier_count:
                                max_barrier_count = count
                                most_common_barrier = barrier.split('/')[-1].replace('_', ' ').title()
                    
                    village_summary.append({
                        'Village': village,
                        'Total Respondents': total_respondents,
                        'Most Common Distance': avg_distance,
                        'Most Common Health Issue': most_common_condition,
                        'Primary Healthcare Barrier': most_common_barrier
                    })
                
                summary_df = pd.DataFrame(village_summary)
                st.dataframe(summary_df, use_container_width=True)
        else:
            st.warning("Address/Village data not available in the dataset.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>Healthcare Access & Disease Prevalence Dashboard</strong></p>
        <p>Data automatically refreshes when the dataset is updated</p>
        <p>Built with Streamlit and Plotly for interactive visualization</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 