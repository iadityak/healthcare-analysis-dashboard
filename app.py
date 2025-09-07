import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
from datetime import datetime
import warnings
import base64
import io
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import plotly.io as pio
warnings.filterwarnings('ignore')

# Ensure consistent light theme and white backgrounds for all exports
pio.templates.default = "plotly_white"

# Use ReportLab's default Image behavior; set explicit sizes where used

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
    
    # Normalize common categorical columns to fix duplicates/inconsistencies
    def _standardize_text(value):
        if pd.isna(value):
            return value
        text = str(value).strip()
        # collapse whitespace and casing
        text = ' '.join(text.split())
        # Common normalizations
        lowered = text.lower()
        mappings = {
            'student': 'Student',
            'students': 'Student',
            'student ': 'Student',
            'housewife': 'Homemaker',
            'home maker': 'Homemaker',
            'homemaker': 'Homemaker',
            'farmer': 'Farmer',
            'shopkeeper': 'Business',
            'self employed': 'Self-Employed',
            'self-employed': 'Self-Employed',
            'labour': 'Laborer',
            'labor': 'Laborer',
            'labourer': 'Laborer',
            'laborer': 'Laborer',
            'business': 'Business',
            'govt job': 'Government Job',
            'government job': 'Government Job',
            'private job': 'Private Job'
        }
        if lowered in mappings:
            return mappings[lowered]
        return text.title()
    
    if 'Occupation' in df.columns:
        df['Occupation'] = df['Occupation'].apply(_standardize_text)
    if 'Education_Level' in df.columns:
        df['Education_Level'] = df['Education_Level'].apply(lambda x: _standardize_text(x))
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].apply(lambda x: _standardize_text(x))
    
    return df

# ---------- Visualization helpers ----------
def sort_order_control(key: str, default: str = 'Descending') -> bool:
    """Render a small sort order control and return ascending flag.
    key: unique key for Streamlit widget scoping.
    returns: True if ascending selected, else False.
    """
    order = st.radio(
        "Sort order",
        options=['Descending', 'Ascending'],
        index=0 if default == 'Descending' else 1,
        horizontal=True,
        key=f"sort_{key}"
    )
    return order == 'Ascending'

def sort_names_values(names, values, ascending: bool):
    """Sort paired names and values by values with given order."""
    paired = list(zip(names, values))
    paired.sort(key=lambda x: (x[1], x[0]), reverse=not ascending)
    sorted_names = [p[0] for p in paired]
    sorted_values = [p[1] for p in paired]
    return sorted_names, sorted_values

def add_bar_value_labels(fig, texts):
    """Attach data labels to bars (outside) with no clipping."""
    fig.update_traces(text=texts, textposition='outside', cliponaxis=False, texttemplate='%{text}')
    fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide')
    return fig

def add_pie_value_labels(fig):
    """Show percent and value on pie slices."""
    fig.update_traces(textinfo='label+percent+value')
    return fig

def clean_recent_health_label(raw_label: str) -> str:
    """Fix known naming inconsistencies for recent health problems."""
    label = raw_label.strip()
    # Normalize casing and remove duplicate suffixes like '.1'
    # Specific fix requested: General Health Conditions.1 -> General Health + Other Conditions
    if label.lower().startswith('general health conditions') and (label.endswith('.1') or label.endswith(' .1')):
        return 'General Health + Other Conditions'
    return label

def get_chronic_conditions_data(df):
    """Extract chronic conditions data with respondent counts"""
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
    total_respondents = len(df)
    
    for condition in chronic_conditions:
        if condition in df.columns:
            condition_name = condition.split('/')[-1].replace('_', ' ').title()
            count = df[condition].sum() if df[condition].dtype in ['int64', 'float64'] else df[condition].value_counts().get(1, 0)
            percentage = (count / total_respondents * 100) if total_respondents > 0 else 0
            condition_data[condition_name] = {
                'count': count,
                'total': total_respondents,
                'percentage': percentage,
                'label': f"{condition_name}\n{percentage:.1f}% ({count} of {total_respondents} households)"
            }
    
    return condition_data

def get_healthcare_providers_data(df):
    """Extract healthcare provider preference data with respondent counts"""
    provider_columns = [
        'where_do_you_usually_re_when_you_are_sick/government_hospital_health_post',
        'where_do_you_usually_re_when_you_are_sick/private_clinic',
        'where_do_you_usually_re_when_you_are_sick/pharmacy',
        'where_do_you_usually_re_when_you_are_sick/traditional_healer',
        'where_do_you_usually_re_when_you_are_sick/home_remedies',
        'where_do_you_usually_re_when_you_are_sick/i_do_not_seek_healthcare'
    ]
    
    provider_data = {}
    total_respondents = len(df)
    
    for provider in provider_columns:
        if provider in df.columns:
            provider_name = provider.split('/')[-1].replace('_', ' ').title()
            count = df[provider].sum() if df[provider].dtype in ['int64', 'float64'] else df[provider].value_counts().get(1, 0)
            percentage = (count / total_respondents * 100) if total_respondents > 0 else 0
            provider_data[provider_name] = {
                'count': count,
                'total': total_respondents,
                'percentage': percentage,
                'label': f"{provider_name}\n{percentage:.1f}% ({count} of {total_respondents} households)"
            }
    
    return provider_data

def get_provider_choice_reasons_data(df):
    """Extract reasons for healthcare provider choice with respondent counts"""
    reason_cols = [col for col in df.columns if 'What_is_the_primary_choosing_this_option' in col and '/' in col]
    
    reason_data = {}
    total_respondents = len(df)
    
    for col in reason_cols:
        reason_name = col.split('/')[-1].replace('_', ' ').title()
        count = df[col].sum() if df[col].dtype in ['int64', 'float64'] else df[col].value_counts().get(1, 0)
        
        if count > 0 and reason_name.lower() not in ['what is the primary choosing this option', 'nan', '']:
            percentage = (count / total_respondents * 100) if total_respondents > 0 else 0
            reason_data[reason_name] = {
                'count': count,
                'total': total_respondents,
                'percentage': percentage,
                'label': f"{reason_name}\n{percentage:.1f}% ({count} of {total_respondents} households)"
            }
    
    return reason_data

def get_barriers_data(df):
    """Extract healthcare barriers data with respondent counts"""
    barrier_columns = [
        'What_are_the_biggest_accessing_healthcare/distance',
        'What_are_the_biggest_accessing_healthcare/cost',
        'What_are_the_biggest_accessing_healthcare/poor_service_quality',
        'What_are_the_biggest_accessing_healthcare/cultural_beliefs',
        'What_are_the_biggest_accessing_healthcare/lack_of_transport'
    ]
    
    barrier_data = {}
    total_respondents = len(df)
    
    for barrier in barrier_columns:
        if barrier in df.columns:
            barrier_name = barrier.split('/')[-1].replace('_', ' ').title()
            count = df[barrier].sum() if df[barrier].dtype in ['int64', 'float64'] else df[barrier].value_counts().get(1, 0)
            percentage = (count / total_respondents * 100) if total_respondents > 0 else 0
            barrier_data[barrier_name] = {
                'count': count,
                'total': total_respondents,
                'percentage': percentage,
                'label': f"{barrier_name}\n{percentage:.1f}% ({count} of {total_respondents} households)"
            }
    
    return barrier_data

def get_preventive_services_data(df):
    """Extract truly preventive services data (excluding treatment services)"""
    # Define truly preventive services only
    preventive_cols = [
        'If_Yes_What_type/vaccination',
        'If_Yes_What_type/regular_check_ups', 
        'If_Yes_What_type/screening__diabetes__bp',
        'If_Yes_What_type/maternal__child_health_services'
    ]
    
    preventive_data = {}
    total_respondents = len(df)
    
    for col in preventive_cols:
        if col in df.columns:
            service_name = col.split('/')[-1].replace('_', ' ').title()
            count = df[col].sum() if df[col].dtype in ['int64', 'float64'] else df[col].value_counts().get(1, 0)
            
            if count > 0:
                percentage = (count / total_respondents * 100) if total_respondents > 0 else 0
                preventive_data[service_name] = {
                    'count': count,
                    'total': total_respondents,
                    'percentage': percentage,
                    'label': f"{service_name}\n{percentage:.1f}% ({count} of {total_respondents} households)"
                }
    
    return preventive_data

def get_child_nutrition_by_age_data(df):
    """Extract age-appropriate child nutrition data"""
    nutrition_data = {}
    total_respondents = len(df)
    
    # Age-appropriate feeding practices
    feeding_practices = {
        'Exclusive Breastfeeding (0-6 months)': 'What_is_their_main_source_of_nutrition/breastfeeding',
        'Complementary Feeding (6-24 months)': 'What_is_their_main_source_of_nutrition/homemade_food',
        'Family Foods (>24 months)': 'What_is_their_main_source_of_nutrition/packaged_food'
    }
    
    for practice_name, col in feeding_practices.items():
        if col in df.columns:
            count = df[col].sum() if df[col].dtype in ['int64', 'float64'] else df[col].value_counts().get(1, 0)
            
            if count > 0:
                percentage = (count / total_respondents * 100) if total_respondents > 0 else 0
                nutrition_data[practice_name] = {
                    'count': count,
                    'total': total_respondents,
                    'percentage': percentage,
                    'label': f"{practice_name}\n{percentage:.1f}% ({count} of {total_respondents} households)"
                }
    
    return nutrition_data

def get_dental_health_data(df):
    """Extract dental health symptoms and behaviors data"""
    dental_problems = [
        'Have_you_experienced_the_past_six_months/tooth_pain_or_sensitivity',
        'Have_you_experienced_the_past_six_months/bleeding_gums',
        'Have_you_experienced_the_past_six_months/swollen_or_red_gums',
        'Have_you_experienced_the_past_six_months/loose_or_missing_teeth',
        'Have_you_experienced_the_past_six_months/bad_breath_or__halitosis',
        'Have_you_experienced_the_past_six_months/cavities_or_tooth_decay',
        'Have_you_experienced_the_past_six_months/mouth_sores_or_ulcers'
    ]
    
    dental_data = {}
    total_respondents = len(df)
    
    for problem in dental_problems:
        if problem in df.columns:
            problem_name = problem.split('/')[-1].replace('_', ' ').title()
            count = df[problem].sum() if df[problem].dtype in ['int64', 'float64'] else df[problem].value_counts().get(1, 0)
            
            if count > 0:
                percentage = (count / total_respondents * 100) if total_respondents > 0 else 0
                dental_data[problem_name] = {
                    'count': count,
                    'total': total_respondents,
                    'percentage': percentage,
                    'label': f"{problem_name}\n{percentage:.1f}% ({count} of {total_respondents} households)"
                }
    
    return dental_data

def get_demographic_cross_analysis(df, demographic_col, health_col):
    """Perform cross-tabulation analysis between demographic and health variables"""
    if demographic_col not in df.columns or health_col not in df.columns:
        return pd.DataFrame()
    
    # Create cross-tabulation
    cross_tab = pd.crosstab(df[demographic_col], df[health_col], margins=True)
    
    # Calculate percentages
    cross_tab_pct = pd.crosstab(df[demographic_col], df[health_col], normalize='index') * 100
    
    return cross_tab, cross_tab_pct

def get_correlation_analysis(df):
    """Analyze correlations between key variables"""
    # Define numeric variables for correlation
    numeric_vars = []
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64'] and col not in ['_id', '_index']:
            numeric_vars.append(col)
    
    if len(numeric_vars) > 1:
        correlation_matrix = df[numeric_vars].corr()
        return correlation_matrix
    
    return pd.DataFrame()

def get_executive_insights(df):
    """Generate executive insights and key findings"""
    insights = {
        'total_respondents': len(df),
        'key_findings': [],
        'health_priorities': [],
        'geographic_insights': [],
        'demographic_patterns': [],
        'recommendations': [],
        'risk_factors': [],
        'infrastructure_gaps': []
    }
    
    # Health Priorities Analysis
    chronic_conditions = get_chronic_conditions_data(df)
    if chronic_conditions:
        sorted_conditions = sorted(chronic_conditions.items(), 
                                 key=lambda x: x[1]['count'], reverse=True)
        top_conditions = sorted_conditions[:5]
        insights['health_priorities'] = [
            {
                'condition': name,
                'prevalence': data['percentage'],
                'affected': data['count'],
                'severity': 'High' if data['percentage'] > 20 else 'Moderate' if data['percentage'] > 10 else 'Low'
            }
            for name, data in top_conditions
        ]
    
    # Healthcare Access Analysis
    barriers = get_barriers_data(df)
    if barriers:
        top_barriers = sorted(barriers.items(), 
                            key=lambda x: x[1]['count'], reverse=True)[:3]
        insights['infrastructure_gaps'] = [
            {
                'barrier': name,
                'prevalence': data['percentage'],
                'affected': data['count']
            }
            for name, data in top_barriers
        ]
    
    # Distance Analysis
    if 'How_far_is_the_neare_healthcare_facility' in df.columns:
        distance_data = df['How_far_is_the_neare_healthcare_facility'].value_counts()
        far_distance_count = 0
        total_responses = distance_data.sum()
        
        for distance, count in distance_data.items():
            if pd.notna(distance) and ('10 km' in str(distance) or 'More than' in str(distance)):
                far_distance_count += count
        
        if far_distance_count > 0:
            insights['key_findings'].append({
                'finding': 'Healthcare Access Challenge',
                'description': f'{far_distance_count} households ({far_distance_count/total_responses*100:.1f}%) live >10km from healthcare facilities',
                'impact': 'High',
                'urgency': 'Critical'
            })
    
    # Demographics Analysis
    if 'Education_Level' in df.columns and 'Occupation' in df.columns:
        edu_data = df['Education_Level'].value_counts()
        occ_data = df['Occupation'].value_counts()
        
        # Check for education-healthcare correlation
        preventive_cols = [
            'If_Yes_What_type/vaccination',
            'If_Yes_What_type/regular_check_ups',
            'If_Yes_What_type/screening__diabetes__bp'
        ]
        
        available_preventive = [c for c in preventive_cols if c in df.columns]
        if available_preventive and len(edu_data) > 1:
            # Analyze preventive care by education
            edu_prev_correlation = []
            for edu_level in edu_data.index:
                if pd.notna(edu_level):
                    edu_group = df[df['Education_Level'] == edu_level]
                    # Count households with ANY preventive service (not sum of all services)
                    households_with_preventive = 0
                    for _, row in edu_group.iterrows():
                        has_any_preventive = False
                        for col in available_preventive:
                            value = row[col]
                            if pd.notna(value) and ((isinstance(value, (int, float)) and value == 1) or (isinstance(value, str) and value.strip() == '1')):
                                has_any_preventive = True
                                break
                        if has_any_preventive:
                            households_with_preventive += 1
                    
                    if len(edu_group) > 0:
                        edu_prev_correlation.append((edu_level, households_with_preventive / len(edu_group)))
            
            if edu_prev_correlation:
                edu_prev_correlation.sort(key=lambda x: x[1], reverse=True)
                insights['demographic_patterns'].append({
                    'pattern': 'Education-Healthcare Correlation',
                    'description': f'Higher education levels show better preventive healthcare utilization',
                    'top_group': edu_prev_correlation[0][0],
                    'utilization_rate': f'{edu_prev_correlation[0][1]*100:.1f}%'
                })
    
    # Village-level Analysis
    if 'Address' in df.columns:
        df_temp = df.copy()
        df_temp['Village'] = df_temp['Address'].apply(clean_village_name)
        village_counts = df_temp['Village'].value_counts()
        
        if len(village_counts) > 1:
            # Identify underrepresented villages
            avg_responses = len(df) / len(village_counts)
            underrepresented = village_counts[village_counts < avg_responses * 0.5]
            high_participation = village_counts[village_counts > avg_responses * 1.5]
            
            if len(underrepresented) > 0:
                insights['geographic_insights'].append({
                    'insight': 'Data Coverage Gaps',
                    'description': f'{len(underrepresented)} villages have limited survey coverage',
                    'villages': list(underrepresented.index)[:5],  # Top 5
                    'recommendation': 'Increase outreach to underrepresented areas'
                })
            
            if len(high_participation) > 0:
                insights['geographic_insights'].append({
                    'insight': 'High Community Engagement',
                    'description': f'{len(high_participation)} villages show excellent survey participation',
                    'villages': list(high_participation.index)[:5],  # Top 5
                    'recommendation': 'Use these villages as models for community engagement strategies'
                })
    
    # Additional Health Insights
    if 'Gender' in df.columns:
        gender_health_analysis = []
        for gender in df['Gender'].unique():
            if pd.notna(gender):
                gender_group = df[df['Gender'] == gender]
                chronic_count = 0
                for condition_col in ['Do_you_have_diagnose_h_chronic_conditions/hypertension',
                                    'Do_you_have_diagnose_h_chronic_conditions/diabetes',
                                    'Do_you_have_diagnose_h_chronic_conditions/asthma']:
                    if condition_col in df.columns:
                        chronic_count += gender_group[condition_col].sum() if gender_group[condition_col].dtype in ['int64', 'float64'] else gender_group[condition_col].value_counts().get(1, 0)
                
                if len(gender_group) > 0:
                    gender_health_analysis.append((gender, chronic_count / len(gender_group) * 100))
        
        if len(gender_health_analysis) > 1:
            gender_health_analysis.sort(key=lambda x: x[1], reverse=True)
            insights['demographic_patterns'].append({
                'pattern': 'Gender Health Disparity',
                'description': f'Health outcomes vary significantly between genders',
                'top_group': gender_health_analysis[0][0],
                'utilization_rate': f'{gender_health_analysis[0][1]:.1f}%'
            })
    
    # Mental Health Analysis
    stress_cols = [col for col in df.columns if 'What_are_the_biggest_stress_in_your_life' in col and '/' in col]
    if stress_cols:
        # Count households with ANY stress factor (not sum of all stress factors)
        households_with_stress = 0
        for _, row in df.iterrows():
            has_stress = False
            for col in stress_cols:
                if col in df.columns:
                    value = row[col] if col in row.index else None
                    if pd.notna(value) and ((isinstance(value, (int, float)) and value == 1) or (isinstance(value, str) and value.strip() == '1')):
                        has_stress = True
                        break
            if has_stress:
                households_with_stress += 1
        
        stress_prevalence = (households_with_stress / len(df)) * 100
        if stress_prevalence > 30:
            insights['key_findings'].append({
                'finding': 'High Community Stress Levels',
                'description': f'{stress_prevalence:.1f}% of households report significant stress factors',
                'impact': 'High',
                'urgency': 'High'
            })
    
    # Substance Use Analysis - Count households with ANY substance use
    substance_cols = ['Tobacco', 'Alcohol', 'Chewing_betel_nut_Pan_Gutka_etc']
    available_substances = [col for col in substance_cols if col in df.columns]
    if available_substances:
        households_with_substance_use = 0
        for _, row in df.iterrows():
            has_substance_use = False
            for col in available_substances:
                if col in df.columns:
                    value = row[col] if col in row.index else None
                    if pd.notna(value):
                        if isinstance(value, str) and ('yes' in value.lower()):
                            has_substance_use = True
                            break
                        elif isinstance(value, (int, float)) and value == 1:
                            has_substance_use = True
                            break
            if has_substance_use:
                households_with_substance_use += 1
        
        substance_prevalence = (households_with_substance_use / len(df)) * 100
        if substance_prevalence > 25:
            insights['risk_factors'].append({
                'factor': 'High Substance Use',
                'prevalence': substance_prevalence,
                'description': f'{substance_prevalence:.1f}% of households report substance use',
                'recommendation': 'Implement substance abuse prevention programs'
            })
    
    # Water and Sanitation Analysis
    if 'Do_you_treat_water_before_drinking' in df.columns:
        water_treatment = df['Do_you_treat_water_before_drinking'].value_counts()
        if water_treatment.get('no', 0) > 0:
            no_treatment_pct = (water_treatment.get('no', 0) / water_treatment.sum()) * 100
            if no_treatment_pct > 20:
                insights['risk_factors'].append({
                    'factor': 'Water Safety Risk',
                    'prevalence': no_treatment_pct,
                    'description': f'{no_treatment_pct:.1f}% of households do not treat water before drinking',
                    'recommendation': 'Implement water safety education and infrastructure improvements'
                })
    
    # Comprehensive Maternal Health Analysis
    maternal_issues = []
    if 'Have_you_or_any_wome_d_any_antenatal_care' in df.columns:
        anc_no = df['Have_you_or_any_wome_d_any_antenatal_care'].str.contains('no', case=False, na=False).sum()
        if anc_no > 0:
            anc_gap_pct = (anc_no / len(df)) * 100
            if anc_gap_pct > 20:
                maternal_issues.append(f'{anc_gap_pct:.1f}% lack antenatal care')
    
    # Home delivery analysis
    home_delivery_cols = [col for col in df.columns if 'Where_do_women_in_yo_household_give_birth' in col and 'home' in col.lower()]
    if home_delivery_cols:
        home_deliveries = 0
        for col in home_delivery_cols:
            home_deliveries += df[col].sum() if df[col].dtype in ['int64', 'float64'] else df[col].value_counts().get(1, 0)
        if home_deliveries > 0:
            home_delivery_pct = (home_deliveries / len(df)) * 100
            if home_delivery_pct > 30:
                maternal_issues.append(f'{home_delivery_pct:.1f}% deliver at home')
    
    if maternal_issues:
        insights['key_findings'].append({
            'finding': 'Maternal Health Gaps',
            'description': f'Critical maternal health challenges: {"; ".join(maternal_issues)}',
            'impact': 'High',
            'urgency': 'High'
        })
    
    # Comprehensive Dental Health Analysis
    dental_data = get_dental_health_data(df)
    if dental_data:
        # Count households with ANY dental issue (not sum of all issues)
        dental_problem_cols = [
            'Have_you_experienced_the_past_six_months/tooth_pain_or_sensitivity',
            'Have_you_experienced_the_past_six_months/bleeding_gums',
            'Have_you_experienced_the_past_six_months/swollen_or_red_gums',
            'Have_you_experienced_the_past_six_months/loose_or_missing_teeth',
            'Have_you_experienced_the_past_six_months/bad_breath_or__halitosis',
            'Have_you_experienced_the_past_six_months/cavities_or_tooth_decay',
            'Have_you_experienced_the_past_six_months/mouth_sores_or_ulcers'
        ]
        
        households_with_dental_issues = 0
        for _, row in df.iterrows():
            has_dental_issue = False
            for col in dental_problem_cols:
                if col in df.columns:
                    value = row[col] if col in row.index else None
                    if pd.notna(value) and ((isinstance(value, (int, float)) and value == 1) or (isinstance(value, str) and value.strip() == '1')):
                        has_dental_issue = True
                        break
            if has_dental_issue:
                households_with_dental_issues += 1
        
        dental_prevalence = (households_with_dental_issues / len(df)) * 100
        if dental_prevalence > 40:
            most_common_dental = max(dental_data.items(), key=lambda x: x[1]['count'])
            insights['key_findings'].append({
                'finding': 'Widespread Dental Health Issues',
                'description': f'{dental_prevalence:.1f}% of households report dental problems. Most common: {most_common_dental[0]} ({most_common_dental[1]["percentage"]:.1f}%)',
                'impact': 'Medium',
                'urgency': 'Medium'
            })
    
    # Comprehensive Water Safety Analysis
    water_risks = []
    if 'Do_you_treat_water_before_drinking' in df.columns:
        no_treatment = df['Do_you_treat_water_before_drinking'].str.contains('no', case=False, na=False).sum()
        if no_treatment > 0:
            no_treatment_pct = (no_treatment / len(df)) * 100
            if no_treatment_pct > 15:
                water_risks.append(f'{no_treatment_pct:.1f}% do not treat drinking water')
    
    # Water source analysis
    unsafe_water_cols = [col for col in df.columns if 'What_is_your_primary_rinking_water_source' in col and ('river' in col.lower() or 'lake' in col.lower())]
    if unsafe_water_cols:
        unsafe_water_users = 0
        for col in unsafe_water_cols:
            unsafe_water_users += df[col].sum() if df[col].dtype in ['int64', 'float64'] else df[col].value_counts().get(1, 0)
        if unsafe_water_users > 0:
            unsafe_water_pct = (unsafe_water_users / len(df)) * 100
            if unsafe_water_pct > 10:
                water_risks.append(f'{unsafe_water_pct:.1f}% use potentially unsafe water sources')
    
    if water_risks:
        insights['key_findings'].append({
            'finding': 'Water Safety Concerns',
            'description': f'Water safety risks identified: {"; ".join(water_risks)}',
            'impact': 'High',
            'urgency': 'Critical'
        })
    
    # Comprehensive High-Altitude Health Analysis
    altitude_issues = []
    
    # UV Exposure Issues
    uv_cols = ['Have_you_had_severe_UV_rays_or_eye_pain', 'Have_you_had_severe_indness_or_eye_pain']
    for col in uv_cols:
        if col in df.columns:
            uv_problems = df[col].str.contains('yes', case=False, na=False).sum() if df[col].dtype == 'object' else df[col].sum()
            if uv_problems > 0:
                uv_prevalence = (uv_problems / len(df)) * 100
                if uv_prevalence > 10:
                    altitude_issues.append(f'{uv_prevalence:.1f}% experience UV-related eye problems')
                    break
    
    # Cold-related injuries
    if 'Have_you_experienced_tion_in_fingers_toes' in df.columns:
        cold_injuries = df['Have_you_experienced_tion_in_fingers_toes'].str.contains('yes', case=False, na=False).sum() if df['Have_you_experienced_tion_in_fingers_toes'].dtype == 'object' else df['Have_you_experienced_tion_in_fingers_toes'].sum()
        if cold_injuries > 0:
            cold_prevalence = (cold_injuries / len(df)) * 100
            if cold_prevalence > 8:
                altitude_issues.append(f'{cold_prevalence:.1f}% experience cold-related injuries')
    
    # UV Protection Gap
    if 'Do_you_use_sunscreen_ective_eyewear_daily' in df.columns:
        no_uv_protection = df['Do_you_use_sunscreen_ective_eyewear_daily'].str.contains('no', case=False, na=False).sum()
        if no_uv_protection > 0:
            no_protection_pct = (no_uv_protection / len(df)) * 100
            if no_protection_pct > 60:
                altitude_issues.append(f'{no_protection_pct:.1f}% do not use daily UV protection')
    
    if altitude_issues:
        insights['key_findings'].append({
            'finding': 'High-Altitude Health Risks',
            'description': f'Altitude-specific health challenges: {"; ".join(altitude_issues)}',
            'impact': 'Medium',
            'urgency': 'Medium'
        })
    
    # Healthcare Service Quality Analysis
    quality_issues = []
    if 'What_are_the_biggest_accessing_healthcare/poor_service_quality' in df.columns:
        poor_quality = df['What_are_the_biggest_accessing_healthcare/poor_service_quality'].sum() if df['What_are_the_biggest_accessing_healthcare/poor_service_quality'].dtype in ['int64', 'float64'] else df['What_are_the_biggest_accessing_healthcare/poor_service_quality'].value_counts().get(1, 0)
        if poor_quality > 0:
            quality_pct = (poor_quality / len(df)) * 100
            if quality_pct > 15:
                quality_issues.append(f'{quality_pct:.1f}% report poor service quality')
    
    # Emergency services availability
    emergency_cols = [col for col in df.columns if 'emergency' in col.lower() and ('What_healthcare_serv_nk_needs_improvement' in col or 'improvement' in col)]
    if emergency_cols:
        emergency_needs = 0
        for col in emergency_cols:
            emergency_needs += df[col].sum() if df[col].dtype in ['int64', 'float64'] else df[col].value_counts().get(1, 0)
        if emergency_needs > 0:
            emergency_pct = (emergency_needs / len(df)) * 100
            if emergency_pct > 20:
                quality_issues.append(f'{emergency_pct:.1f}% need emergency services improvement')
    
    if quality_issues:
        insights['infrastructure_gaps'].append({
            'barrier': 'Service Quality',
            'prevalence': max([15, 20]) if quality_issues else 0,  # Use highest percentage
            'affected': len([issue for issue in quality_issues]),
            'description': f'Healthcare quality concerns: {"; ".join(quality_issues)}'
        })
    
    # Nutrition and Food Security Analysis
    nutrition_issues = []
    if 'How_many_meals_do_you_eat_per_day' in df.columns:
        low_meals = df[df['How_many_meals_do_you_eat_per_day'] < 3].shape[0]
        if low_meals > 0:
            low_meals_pct = (low_meals / len(df)) * 100
            if low_meals_pct > 10:
                nutrition_issues.append(f'{low_meals_pct:.1f}% eat fewer than 3 meals per day')
    
    # Child nutrition analysis - Count households with packaged food reliance
    child_nutrition_cols = [col for col in df.columns if 'What_is_their_main_source_of_nutrition' in col and 'packaged_food' in col]
    if child_nutrition_cols:
        households_with_packaged_nutrition = 0
        for _, row in df.iterrows():
            has_packaged_nutrition = False
            for col in child_nutrition_cols:
                if col in df.columns:
                    value = row[col] if col in row.index else None
                    if pd.notna(value) and ((isinstance(value, (int, float)) and value == 1) or (isinstance(value, str) and value.strip() == '1')):
                        has_packaged_nutrition = True
                        break
            if has_packaged_nutrition:
                households_with_packaged_nutrition += 1
        
        if households_with_packaged_nutrition > 0:
            inadequate_pct = (households_with_packaged_nutrition / len(df)) * 100
            if inadequate_pct > 25:
                nutrition_issues.append(f'{inadequate_pct:.1f}% rely primarily on packaged foods for child nutrition')
    
    if nutrition_issues:
        # Calculate the highest prevalence from the issues
        prevalence_values = []
        for issue in nutrition_issues:
            # Extract percentage from the issue string
            pct_match = issue.split('%')[0].split()[-1]
            try:
                prevalence_values.append(float(pct_match))
            except:
                prevalence_values.append(0)
        
        max_prevalence = max(prevalence_values) if prevalence_values else 0
        
        insights['risk_factors'].append({
            'factor': 'Nutritional Risk',
            'prevalence': max_prevalence,
            'description': f'Nutrition and food security concerns: {"; ".join(nutrition_issues)}',
            'recommendation': 'Implement nutrition education and food security programs'
        })
    
    # Generate Recommendations
    insights['recommendations'] = generate_executive_recommendations(insights, df)
    
    return insights

def generate_executive_recommendations(insights, df):
    """Generate actionable recommendations based on insights"""
    recommendations = []
    
    # Health Priority Recommendations
    if insights['health_priorities']:
        top_condition = insights['health_priorities'][0]
        if top_condition['prevalence'] > 15:
            recommendations.append({
                'category': 'Public Health',
                'priority': 'High',
                'title': f'Address {top_condition["condition"]} Epidemic',
                'description': f'With {top_condition["prevalence"]:.1f}% prevalence, {top_condition["condition"]} requires immediate intervention',
                'actions': [
                    'Implement targeted screening programs',
                    'Develop prevention education campaigns',
                    'Ensure adequate treatment supplies'
                ],
                'timeline': '3-6 months',
                'impact': 'High'
            })
    
    # Infrastructure Recommendations
    if insights['infrastructure_gaps']:
        top_barrier = insights['infrastructure_gaps'][0]
        if top_barrier['prevalence'] > 20:
            recommendations.append({
                'category': 'Infrastructure',
                'priority': 'Critical',
                'title': f'Address {top_barrier["barrier"]} Issues',
                'description': f'{top_barrier["prevalence"]:.1f}% of households report {top_barrier["barrier"]} as a major barrier',
                'actions': [
                    'Develop targeted infrastructure improvements',
                    'Explore mobile healthcare solutions',
                    'Partner with local transport providers'
                ],
                'timeline': '6-12 months',
                'impact': 'High'
            })
    
    # Preventive Care Recommendations
    preventive_cols = [
        'If_Yes_What_type/vaccination',
        'If_Yes_What_type/regular_check_ups',
        'If_Yes_What_type/screening__diabetes__bp'
    ]
    
    available_preventive = [c for c in preventive_cols if c in df.columns]
    if available_preventive:
        total_with_preventive = 0
        for col in available_preventive:
            total_with_preventive += df[col].sum() if df[col].dtype in ['int64', 'float64'] else df[col].value_counts().get(1, 0)
        
        preventive_rate = (total_with_preventive / len(df)) * 100
        if preventive_rate < 30:
            recommendations.append({
                'category': 'Preventive Care',
                'priority': 'Medium',
                'title': 'Improve Preventive Healthcare Uptake',
                'description': f'Only {preventive_rate:.1f}% average utilization of preventive services',
                'actions': [
                    'Launch community health education programs',
                    'Implement reminder systems for regular check-ups',
                    'Train local health workers'
                ],
                'timeline': '6-9 months',
                'impact': 'Medium'
            })
    
    # Demographics-based Recommendations
    if insights['demographic_patterns']:
        for pattern in insights['demographic_patterns']:
            if 'Education-Healthcare' in pattern['pattern']:
                recommendations.append({
                    'category': 'Health Equity',
                    'priority': 'Medium',
                    'title': 'Address Education-Based Health Disparities',
                    'description': 'Healthcare utilization varies significantly by education level',
                    'actions': [
                        'Develop targeted outreach for lower education groups',
                        'Create culturally appropriate health materials',
                        'Train community health workers'
                    ],
                    'timeline': '9-12 months',
                    'impact': 'Medium'
                })
            elif 'Gender' in pattern['pattern']:
                recommendations.append({
                    'category': 'Gender Health',
                    'priority': 'High',
                    'title': 'Address Gender-Based Health Disparities',
                    'description': 'Significant health outcome differences between genders',
                    'actions': [
                        'Implement gender-specific health programs',
                        'Conduct targeted health screenings',
                        'Develop gender-sensitive healthcare protocols'
                    ],
                    'timeline': '6-9 months',
                    'impact': 'High'
                })
    
    # Risk Factor Recommendations
    if insights['risk_factors']:
        for risk in insights['risk_factors']:
            if 'Substance Use' in risk['factor']:
                recommendations.append({
                    'category': 'Public Health',
                    'priority': 'High',
                    'title': 'Substance Abuse Prevention Program',
                    'description': risk['description'],
                    'actions': [
                        'Establish community-based counseling services',
                        'Implement awareness campaigns about health risks',
                        'Train healthcare workers in addiction treatment',
                        'Create support groups for substance users'
                    ],
                    'timeline': '6-12 months',
                    'impact': 'High'
                })
            elif 'Water Safety' in risk['factor']:
                recommendations.append({
                    'category': 'Infrastructure',
                    'priority': 'Critical',
                    'title': 'Water Safety and Sanitation Program',
                    'description': risk['description'],
                    'actions': [
                        'Distribute water purification tablets/systems',
                        'Conduct water safety education campaigns',
                        'Improve water treatment infrastructure',
                        'Regular water quality testing'
                    ],
                    'timeline': '3-6 months',
                    'impact': 'High'
                })
            elif 'High-Altitude' in risk['factor']:
                recommendations.append({
                    'category': 'Environmental Health',
                    'priority': 'Medium',
                    'title': 'High-Altitude Health Protection Program',
                    'description': risk['description'],
                    'actions': [
                        'Distribute UV protection gear (sunglasses, sunscreen)',
                        'Provide cold weather protection equipment',
                        'Educate on high-altitude health risks',
                        'Train healthcare workers on altitude-related conditions'
                    ],
                    'timeline': '3-9 months',
                    'impact': 'Medium'
                })
    
    # Mental Health Recommendations
    for finding in insights['key_findings']:
        if 'Stress' in finding['finding']:
            recommendations.append({
                'category': 'Mental Health',
                'priority': 'High',
                'title': 'Community Mental Health Support Program',
                'description': finding['description'],
                'actions': [
                    'Establish community mental health centers',
                    'Train local counselors and support workers',
                    'Implement stress reduction programs',
                    'Create peer support networks'
                ],
                'timeline': '6-12 months',
                'impact': 'High'
            })
    
    # Geographic-based Recommendations
    for geo_insight in insights['geographic_insights']:
        if 'Coverage Gaps' in geo_insight['insight']:
            recommendations.append({
                'category': 'Data Collection',
                'priority': 'Medium',
                'title': 'Improve Survey Coverage in Underrepresented Areas',
                'description': geo_insight['description'],
                'actions': [
                    'Increase outreach teams in identified villages',
                    'Partner with local community leaders',
                    'Use mobile survey collection methods',
                    'Provide incentives for participation'
                ],
                'timeline': '3-6 months',
                'impact': 'Medium'
            })
        elif 'High Community Engagement' in geo_insight['insight']:
            recommendations.append({
                'category': 'Best Practices',
                'priority': 'Low',
                'title': 'Replicate Successful Community Engagement Models',
                'description': geo_insight['description'],
                'actions': [
                    'Study successful engagement strategies in high-participation villages',
                    'Document and share best practices',
                    'Train outreach teams using successful models',
                    'Recognize and reward community leaders'
                ],
                'timeline': '6-9 months',
                'impact': 'Medium'
            })
    
    return recommendations

def get_trend_analysis(df):
    """Analyze trends in the healthcare data"""
    trends = {
        'temporal_trends': [],
        'demographic_trends': [],
        'geographic_trends': [],
        'service_utilization_trends': []
    }
    
    # Temporal Analysis (if date data is available)
    if 'start' in df.columns:
        df_temp = df.copy()
        df_temp['Month'] = df_temp['start'].dt.to_period('M')
        monthly_counts = df_temp.groupby('Month').size()
        
        if len(monthly_counts) > 1:
            # Calculate trend direction
            trend_slope = np.polyfit(range(len(monthly_counts)), monthly_counts.values, 1)[0]
            trend_direction = 'Increasing' if trend_slope > 0 else 'Decreasing' if trend_slope < 0 else 'Stable'
            
            trends['temporal_trends'].append({
                'metric': 'Survey Response Rate',
                'direction': trend_direction,
                'magnitude': abs(trend_slope),
                'description': f'Survey responses are {trend_direction.lower()} over time'
            })
    
    # Age Group Trends
    if 'Basic_Demographics' in df.columns:
        age_col = 'Basic_Demographics'
        df_temp = df.copy()
        df_temp[age_col] = pd.to_numeric(df_temp[age_col], errors='coerce')
        
        age_groups = []
        df_temp.loc[df_temp[age_col] < 30, 'Age_Group'] = 'Young (18-29)'
        df_temp.loc[(df_temp[age_col] >= 30) & (df_temp[age_col] < 50), 'Age_Group'] = 'Middle-aged (30-49)'
        df_temp.loc[df_temp[age_col] >= 50, 'Age_Group'] = 'Older (50+)'
        
        age_health_analysis = df_temp.groupby('Age_Group').size()
        if len(age_health_analysis) > 0:
            trends['demographic_trends'].append({
                'category': 'Age Distribution',
                'pattern': f'Largest group: {age_health_analysis.idxmax()}',
                'percentage': f'{age_health_analysis.max()/len(df)*100:.1f}%',
                'insight': 'Age-specific healthcare strategies needed'
            })
    
    return trends

def detect_anomalies(df):
    """Detect anomalies and outliers in the data"""
    anomalies = {
        'statistical_anomalies': [],
        'geographic_anomalies': [],
        'demographic_anomalies': [],
        'health_anomalies': []
    }
    
    # Geographic Anomalies
    if 'Address' in df.columns:
        df_temp = df.copy()
        df_temp['Village'] = df_temp['Address'].apply(clean_village_name)
        village_counts = df_temp['Village'].value_counts()
        
        if len(village_counts) > 3:
            # Identify villages with unusually high or low response rates
            q75 = village_counts.quantile(0.75)
            q25 = village_counts.quantile(0.25)
            iqr = q75 - q25
            
            low_outliers = village_counts[village_counts < (q25 - 1.5 * iqr)]
            
            if len(low_outliers) > 0:
                anomalies['geographic_anomalies'].append({
                    'type': 'Low Response Rate',
                    'villages': list(low_outliers.index),
                    'values': list(low_outliers.values),
                    'description': 'Villages with unusually low survey participation'
                })
    
    # Health Anomalies - Check for unusual disease patterns
    chronic_conditions = get_chronic_conditions_data(df)
    if chronic_conditions:
        condition_percentages = [v['percentage'] for v in chronic_conditions.values()]
        if condition_percentages:
            mean_prev = np.mean(condition_percentages)
            std_prev = np.std(condition_percentages)
            
            for name, data in chronic_conditions.items():
                z_score = (data['percentage'] - mean_prev) / std_prev if std_prev > 0 else 0
                if abs(z_score) > 2:  # More than 2 standard deviations
                    anomalies['health_anomalies'].append({
                        'condition': name,
                        'prevalence': data['percentage'],
                        'z_score': z_score,
                        'type': 'Unusually High' if z_score > 0 else 'Unusually Low',
                        'description': f'{name} prevalence is {"significantly higher" if z_score > 0 else "significantly lower"} than expected'
                    })
    
    return anomalies

def ensure_reports_directory():
    """Ensure the reports directory exists in the app's working directory"""
    import os
    reports_dir = os.path.join(os.getcwd(), "reports")
    if not os.path.exists(reports_dir):
        try:
            os.makedirs(reports_dir, exist_ok=True)
            print(f"✅ Created reports directory: {reports_dir}")
        except Exception as e:
            print(f"⚠️ Could not create reports directory: {e}")
            return None
    return reports_dir

def get_cached_image_path(filename_prefix):
    """Check if a cached image exists and is less than 1 hour old"""
    import os
    import time
    import glob
    
    reports_dir = ensure_reports_directory()
    if not reports_dir:
        return None
    
    # Look for existing files with this prefix (ignoring timestamps)
    pattern = os.path.join(reports_dir, f"{filename_prefix}_*.png")
    existing_files = glob.glob(pattern)
    
    if not existing_files:
        return None
    
    # Check if any file is less than 1 hour old
    current_time = time.time()
    one_hour = 3600  # 1 hour in seconds
    
    for file_path in existing_files:
        try:
            file_mtime = os.path.getmtime(file_path)
            file_age = current_time - file_mtime
            
            if file_age < one_hour:
                # File is less than 1 hour old, use it
                print(f"🔄 Using cached image: {os.path.basename(file_path)} (age: {file_age/60:.1f} minutes)")
                return file_path
        except Exception as e:
            print(f"⚠️ Error checking file {file_path}: {e}")
            continue
    
    # All files are older than 1 hour, clean them up
    for file_path in existing_files:
        try:
            os.unlink(file_path)
            print(f"🗑️ Cleaned up expired cache: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"⚠️ Could not delete expired file {file_path}: {e}")
    
    return None

def cleanup_old_cache_files():
    """Clean up cache files older than 1 hour"""
    import os
    import time
    
    reports_dir = os.path.join(os.getcwd(), "reports")
    if not os.path.exists(reports_dir):
        return
    
    current_time = time.time()
    one_hour = 3600  # 1 hour in seconds
    
    try:
        for filename in os.listdir(reports_dir):
            if filename.endswith('.png'):
                file_path = os.path.join(reports_dir, filename)
                try:
                    file_mtime = os.path.getmtime(file_path)
                    file_age = current_time - file_mtime
                    
                    if file_age > one_hour:
                        os.unlink(file_path)
                        print(f"🗑️ Cleaned up old cache file: {filename} (age: {file_age/3600:.1f} hours)")
                except Exception as e:
                    print(f"⚠️ Error processing {filename}: {e}")
    except Exception as e:
        print(f"⚠️ Error during cache cleanup: {e}")

def save_plotly_as_bytesio(fig, filename_prefix):
    """Save a Plotly figure as BytesIO object for direct use in ReportLab"""
    try:
        # Check chart type
        is_pie_chart = any(trace.type == 'pie' for trace in fig.data)
        is_bar_chart = any(trace.type in ['bar', 'histogram'] for trace in fig.data)
        
        if is_pie_chart:
            # Configuration for pie charts - preserve colors, improve text
            fig.update_layout(
                font=dict(size=16, family="Arial, sans-serif", color="black"),
                paper_bgcolor='white',
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle", 
                    y=0.5,
                    xanchor="left",
                    x=1.02,
                    font=dict(size=14, color="black"),
                    bgcolor="rgba(255,255,255,0.8)"
                ),
                margin=dict(l=30, r=150, t=100, b=30),
                xaxis=dict(automargin=True),
                yaxis=dict(automargin=True),
                uniformtext_minsize=10,
                uniformtext_mode='hide',
                width=1200,
                height=600
            )
            # Make text more readable - use automatic color contrast
            fig.update_traces(
                textfont=dict(size=14),
                textinfo="percent",
                textposition="auto",
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}<extra></extra>"
            )
        elif is_bar_chart:
            # Configuration for bar charts - ensure colors are vibrant
            fig.update_layout(
                font=dict(size=14, family="Arial, sans-serif", color="black"),
                plot_bgcolor='white',
                paper_bgcolor='white',
                showlegend=False,
                margin=dict(l=80, r=60, t=100, b=80),
                xaxis=dict(automargin=True),
                yaxis=dict(automargin=True),
                uniformtext_minsize=10,
                uniformtext_mode='hide',
                width=1200,
                height=600
            )
            # Enhance bar chart appearance
            fig.update_traces(
                textfont=dict(size=12, color="black"),
                textposition="outside",
                cliponaxis=False,
                hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>"
            )
            # Ensure colorful bars if no color specified
            if fig.data:
                trace = fig.data[0]
                marker = getattr(trace, 'marker', None)
                color = getattr(marker, 'color', None)
                has_color = (color is not None) and (not hasattr(color, '__len__') or len(color) > 0)
                if not has_color:
                    # Apply a vibrant color palette
                    base_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                    num_bars = len(getattr(trace, 'x', []) or getattr(trace, 'y', []))
                    if num_bars <= len(base_palette):
                        colors_palette = base_palette[:num_bars]
                    else:
                        repeats = (num_bars + len(base_palette) - 1) // len(base_palette)
                        colors_palette = (base_palette * repeats)[:num_bars]
                    fig.update_traces(marker_color=colors_palette)
        else:
            # Configuration for other chart types
            fig.update_layout(
                font=dict(size=14, family="Arial, sans-serif", color="black"),
                plot_bgcolor='white',
                paper_bgcolor='white',
                showlegend=True if fig.data and len(fig.data) > 1 else False,
                margin=dict(l=80, r=60, t=100, b=80),
                xaxis=dict(automargin=True),
                yaxis=dict(automargin=True),
                uniformtext_minsize=10,
                uniformtext_mode='hide',
                width=1200,
                height=600
            )
        
        # Try multiple export methods
        img_bytes = None
        
        # Method 1: Try with kaleido (most reliable)
        try:
            import kaleido
            img_bytes = pio.to_image(
                fig, 
                format="png", 
                width=1200, 
                height=600, 
                scale=1,
                engine="kaleido"
            )
            print(f"✅ Successfully generated image for {filename_prefix} using kaleido")
        except Exception as e1:
            print(f"⚠️ Kaleido method failed for {filename_prefix}: {e1}")
            
            # Method 2: Try without specifying engine
            try:
                img_bytes = pio.to_image(
                    fig, 
                    format="png", 
                    width=1200, 
                    height=600, 
                    scale=1  # Lower scale for fallback
                )
                print(f"✅ Successfully generated image for {filename_prefix} using default engine")
            except Exception as e2:
                print(f"⚠️ Default method also failed for {filename_prefix}: {e2}")
                
                # Method 3: Try with different settings
                try:
                    # Simplify the figure for export
                    fig.update_layout(
                        width=800,
                        height=500,
                        showlegend=False,
                        font=dict(size=10)
                    )
                    img_bytes = pio.to_image(fig, format="png")
                    print(f"✅ Successfully generated image for {filename_prefix} using simplified settings")
                except Exception as e3:
                    print(f"❌ All methods failed for {filename_prefix}: {e3}")
                    return None
        
        if img_bytes and len(img_bytes) > 0:
            # Return BytesIO object directly
            import io
            img_buffer = io.BytesIO(img_bytes)
            print(f"✅ Image BytesIO created for {filename_prefix} ({len(img_bytes)} bytes)")
            return img_buffer
        else:
            print(f"❌ No image data generated for {filename_prefix}")
            return None
            
    except Exception as e:
        print(f"❌ Critical error generating image for {filename_prefix}: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return None

def save_plotly_as_image(fig, filename_prefix):
    """Save a Plotly figure as PNG image and return the image path (fallback method)"""
    try:
        # Check for cached image first
        cached_path = get_cached_image_path(filename_prefix)
        if cached_path:
            return cached_path
            
        # Check chart type
        is_pie_chart = any(trace.type == 'pie' for trace in fig.data)
        is_bar_chart = any(trace.type in ['bar', 'histogram'] for trace in fig.data)
        
        if is_pie_chart:
            # Configuration for pie charts - preserve colors, improve text
            fig.update_layout(
                font=dict(size=16, family="Arial, sans-serif", color="black"),
                paper_bgcolor='white',
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle", 
                    y=0.5,
                    xanchor="left",
                    x=1.02,
                    font=dict(size=14, color="black"),
                    bgcolor="rgba(255,255,255,0.8)"
                ),
                margin=dict(l=30, r=150, t=100, b=30),
                xaxis=dict(automargin=True),
                yaxis=dict(automargin=True),
                uniformtext_minsize=10,
                uniformtext_mode='hide',
                width=1200,
                height=600
            )
            # Make text more readable - use automatic color contrast
            fig.update_traces(
                textfont=dict(size=14),
                textinfo="percent",
                textposition="auto",
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}<extra></extra>"
            )
        elif is_bar_chart:
            # Configuration for bar charts - ensure colors are vibrant
            fig.update_layout(
                font=dict(size=14, family="Arial, sans-serif", color="black"),
                plot_bgcolor='white',
                paper_bgcolor='white',
                showlegend=False,
                margin=dict(l=80, r=60, t=100, b=80),
                xaxis=dict(automargin=True),
                yaxis=dict(automargin=True),
                uniformtext_minsize=10,
                uniformtext_mode='hide',
                width=1200,
                height=600
            )
            # Enhance bar chart appearance
            fig.update_traces(
                textfont=dict(size=12, color="black"),
                textposition="outside",
                cliponaxis=False,
                hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>"
            )
            # Ensure colorful bars if no color specified
            if fig.data:
                trace = fig.data[0]
                marker = getattr(trace, 'marker', None)
                color = getattr(marker, 'color', None)
                has_color = (color is not None) and (not hasattr(color, '__len__') or len(color) > 0)
                if not has_color:
                    # Apply a vibrant color palette
                    base_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                    num_bars = len(getattr(trace, 'x', []) or getattr(trace, 'y', []))
                    if num_bars <= len(base_palette):
                        colors_palette = base_palette[:num_bars]
                    else:
                        repeats = (num_bars + len(base_palette) - 1) // len(base_palette)
                        colors_palette = (base_palette * repeats)[:num_bars]
                    fig.update_traces(marker_color=colors_palette)
        else:
            # Configuration for other chart types
            fig.update_layout(
                font=dict(size=14, family="Arial, sans-serif", color="black"),
                plot_bgcolor='white',
                paper_bgcolor='white',
                showlegend=True if fig.data and len(fig.data) > 1 else False,
                margin=dict(l=80, r=60, t=100, b=80),
                xaxis=dict(automargin=True),
                yaxis=dict(automargin=True),
                uniformtext_minsize=10,
                uniformtext_mode='hide',
                width=1200,
                height=600
            )
        
        # Try multiple export methods
        img_bytes = None
        
        # Method 1: Try with kaleido (most reliable)
        try:
            import kaleido
            img_bytes = pio.to_image(
                fig, 
                format="png", 
                width=1200, 
                height=600, 
                scale=1,
                engine="kaleido"
            )
            print(f"✅ Successfully generated image for {filename_prefix} using kaleido")
        except Exception as e1:
            print(f"⚠️ Kaleido method failed for {filename_prefix}: {e1}")
            
            # Method 2: Try without specifying engine
            try:
                img_bytes = pio.to_image(
                    fig, 
                    format="png", 
                    width=1200, 
                    height=600, 
                    scale=1  # Lower scale for fallback
                )
                print(f"✅ Successfully generated image for {filename_prefix} using default engine")
            except Exception as e2:
                print(f"⚠️ Default method also failed for {filename_prefix}: {e2}")
                
                # Method 3: Try with different settings
                try:
                    # Simplify the figure for export
                    fig.update_layout(
                        width=1200,
                        height=600,
                        showlegend=False,
                        font=dict(size=10)
                    )
                    img_bytes = pio.to_image(fig, format="png")
                    print(f"✅ Successfully generated image for {filename_prefix} using simplified settings")
                except Exception as e3:
                    print(f"❌ All methods failed for {filename_prefix}: {e3}")
                    return None
        
        if img_bytes:
            # Create file in reports directory
            import os
            import time
            from datetime import datetime
            
            # Ensure reports directory exists
            reports_dir = ensure_reports_directory()
            if reports_dir is None:
                print(f"❌ Cannot create reports directory, falling back to system temp")
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png', prefix=f'{filename_prefix}_')
                file_path = temp_file.name
                temp_file.close()
            else:
                # Create file in reports directory with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # microseconds to milliseconds
                filename = f"{filename_prefix}_{timestamp}.png"
                file_path = os.path.join(reports_dir, filename)
            
            # Write image data to file
            try:
                with open(file_path, 'wb') as img_file:
                    img_file.write(img_bytes)
                    img_file.flush()  # Ensure data is written to disk
                
                # Give a moment for filesystem to sync
                time.sleep(0.1)
                
                # Verify file was created and is readable
                if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                    # Test that the file is actually readable
                    try:
                        with open(file_path, 'rb') as test_file:
                            test_data = test_file.read(100)  # Read first 100 bytes
                            if len(test_data) > 0:
                                print(f"✅ Image file created in reports dir: {file_path} ({os.path.getsize(file_path)} bytes)")
                                return file_path
                            else:
                                print(f"❌ Image file is empty: {file_path}")
                                return None
                    except Exception as read_error:
                        print(f"❌ Cannot read created image file {file_path}: {read_error}")
                        return None
                else:
                    print(f"❌ Image file was not created or is empty: {file_path}")
                    return None
                    
            except Exception as write_error:
                print(f"❌ Error writing image file {file_path}: {write_error}")
                return None
        else:
            print(f"❌ No image data generated for {filename_prefix}")
            return None
            
    except Exception as e:
        print(f"❌ Critical error saving plot {filename_prefix}: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return None

def add_chart_to_pdf(elements, fig, filename_prefix, chart_title, styles):
    """Helper function to add chart to PDF with fallback handling using BytesIO"""
    # Use globally patched Image and units
    from reportlab.lib.units import inch
    import os
    
    subheading_style = styles['subheading']
    normal_style = styles['normal']
    
    # Check for cached image first (fastest option)
    cached_path = get_cached_image_path(filename_prefix)
    if cached_path:
        try:
            # Verify the cached file exists and is readable
            if os.path.exists(cached_path) and os.path.getsize(cached_path) > 0:
                with open(cached_path, 'rb') as test_file:
                    test_data = test_file.read(100)
                    if len(test_data) > 0:
                        cached_image = Image(cached_path, width=6*inch, height=3*inch)
                        elements.append(cached_image)
                        elements.append(Spacer(1, 12))
                        print(f"✅ Successfully used cached image {os.path.basename(cached_path)} in PDF")
                        return
        except Exception as cache_error:
            print(f"⚠️ Cached file {cached_path} exists but has issues: {cache_error}")
    
    # Try BytesIO method first (more reliable)
    img_buffer = save_plotly_as_bytesio(fig, filename_prefix)
    if img_buffer:
        try:
            # Reset buffer position to beginning
            img_buffer.seek(0)
            # Create Image object directly from BytesIO
            chart_image = Image(img_buffer, width=6*inch, height=3*inch)
            elements.append(chart_image)
            elements.append(Spacer(1, 12))
            print(f"✅ Successfully added BytesIO image {filename_prefix} to PDF")
            return
        except Exception as img_error:
            print(f"❌ ReportLab cannot create Image object from BytesIO for {filename_prefix}: {img_error}")
            # Fall through to file method
    
    # Fallback to file method
    img_path = save_plotly_as_image(fig, filename_prefix)
    if img_path:
        try:
            # Verify the image file exists and is accessible before adding to PDF
            if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                # Test that ReportLab can create an Image object
                try:
                    test_image = Image(img_path, width=6*inch, height=3*inch)
                    elements.append(test_image)
                    elements.append(Spacer(1, 12))
                    print(f"✅ Successfully added file image {filename_prefix} to PDF")
                    return
                except Exception as img_error:
                    print(f"❌ ReportLab cannot create Image object for {img_path}: {img_error}")
                    # Fall through to text fallback
            else:
                print(f"❌ Image file not accessible: {img_path}")
                # Fall through to text fallback
        except Exception as e:
            print(f"❌ Error processing image {img_path}: {e}")
            # Fall through to text fallback
    
    # Text fallback if all image methods fail
    elements.append(Paragraph(f"📊 {chart_title}", subheading_style))
    elements.append(Paragraph("❌ Chart could not be generated due to image export issues.", normal_style))
    elements.append(Paragraph("💡 Please check that kaleido is properly installed and permissions are correct.", normal_style))
    elements.append(Spacer(1, 12))

def generate_comprehensive_pdf_report(insights, trends, anomalies, df):
    """Generate a comprehensive PDF report with graphs from all tabs"""
    # Clean up old cache files (older than 1 hour) before starting
    cleanup_old_cache_files()
    
    # Import all necessary components
    import io
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    import plotly.express as px
    import plotly.graph_objects as go
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=22,
        spaceAfter=20,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#1f4e79')
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=8,
        spaceBefore=15,
        textColor=colors.HexColor('#2c5aa0')
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubheading',
        parent=styles['Heading3'],
        fontSize=12,
        spaceAfter=6,
        spaceBefore=10,
        textColor=colors.HexColor('#2c5aa0')
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6,
        leading=12
    )
    
    # Title
    elements.append(Paragraph("🏥 Ladakh Healthcare Survey Comprehensive Report", title_style))
    elements.append(Spacer(1, 15))
    
    # Date and Summary
    current_date = datetime.now().strftime("%B %d, %Y")
    elements.append(Paragraph(f"<b>Report Generated:</b> {current_date}", normal_style))
    elements.append(Paragraph(f"<b>Total Households Surveyed:</b> {insights['total_respondents']:,}", normal_style))
    elements.append(Spacer(1, 12))
    
    # Executive Summary KPIs
    elements.append(Paragraph("🎯 Executive Summary - Key Performance Indicators", heading_style))
    
    kpi_data = [
        ['Health Metric', 'Value', 'Status'],
        ['Total Households Surveyed', f"{insights['total_respondents']:,}", 'Complete'],
    ]
    
    # Add comprehensive KPI data
    if insights['health_priorities']:
        top_condition = insights['health_priorities'][0]
        kpi_data.append(['Primary Health Concern', f"{top_condition['condition']} ({top_condition['prevalence']:.1f}%)", 
                        'High Priority' if top_condition['prevalence'] > 20 else 'Monitor'])
    
    if insights['infrastructure_gaps']:
        top_barrier = insights['infrastructure_gaps'][0]
        kpi_data.append(['Major Access Barrier', f"{top_barrier['barrier']} ({top_barrier['prevalence']:.1f}%)", 
                        'Critical' if top_barrier['prevalence'] > 30 else 'Important'])
    
    # Add more KPIs
    chronic_conditions = get_chronic_conditions_data(df)
    if chronic_conditions:
        total_chronic = sum(v['count'] for v in chronic_conditions.values())
        risk_score = (total_chronic / len(df)) * 100
        kpi_data.append(['Community Health Risk Score', f"{risk_score:.1f}%", 
                        'High Risk' if risk_score > 50 else 'Moderate Risk' if risk_score > 25 else 'Low Risk'])
    
    # Water Safety
    if 'Do_you_treat_water_before_drinking' in df.columns:
        water_yes = df['Do_you_treat_water_before_drinking'].str.contains('yes', case=False, na=False).sum()
        water_total = df['Do_you_treat_water_before_drinking'].count()
        water_safety_score = (water_yes / water_total * 100) if water_total > 0 else 0
        kpi_data.append(['Water Safety Practices', f"{water_safety_score:.1f}%", 
                        'Good' if water_safety_score > 80 else 'Needs Improvement'])
    
    # Maternal Health
    if 'Have_you_or_any_wome_d_any_antenatal_care' in df.columns:
        anc_yes = df['Have_you_or_any_wome_d_any_antenatal_care'].str.contains('yes', case=False, na=False).sum()
        anc_total = df['Have_you_or_any_wome_d_any_antenatal_care'].count()
        anc_percentage = (anc_yes / anc_total * 100) if anc_total > 0 else 0
        kpi_data.append(['Antenatal Care Coverage', f"{anc_percentage:.1f}%", 
                        'Excellent' if anc_percentage > 90 else 'Good' if anc_percentage > 70 else 'Needs Improvement'])
    
    kpi_table = Table(kpi_data, colWidths=[2.5*inch, 1.8*inch, 1.2*inch])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    
    elements.append(kpi_table)
    elements.append(Spacer(1, 12))
    
    # Generate and add graphs
    try:
        # Health Priorities Chart
        if insights['health_priorities']:
            elements.append(Paragraph("📊 Health Priorities Analysis", heading_style))
            
            # Create health priorities bar chart
            conditions = [item['condition'] for item in insights['health_priorities']]
            prevalences = [item['prevalence'] for item in insights['health_priorities']]
            severities = [item['severity'] for item in insights['health_priorities']]
            
            color_map = {'High': '#d32f2f', 'Moderate': '#f57c00', 'Low': '#388e3c'}
            colors_list = [color_map[severity] for severity in severities]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=prevalences,
                    y=conditions,
                    orientation='h',
                    marker_color=colors_list,
                    text=[f"{p:.1f}%" for p in prevalences],
                    textposition='outside',
                    name='Prevalence'
                )
            ])
            
            fig.update_layout(
                title="Top 5 Health Priorities by Prevalence",
                xaxis_title="Prevalence (%)",
                yaxis_title="Health Conditions",
                height=400,
                margin=dict(l=150),
                showlegend=False
            )
            
            img_path = save_plotly_as_image(fig, "health_priorities")
            if img_path:
                elements.append(Image(img_path, width=6*inch, height=3*inch))
                elements.append(Spacer(1, 10))
            
            # Add text summary
            for priority in insights['health_priorities'][:3]:  # Top 3
                severity_color = '#d32f2f' if priority['severity'] == 'High' else '#f57c00' if priority['severity'] == 'Moderate' else '#388e3c'
                elements.append(Paragraph(
                    f"• <b>{priority['condition']}</b>: {priority['prevalence']:.1f}% prevalence "
                    f"({priority['affected']} households) - <font color='{severity_color}'>{priority['severity']} Priority</font>",
                    normal_style
                ))
            
            elements.append(Spacer(1, 10))
            
        # Infrastructure Barriers Chart
        if insights['infrastructure_gaps']:
            elements.append(Paragraph("🚧 Healthcare Access Barriers", heading_style))
            
            barriers = [item['barrier'] for item in insights['infrastructure_gaps']]
            barrier_prevalences = [item['prevalence'] for item in insights['infrastructure_gaps']]
            
            fig = px.pie(
                values=barrier_prevalences,
                names=barriers,
                title="Primary Healthcare Access Barriers"
            )
            
            fig.update_traces(textinfo='percent+label', textposition='inside')
            fig.update_layout(height=400, showlegend=True)
            
            img_path = save_plotly_as_image(fig, "access_barriers")
            if img_path:
                elements.append(Image(img_path, width=6*inch, height=3*inch))
                elements.append(Spacer(1, 10))
    
    except Exception as e:
        elements.append(Paragraph(f"Note: Chart generation encountered an issue: {str(e)}", normal_style))
        elements.append(Spacer(1, 10))
    
    # Critical Findings (compact format)
    if insights['key_findings']:
        elements.append(Paragraph("🚨 Critical Findings", heading_style))
        for finding in insights['key_findings']:
            urgency_color = '#d32f2f' if finding['urgency'] == 'Critical' else '#f57c00'
            elements.append(Paragraph(
                f"• <font color='{urgency_color}'><b>{finding['finding']}</b></font>: {finding['description']}",
                normal_style
            ))
        elements.append(Spacer(1, 8))
    
    # Risk Factors (compact format)
    if insights['risk_factors']:
        elements.append(Paragraph("🚩 Identified Risk Factors", heading_style))
        for risk in insights['risk_factors']:
            elements.append(Paragraph(
                f"• <b>{risk['factor']}</b>: {risk['prevalence']:.1f}% prevalence - {risk['description']}",
                normal_style
            ))
        elements.append(Spacer(1, 8))
    
    # Page break before detailed analysis
    elements.append(PageBreak())
    
    # Disease Prevalence Analysis with Charts
    elements.append(Paragraph("🦠 Disease Prevalence Analysis", heading_style))
    
    try:
        # Chronic conditions chart
        chronic_conditions = get_chronic_conditions_data(df)
        if chronic_conditions:
            names = list(chronic_conditions.keys())
            values = [v['count'] for v in chronic_conditions.values()]
            
            fig = px.bar(
                x=names,
                y=values,
                title=f"Chronic Conditions Prevalence (Total: {len(df)} households)",
                labels={'x': 'Condition', 'y': 'Number of Cases'},
                color=values,
                color_continuous_scale='Reds'
            )
            
            fig.update_layout(xaxis_tickangle=-45, height=400, showlegend=False)
            
            img_path = save_plotly_as_image(fig, "chronic_conditions")
            if img_path:
                elements.append(Image(img_path, width=6*inch, height=3*inch))
                elements.append(Spacer(1, 10))
        
        # Recent health problems
        recent_health_cols = [col for col in df.columns if 'In_the_past_6_months_any_of_the_following' in col and '/' in col]
        if recent_health_cols:
            recent_health_data = {}
            total_respondents = len(df)
            
            for col in recent_health_cols:
                condition_name = col.split('/')[-1].replace('_', ' ').title()
                count = df[col].sum() if df[col].dtype in ['int64', 'float64'] else df[col].value_counts().get(1, 0)
                
                if count > 0 and condition_name.lower() not in ['in the past 6 months any of the following', 'nan', '']:
                    recent_health_data[condition_name] = count
            
            if recent_health_data:
                fig = px.pie(
                    values=list(recent_health_data.values()),
                    names=list(recent_health_data.keys()),
                    title=f"Recent Health Problems (Past 6 Months) - Total: {len(df)} households"
                )
                fig.update_traces(textinfo='percent+label', textposition='inside')
                fig.update_layout(height=400)
                
                img_path = save_plotly_as_image(fig, "recent_health")
                if img_path:
                    elements.append(Image(img_path, width=6*inch, height=3*inch))
                    elements.append(Spacer(1, 10))
    
    except Exception as e:
        elements.append(Paragraph(f"Note: Disease prevalence chart generation encountered an issue: {str(e)}", normal_style))
    
    elements.append(Spacer(1, 8))
    
    # Add more detailed analysis sections with charts
    
    # Maternal & Child Health Analysis
    elements.append(Paragraph("👶 Maternal & Child Health Analysis", heading_style))
    
    try:
        # Antenatal care chart
        if 'Have_you_or_any_wome_d_any_antenatal_care' in df.columns:
            anc_data = df['Have_you_or_any_wome_d_any_antenatal_care'].value_counts()
            total_anc = anc_data.sum()
            
            fig = px.pie(
                values=anc_data.values,
                names=[f"{name} ({value}/{total_anc})" for name, value in zip(anc_data.index, anc_data.values)],
                title=f"Antenatal Care Utilization ({total_anc}/{len(df)} households)"
            )
            fig.update_traces(textinfo='percent+label', textposition='inside')
            fig.update_layout(height=350)
            
            img_path = save_plotly_as_image(fig, "anc_utilization")
            if img_path:
                elements.append(Image(img_path, width=5*inch, height=2.5*inch))
                elements.append(Spacer(1, 8))
        
        # Child vaccination chart
        if 'Are_children_in_your_old_fully_vaccinated' in df.columns:
            vacc_data = df['Are_children_in_your_old_fully_vaccinated'].value_counts()
            total_vacc = vacc_data.sum()
            
            fig = px.pie(
                values=vacc_data.values,
                names=[f"{name} ({value}/{total_vacc})" for name, value in zip(vacc_data.index, vacc_data.values)],
                title=f"Child Vaccination Status ({total_vacc}/{len(df)} households)"
            )
            fig.update_traces(textinfo='percent+label', textposition='inside')
            fig.update_layout(height=350)
            
            img_path = save_plotly_as_image(fig, "vaccination_status")
            if img_path:
                elements.append(Image(img_path, width=5*inch, height=2.5*inch))
                elements.append(Spacer(1, 8))
    
    except Exception as e:
        elements.append(Paragraph(f"Note: Maternal health chart generation encountered an issue: {str(e)}", normal_style))
    
    # Water & Sanitation Analysis
    elements.append(Paragraph("💧 Water & Sanitation Analysis", heading_style))
    
    try:
        # Water treatment chart
        if 'Do_you_treat_water_before_drinking' in df.columns:
            water_data = df['Do_you_treat_water_before_drinking'].value_counts()
            total_water = water_data.sum()
            
            fig = px.bar(
                x=list(water_data.index),
                y=list(water_data.values),
                title=f"Water Treatment Practices ({total_water}/{len(df)} households)",
                labels={'x': 'Treatment Practice', 'y': 'Number of Households'}
            )
            fig.update_layout(height=350, showlegend=False)
            
            img_path = save_plotly_as_image(fig, "water_treatment")
            if img_path:
                elements.append(Image(img_path, width=5*inch, height=2.5*inch))
                elements.append(Spacer(1, 8))
    
    except Exception as e:
        elements.append(Paragraph(f"Note: Water analysis chart generation encountered an issue: {str(e)}", normal_style))
    
    # Dental Health Analysis
    elements.append(Paragraph("🦷 Dental Health Analysis", heading_style))
    
    try:
        dental_data = get_dental_health_data(df)
        if dental_data:
            names = list(dental_data.keys())
            values = [v['count'] for v in dental_data.values()]
            
            fig = px.bar(
                x=names,
                y=values,
                title=f"Dental Problems (Past 6 Months) - Total: {len(df)} households",
                labels={'x': 'Dental Problem', 'y': 'Number of Cases'},
                color=values,
                color_continuous_scale='Oranges'
            )
            fig.update_layout(xaxis_tickangle=-45, height=400, showlegend=False)
            
            img_path = save_plotly_as_image(fig, "dental_health")
            if img_path:
                elements.append(Image(img_path, width=6*inch, height=3*inch))
                elements.append(Spacer(1, 8))
    
    except Exception as e:
        elements.append(Paragraph(f"Note: Dental health chart generation encountered an issue: {str(e)}", normal_style))
    
    # New page for recommendations
    elements.append(PageBreak())
    
    # Priority Recommendations (more compact)
    elements.append(Paragraph("🎯 Priority Recommendations", heading_style))
    
    if insights['recommendations']:
        # Group by priority
        critical_recs = [r for r in insights['recommendations'] if r['priority'] == 'Critical']
        high_recs = [r for r in insights['recommendations'] if r['priority'] == 'High']
        medium_recs = [r for r in insights['recommendations'] if r['priority'] == 'Medium']
        
        for priority, recs, color in [
            ('🔴 Critical Priority', critical_recs, '#d32f2f'),
            ('🟠 High Priority', high_recs, '#f57c00'),
            ('🟡 Medium Priority', medium_recs, '#fbc02d')
        ]:
            if recs:
                elements.append(Paragraph(f"{priority} Actions:", subheading_style))
                for rec in recs[:3]:  # Limit to top 3 per priority
                    elements.append(Paragraph(
                        f"<font color='{color}'><b>{rec['title']}</b></font>",
                        normal_style
                    ))
                    elements.append(Paragraph(f"<b>Category:</b> {rec['category']}", normal_style))
                    elements.append(Paragraph(f"<b>Description:</b> {rec['description']}", normal_style))
                    elements.append(Paragraph("<b>Key Actions:</b>", normal_style))
                    for action in rec['actions'][:3]:  # Limit to top 3 actions
                        elements.append(Paragraph(f"  • {action}", normal_style))
                    elements.append(Paragraph(f"<b>Expected Impact:</b> {rec['impact']}", normal_style))
                    elements.append(Spacer(1, 6))
    
    # Action Items Summary Table
    elements.append(Paragraph("📋 Action Items Summary", heading_style))
    
    if insights['recommendations']:
        action_data = [['Priority', 'Category', 'Action', 'Impact']]
        for rec in insights['recommendations']:
            action_data.append([
                rec['priority'],
                rec['category'],
                rec['title'][:50] + '...' if len(rec['title']) > 50 else rec['title'],
                rec['impact']
            ])
        
        action_table = Table(action_data, colWidths=[1*inch, 1.3*inch, 3.5*inch, 1*inch])
        action_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        
        elements.append(action_table)
    
    # Final summary
    elements.append(Spacer(1, 15))
    elements.append(Paragraph("📄 Report Summary", heading_style))
    elements.append(Paragraph(
        f"This comprehensive healthcare survey analysis covers {insights['total_respondents']:,} households "
        f"across multiple health domains including disease prevalence, healthcare access, maternal and child health, "
        f"water safety, dental health, and high-altitude health risks. "
        f"The analysis identified {len(insights.get('key_findings', []))} critical findings and generated "
        f"{len(insights.get('recommendations', []))} actionable recommendations for improving community health outcomes.",
        normal_style
    ))
    
    # Note: Images are now cached in reports directory for 1 hour for better performance
    
    # Build PDF
    doc.build(elements)
    
    # Get the value of the BytesIO buffer and return it
    pdf_data = buffer.getvalue()
    buffer.close()
    
    return pdf_data

def generate_all_tabs_pdf_report(insights, trends, anomalies, df):
    """Generate a comprehensive PDF report with ALL graphs from ALL tabs"""
    # Clean up old cache files (older than 1 hour) before starting
    cleanup_old_cache_files()
    
    # Import all necessary components
    import io
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    import plotly.express as px
    import plotly.graph_objects as go
    
    buffer = io.BytesIO()
    # Use landscape orientation for better chart viewing
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=22,
        spaceAfter=20,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#1f4e79')
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=8,
        spaceBefore=15,
        textColor=colors.HexColor('#2c5aa0')
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubheading',
        parent=styles['Heading3'],
        fontSize=12,
        spaceAfter=6,
        spaceBefore=10,
        textColor=colors.HexColor('#2c5aa0')
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6,
        leading=12
    )
    
    # Title
    elements.append(Paragraph("🏥 Ladakh Healthcare Survey - Complete Dashboard Report", title_style))
    elements.append(Spacer(1, 15))
    
    # Date and Summary
    current_date = datetime.now().strftime("%B %d, %Y")
    elements.append(Paragraph(f"<b>Report Generated:</b> {current_date}", normal_style))
    elements.append(Paragraph(f"<b>Total Households Surveyed:</b> {insights['total_respondents']:,}", normal_style))
    elements.append(Paragraph("<b>Coverage:</b> Complete analysis across all health domains", normal_style))
    
    # Add diagnostic information
    elements.append(Spacer(1, 8))
    elements.append(Paragraph("🔧 System Diagnostics", subheading_style))
    
    # Check kaleido availability
    try:
        import kaleido
        kaleido_version = kaleido.__version__ if hasattr(kaleido, '__version__') else "Unknown"
        elements.append(Paragraph(f"✅ Kaleido available: Version {kaleido_version}", normal_style))
    except ImportError:
        elements.append(Paragraph("❌ Kaleido not found. Charts may not display. Install with: pip install kaleido", normal_style))
    
    # Check plotly version
    try:
        import plotly
        plotly_version = plotly.__version__
        elements.append(Paragraph(f"✅ Plotly version: {plotly_version}", normal_style))
    except:
        elements.append(Paragraph("❌ Plotly version could not be determined", normal_style))
    
    # Reports directory information
    reports_dir = ensure_reports_directory()
    if reports_dir:
        elements.append(Paragraph(f"📁 Reports directory: {reports_dir}", normal_style))
        import os
        if os.path.exists(reports_dir):
            png_files = [f for f in os.listdir(reports_dir) if f.endswith('.png')]
            elements.append(Paragraph(f"🖼️ Temporary image files: {len(png_files)} PNG files in reports directory", normal_style))
        else:
            elements.append(Paragraph("❌ Reports directory not accessible", normal_style))
    else:
        elements.append(Paragraph("❌ Could not create reports directory", normal_style))
    
    elements.append(Spacer(1, 12))
    

    
    try:
        # === TAB 1: EXECUTIVE SUMMARY ===
        elements.append(Paragraph("📋 EXECUTIVE SUMMARY", heading_style))
        
        # Health Priorities Chart
        if insights['health_priorities']:
            conditions = [item['condition'] for item in insights['health_priorities']]
            prevalences = [item['prevalence'] for item in insights['health_priorities']]
            severities = [item['severity'] for item in insights['health_priorities']]
            
            color_map = {'High': '#d32f2f', 'Moderate': '#f57c00', 'Low': '#388e3c'}
            colors_list = [color_map[severity] for severity in severities]
            
            fig = go.Figure(data=[
                go.Bar(x=prevalences, y=conditions, orientation='h', marker_color=colors_list,
                       text=[f"{p:.1f}%" for p in prevalences], textposition='outside')
            ])
            fig.update_layout(title="Executive Summary: Top Health Priorities", xaxis_title="Prevalence (%)",
                            yaxis_title="Health Conditions", height=400, margin=dict(l=150), showlegend=False)
            
            # Use the safe image addition function
            styles_dict = {'subheading': subheading_style, 'normal': normal_style}
            add_chart_to_pdf(elements, fig, "exec_health_priorities", "Executive Summary: Top Health Priorities Chart", styles_dict)
            
            # Add text-based data as additional information
            if insights['health_priorities']:
                elements.append(Paragraph("📋 Health Priority Data:", normal_style))
                for item in insights['health_priorities'][:5]:
                    elements.append(Paragraph(f"• {item['condition']}: {item['prevalence']:.1f}% ({item['affected']} households)", normal_style))
                elements.append(Spacer(1, 8))
        
        # === TAB 2: DISEASE PREVALENCE ===
        elements.append(PageBreak())
        elements.append(Paragraph("🦠 DISEASE PREVALENCE ANALYSIS", heading_style))
        
        # Chronic conditions chart
        chronic_conditions = get_chronic_conditions_data(df)
        if chronic_conditions:
            names = list(chronic_conditions.keys())
            values = [v['count'] for v in chronic_conditions.values()]
            
            fig = px.bar(x=names, y=values, title="Chronic Conditions Prevalence",
                        labels={'x': 'Condition', 'y': 'Number of Cases'}, 
                        color=values, color_continuous_scale='Reds')
            fig.update_layout(xaxis_tickangle=-45, height=400, showlegend=False)
            
            add_chart_to_pdf(elements, fig, "disease_chronic", "Chronic Conditions Prevalence Chart", styles_dict)
        
        # Recent health problems
        recent_health_cols = [col for col in df.columns if 'In_the_past_6_months_any_of_the_following' in col and '/' in col]
        if recent_health_cols:
            recent_health_data = {}
            for col in recent_health_cols:
                condition_name = col.split('/')[-1].replace('_', ' ').title()
                count = df[col].sum() if df[col].dtype in ['int64', 'float64'] else df[col].value_counts().get(1, 0)
                if count > 0:
                    recent_health_data[condition_name] = count
            
            if recent_health_data:
                fig = px.pie(values=list(recent_health_data.values()), names=list(recent_health_data.keys()),
                           title="Recent Health Problems (Past 6 Months)")
                fig.update_traces(textinfo='percent+label', textposition='inside')
                fig.update_layout(height=400)
                
                add_chart_to_pdf(elements, fig, "disease_recent", "Recent Health Problems Chart", styles_dict)
        
        # === TAB 3: HEALTHCARE ACCESS ===
        elements.append(PageBreak())
        elements.append(Paragraph("🏥 HEALTHCARE ACCESS ANALYSIS", heading_style))
        
        # Healthcare provider preferences
        provider_data = get_healthcare_providers_data(df)
        if provider_data:
            names = list(provider_data.keys())
            values = [v['count'] for v in provider_data.values()]
            
            fig = px.bar(x=names, y=values, title="Healthcare Provider Preferences",
                        labels={'x': 'Provider Type', 'y': 'Number of Users'},
                        color=values, color_continuous_scale='Blues')
            fig.update_layout(xaxis_tickangle=-45, height=400, showlegend=False)
            
            add_chart_to_pdf(elements, fig, "access_providers", "Healthcare Provider Preferences Chart", styles_dict)
        
        # Distance to healthcare facility
        if 'How_far_is_the_neare_healthcare_facility' in df.columns:
            distance_data = df['How_far_is_the_neare_healthcare_facility'].value_counts()
            
            fig = px.pie(values=distance_data.values, names=distance_data.index,
                       title="Distance to Nearest Healthcare Facility")
            fig.update_traces(textinfo='percent+label', textposition='inside')
            fig.update_layout(height=400)
            
            img_path = save_plotly_as_image(fig, "access_distance")
            if img_path:
                elements.append(Image(img_path, width=6*inch, height=3*inch))
                elements.append(Spacer(1, 8))
        
        # Healthcare barriers
        barriers_data = get_barriers_data(df)
        if barriers_data:
            names = list(barriers_data.keys())
            values = [v['count'] for v in barriers_data.values()]
            
            fig = px.bar(x=names, y=values, title="Healthcare Access Barriers",
                        labels={'x': 'Barrier Type', 'y': 'Number of Reports'},
                        color=values, color_continuous_scale='Reds')
            fig.update_layout(xaxis_tickangle=-45, height=400, showlegend=False)
            
            img_path = save_plotly_as_image(fig, "access_barriers")
            if img_path:
                elements.append(Image(img_path, width=6*inch, height=3*inch))
                elements.append(Spacer(1, 8))
        
        # === TAB 4: HEALTH SEEKING BEHAVIOR ===
        elements.append(PageBreak())
        elements.append(Paragraph("🎯 HEALTH SEEKING BEHAVIOR", heading_style))
        
        # Preventive services
        preventive_data = get_preventive_services_data(df)
        if preventive_data:
            names = list(preventive_data.keys())
            values = [v['count'] for v in preventive_data.values()]
            
            fig = px.bar(x=names, y=values, title="Preventive Services Utilization",
                        labels={'x': 'Preventive Service', 'y': 'Number of Households'},
                        color=values, color_continuous_scale='Greens')
            fig.update_layout(xaxis_tickangle=-20, height=400, showlegend=False)
            
            img_path = save_plotly_as_image(fig, "behavior_preventive")
            if img_path:
                elements.append(Image(img_path, width=6*inch, height=3*inch))
                elements.append(Spacer(1, 8))
        
        # Healthcare expenditure
        if 'How_much_do_you_usua_d_consultation_costs' in df.columns:
            cost_data = df['How_much_do_you_usua_d_consultation_costs'].value_counts()
            
            fig = px.bar(x=list(cost_data.index), y=list(cost_data.values),
                       title="Healthcare Consultation Cost Distribution",
                       labels={'x': 'Cost Range', 'y': 'Number of Households'})
            fig.update_layout(xaxis_tickangle=-30, height=400, showlegend=False)
            
            img_path = save_plotly_as_image(fig, "behavior_costs")
            if img_path:
                elements.append(Image(img_path, width=6*inch, height=3*inch))
                elements.append(Spacer(1, 8))
        
        # === TAB 5: MATERNAL & CHILD HEALTH ===
        elements.append(PageBreak())
        elements.append(Paragraph("👶 MATERNAL & CHILD HEALTH", heading_style))
        
        # Place of delivery
        delivery_cols = [col for col in df.columns if 'Where_do_women_in_yo_household_give_birth' in col and '/' in col]
        if delivery_cols:
            delivery_data = {}
            for col in delivery_cols:
                place_name = col.split('/')[-1].replace('_', ' ').title()
                count = df[col].sum() if df[col].dtype in ['int64', 'float64'] else df[col].value_counts().get(1, 0)
                if count > 0:
                    delivery_data[place_name] = count
            
            if delivery_data:
                fig = px.pie(values=list(delivery_data.values()), names=list(delivery_data.keys()),
                           title="Place of Delivery")
                fig.update_traces(textinfo='percent+label', textposition='inside')
                fig.update_layout(height=400)
                
                img_path = save_plotly_as_image(fig, "maternal_delivery")
                if img_path:
                    elements.append(Image(img_path, width=6*inch, height=3*inch))
                    elements.append(Spacer(1, 8))
        
        # Antenatal care utilization
        if 'Have_you_or_any_wome_d_any_antenatal_care' in df.columns:
            anc_data = df['Have_you_or_any_wome_d_any_antenatal_care'].value_counts()
            
            fig = px.pie(values=anc_data.values, names=anc_data.index,
                       title="Antenatal Care Utilization")
            fig.update_traces(textinfo='percent+label', textposition='inside')
            fig.update_layout(height=400)
            
            img_path = save_plotly_as_image(fig, "maternal_anc")
            if img_path:
                elements.append(Image(img_path, width=6*inch, height=3*inch))
                elements.append(Spacer(1, 8))
        
        # Child vaccination status
        if 'Are_children_in_your_old_fully_vaccinated' in df.columns:
            vacc_data = df['Are_children_in_your_old_fully_vaccinated'].value_counts()
            
            fig = px.pie(values=vacc_data.values, names=vacc_data.index,
                       title="Child Vaccination Status")
            fig.update_traces(textinfo='percent+label', textposition='inside')
            fig.update_layout(height=400)
            
            img_path = save_plotly_as_image(fig, "maternal_vaccination")
            if img_path:
                elements.append(Image(img_path, width=6*inch, height=3*inch))
                elements.append(Spacer(1, 8))
        
        # === TAB 6: MENTAL HEALTH & SUBSTANCE USE ===
        elements.append(PageBreak())
        elements.append(Paragraph("🧠 MENTAL HEALTH & SUBSTANCE USE", heading_style))
        
        # Stress sources
        stress_cols = [col for col in df.columns if 'What_are_the_biggest_stress_in_your_life' in col and '/' in col]
        if stress_cols:
            stress_data = {}
            for col in stress_cols:
                stress_name = col.split('/')[-1].replace('_', ' ').title()
                count = df[col].sum() if df[col].dtype in ['int64', 'float64'] else df[col].value_counts().get(1, 0)
                if count > 0:
                    stress_data[stress_name] = count
            
            if stress_data:
                fig = px.bar(x=list(stress_data.keys()), y=list(stress_data.values()),
                           title="Major Sources of Stress",
                           labels={'x': 'Stress Source', 'y': 'Number of Cases'})
                fig.update_layout(xaxis_tickangle=-45, height=400, showlegend=False)
                
                img_path = save_plotly_as_image(fig, "mental_stress")
                if img_path:
                    elements.append(Image(img_path, width=6*inch, height=3*inch))
                    elements.append(Spacer(1, 8))
        
        # Substance use
        substance_cols = ['Tobacco', 'Alcohol', 'Chewing_betel_nut_Pan_Gutka_etc']
        substance_data = {}
        for col in substance_cols:
            if col in df.columns:
                substance_name = col.replace('_', ' ').title()
                count = df[col].str.contains('yes', case=False, na=False).sum() if df[col].dtype == 'object' else df[col].sum()
                if count > 0:
                    substance_data[substance_name] = count
        
        if substance_data:
            fig = px.bar(x=list(substance_data.keys()), y=list(substance_data.values()),
                       title="Substance Use Patterns",
                       labels={'x': 'Substance', 'y': 'Number of Users'})
            fig.update_layout(height=400, showlegend=False)
            
            img_path = save_plotly_as_image(fig, "mental_substance")
            if img_path:
                elements.append(Image(img_path, width=6*inch, height=3*inch))
                elements.append(Spacer(1, 8))
        
        # === TAB 7: INFRASTRUCTURE NEEDS ===
        elements.append(PageBreak())
        elements.append(Paragraph("🏗️ INFRASTRUCTURE NEEDS", heading_style))
        
        # Water sources
        water_cols = [col for col in df.columns if 'What_is_your_primary_rinking_water_source' in col and '/' in col]
        if water_cols:
            water_data = {}
            for col in water_cols:
                water_name = col.split('/')[-1].replace('_', ' ').title()
                count = df[col].sum() if df[col].dtype in ['int64', 'float64'] else df[col].value_counts().get(1, 0)
                if count > 0:
                    water_data[water_name] = count
            
            if water_data:
                fig = px.pie(values=list(water_data.values()), names=list(water_data.keys()),
                           title="Primary Water Sources")
                fig.update_traces(textinfo='percent+label', textposition='inside')
                fig.update_layout(height=400)
                
                img_path = save_plotly_as_image(fig, "infrastructure_water")
                if img_path:
                    elements.append(Image(img_path, width=6*inch, height=3*inch))
                    elements.append(Spacer(1, 8))
        
        # Water treatment
        if 'Do_you_treat_water_before_drinking' in df.columns:
            treat_data = df['Do_you_treat_water_before_drinking'].value_counts()
            
            fig = px.bar(x=list(treat_data.index), y=list(treat_data.values),
                       title="Water Treatment Practices",
                       labels={'x': 'Response', 'y': 'Number of Households'})
            fig.update_layout(height=400, showlegend=False)
            
            img_path = save_plotly_as_image(fig, "infrastructure_treatment")
            if img_path:
                elements.append(Image(img_path, width=6*inch, height=3*inch))
                elements.append(Spacer(1, 8))
        
        # === TAB 8: DENTAL HEALTH ===
        elements.append(PageBreak())
        elements.append(Paragraph("🦷 DENTAL HEALTH", heading_style))
        
        # Dental health problems
        dental_data = get_dental_health_data(df)
        if dental_data:
            names = list(dental_data.keys())
            values = [v['count'] for v in dental_data.values()]
            
            fig = px.bar(x=names, y=values, title="Dental Problems (Past 6 Months)",
                        labels={'x': 'Dental Problem', 'y': 'Number of Cases'},
                        color=values, color_continuous_scale='Oranges')
            fig.update_layout(xaxis_tickangle=-45, height=400, showlegend=False)
            
            img_path = save_plotly_as_image(fig, "dental_problems")
            if img_path:
                elements.append(Image(img_path, width=6*inch, height=3*inch))
                elements.append(Spacer(1, 8))
        
        # Dental care behaviors
        if 'How_often_do_you_brush_your_teeth' in df.columns:
            brushing_data = df['How_often_do_you_brush_your_teeth'].value_counts()
            
            fig = px.pie(values=brushing_data.values, names=brushing_data.index,
                       title="Tooth Brushing Frequency")
            fig.update_traces(textinfo='percent+label', textposition='inside')
            fig.update_layout(height=400)
            
            img_path = save_plotly_as_image(fig, "dental_brushing")
            if img_path:
                elements.append(Image(img_path, width=6*inch, height=3*inch))
                elements.append(Spacer(1, 8))
        
        # === TAB 9: DEMOGRAPHIC ANALYSIS ===
        elements.append(PageBreak())
        elements.append(Paragraph("📊 DEMOGRAPHIC ANALYSIS", heading_style))
        
        # Education distribution
        if 'Education_Level' in df.columns:
            edu_data = df['Education_Level'].value_counts()
            
            fig = px.bar(x=list(edu_data.index), y=list(edu_data.values),
                       title="Education Level Distribution",
                       labels={'x': 'Education Level', 'y': 'Number of Households'})
            fig.update_layout(xaxis_tickangle=-45, height=400, showlegend=False)
            
            img_path = save_plotly_as_image(fig, "demo_education")
            if img_path:
                elements.append(Image(img_path, width=6*inch, height=3*inch))
                elements.append(Spacer(1, 8))
        
        # Occupation distribution
        if 'Occupation' in df.columns:
            occ_data = df['Occupation'].value_counts()
            
            fig = px.bar(x=list(occ_data.index), y=list(occ_data.values),
                       title="Occupation Distribution",
                       labels={'x': 'Occupation', 'y': 'Number of Households'})
            fig.update_layout(xaxis_tickangle=-45, height=400, showlegend=False)
            
            img_path = save_plotly_as_image(fig, "demo_occupation")
            if img_path:
                elements.append(Image(img_path, width=6*inch, height=3*inch))
                elements.append(Spacer(1, 8))
        
        # Income distribution
        if 'Total_number_of_hous_ncome_monthly_Approx' in df.columns:
            income_data = df['Total_number_of_hous_ncome_monthly_Approx'].value_counts()
            cleaned_income_data = {}
            for income, count in income_data.items():
                cleaned_label = clean_income_label(income)
                cleaned_income_data[cleaned_label] = count
            
            fig = px.bar(x=list(cleaned_income_data.keys()), y=list(cleaned_income_data.values()),
                       title="Income Distribution",
                       labels={'x': 'Monthly Household Income', 'y': 'Number of Households'})
            fig.update_layout(xaxis_tickangle=-20, height=400, showlegend=False)
            
            img_path = save_plotly_as_image(fig, "demo_income")
            if img_path:
                elements.append(Image(img_path, width=6*inch, height=3*inch))
                elements.append(Spacer(1, 8))
        
        # === TAB 10: HIGH-ALTITUDE HEALTH ===
        elements.append(PageBreak())
        elements.append(Paragraph("🏔️ HIGH-ALTITUDE HEALTH", heading_style))
        
        # UV protection usage
        if 'Do_you_use_sunscreen_ective_eyewear_daily' in df.columns:
            uv_data = df['Do_you_use_sunscreen_ective_eyewear_daily'].value_counts()
            
            fig = px.bar(x=list(uv_data.index), y=list(uv_data.values),
                       title="Daily UV Protection Use",
                       labels={'x': 'Response', 'y': 'Number of Households'})
            fig.update_layout(height=400, showlegend=False)
            
            img_path = save_plotly_as_image(fig, "altitude_uv")
            if img_path:
                elements.append(Image(img_path, width=6*inch, height=3*inch))
                elements.append(Spacer(1, 8))
        
        # High-altitude health issues
        altitude_health_data = {}
        altitude_cols = ['Have_you_had_severe_UV_rays_or_eye_pain', 'Have_you_experienced_tion_in_fingers_toes']
        for col in altitude_cols:
            if col in df.columns:
                issue_name = 'UV/Eye Problems' if 'UV' in col else 'Cold Injuries'
                count = df[col].str.contains('yes', case=False, na=False).sum() if df[col].dtype == 'object' else df[col].sum()
                if count > 0:
                    altitude_health_data[issue_name] = count
        
        if altitude_health_data:
            fig = px.bar(x=list(altitude_health_data.keys()), y=list(altitude_health_data.values()),
                       title="High-Altitude Health Issues",
                       labels={'x': 'Issue Type', 'y': 'Number of Cases'})
            fig.update_layout(height=400, showlegend=False)
            
            img_path = save_plotly_as_image(fig, "altitude_issues")
            if img_path:
                elements.append(Image(img_path, width=6*inch, height=3*inch))
                elements.append(Spacer(1, 8))
        
        # Final summary
        elements.append(PageBreak())
        elements.append(Paragraph("📄 Complete Dashboard Report Summary", heading_style))
        elements.append(Paragraph(
            f"This comprehensive report includes ALL visualizations from ALL dashboard tabs, covering {insights['total_respondents']:,} households "
            f"across the complete spectrum of health domains: disease prevalence, healthcare access, health seeking behavior, "
            f"maternal and child health, mental health and substance use, infrastructure needs, dental health, "
            f"demographic analysis, high-altitude health, and village-level analysis. "
            f"This complete visual analysis provides stakeholders with every chart and graph from the full dashboard for comprehensive healthcare planning.",
            normal_style
        ))
        
    except Exception as e:
        elements.append(Paragraph(f"Note: Some charts could not be generated due to: {str(e)}", normal_style))
    
    # Note: Images are now cached in reports directory for 1 hour for better performance
    
    # Build PDF
    doc.build(elements)
    
    # Get the value of the BytesIO buffer and return it
    pdf_data = buffer.getvalue()
    buffer.close()
    
    return pdf_data

# Legacy function name for compatibility
def generate_pdf_report(insights, trends, anomalies, df):
    return generate_comprehensive_pdf_report(insights, trends, anomalies, df)

def analyze_others_responses(df, column_name):
    """Analyze and categorize 'Others specify' responses"""
    others_col = f"{column_name}_Others_Specify"
    
    if others_col not in df.columns:
        return {}
    
    # Get all non-null responses
    responses = df[others_col].dropna().astype(str)
    
    if len(responses) == 0:
        return {}
    
    # Simple categorization (can be enhanced with NLP)
    categorized_responses = {}
    
    for response in responses:
        response = response.lower().strip()
        
        # Define categories based on common themes
        if any(word in response for word in ['doctor', 'physician', 'specialist']):
            category = 'Medical Professional Preference'
        elif any(word in response for word in ['cost', 'expensive', 'money', 'price']):
            category = 'Cost-related'
        elif any(word in response for word in ['distance', 'far', 'travel', 'transport']):
            category = 'Distance/Transport'
        elif any(word in response for word in ['quality', 'service', 'facility']):
            category = 'Service Quality'
        elif any(word in response for word in ['traditional', 'home', 'family']):
            category = 'Traditional/Home Care'
        else:
            category = 'Other Specific'
        
        categorized_responses[category] = categorized_responses.get(category, 0) + 1
    
    return categorized_responses

def clean_income_label(income_label):
    """Clean and standardize income labels for better readability"""
    if pd.isna(income_label) or income_label == '':
        return 'Not Specified'
    
    income_str = str(income_label).strip().lower()
    
    # Map to more readable formats
    income_mappings = {
        'less_than_50_000_inr': 'Under ₹50K',
        'less than 50,000 inr': 'Under ₹50K',
        '50_000_100000_inr': '₹50K-₹100K',
        '50,000-100,000 inr': '₹50K-₹100K',
        '100000_150000_inr': '₹100K-₹150K',
        '100,000-150,000 inr': '₹100K-₹150K',
        '150000_200000_inr': '₹150K-₹200K',
        '150,000-200,000 inr': '₹150K-₹200K',
        'over_200000_inr': 'Over ₹200K',
        'over 200,000 inr': 'Over ₹200K'
    }
    
    # Check for exact matches first
    if income_str in income_mappings:
        return income_mappings[income_str]
    
    # Check for partial matches
    for key, value in income_mappings.items():
        if key.replace('_', ' ').replace(',', '') in income_str.replace('_', ' ').replace(',', ''):
            return value
    
    # Return cleaned version if no mapping found
    return income_str.replace('_', ' ').title()

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
    tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "📋 Executive Summary",
        "🦠 Disease Prevalence", 
        "🏥 Healthcare Access", 
        "🎯 Health Seeking Behavior", 
        "👶 Maternal & Child Health", 
        "🧠 Mental Health & Substance Use",
        "🏗️ Infrastructure Needs",
        "🦷 Dental Health",
        "📊 Demographic Analysis",
        "🏔️ High-Altitude Health",
        "🏘️ Village-Level Analysis"
    ])
    
    with tab0:
        st.markdown('<h1 class="section-header">📋 Executive Summary</h1>', unsafe_allow_html=True)
        
        # Generate insights, trends, and anomalies
        insights = get_executive_insights(df)
        trends = get_trend_analysis(df)
        anomalies = detect_anomalies(df)
        
        # Key Performance Indicators
        st.markdown("## 🎯 Key Performance Indicators")
        
        # Row 1: Primary Health Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Households Surveyed", 
                f"{insights['total_respondents']:,}",
                help="Total number of households that participated in the health survey"
            )
        
        with col2:
            if insights['health_priorities']:
                top_condition = insights['health_priorities'][0]
                st.metric(
                    "Primary Health Concern", 
                    top_condition['condition'],
                    f"{top_condition['prevalence']:.1f}% prevalence",
                    help=f"Most prevalent chronic condition affecting {top_condition['affected']} households"
                )
        
        with col3:
            if insights['infrastructure_gaps']:
                top_barrier = insights['infrastructure_gaps'][0]
                st.metric(
                    "Major Access Barrier", 
                    top_barrier['barrier'],
                    f"{top_barrier['prevalence']:.1f}% affected",
                    help=f"Primary healthcare access challenge affecting {top_barrier['affected']} households"
                )
        
        with col4:
            # Calculate overall health risk score
            chronic_conditions = get_chronic_conditions_data(df)
            if chronic_conditions:
                total_chronic = sum(v['count'] for v in chronic_conditions.values())
                risk_score = (total_chronic / len(df)) * 100
                st.metric(
                    "Community Health Risk", 
                    f"{risk_score:.1f}%",
                    "Overall chronic disease burden",
                    help="Percentage of households reporting any chronic health condition"
                )
        
        # Row 2: Maternal & Child Health
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Maternal Health Coverage
            if 'Have_you_or_any_wome_d_any_antenatal_care' in df.columns:
                anc_yes = df['Have_you_or_any_wome_d_any_antenatal_care'].str.contains('yes', case=False, na=False).sum()
                anc_total = df['Have_you_or_any_wome_d_any_antenatal_care'].count()
                anc_percentage = (anc_yes / anc_total * 100) if anc_total > 0 else 0
                st.metric(
                    "Antenatal Care Coverage",
                    f"{anc_percentage:.1f}%",
                    f"{anc_yes} of {anc_total} households",
                    help="Percentage of households with antenatal care access"
                )
        
        with col2:
            # Child Vaccination Rate
            if 'Are_children_in_your_old_fully_vaccinated' in df.columns:
                vacc_yes = df['Are_children_in_your_old_fully_vaccinated'].str.contains('yes', case=False, na=False).sum()
                vacc_total = df['Are_children_in_your_old_fully_vaccinated'].count()
                vacc_percentage = (vacc_yes / vacc_total * 100) if vacc_total > 0 else 0
                st.metric(
                    "Child Vaccination Rate",
                    f"{vacc_percentage:.1f}%",
                    f"{vacc_yes} of {vacc_total} households",
                    help="Percentage of households with fully vaccinated children"
                )
        
        with col3:
            # Preventive Care Utilization
            preventive_cols = [
                'If_Yes_What_type/vaccination',
                'If_Yes_What_type/regular_check_ups',
                'If_Yes_What_type/screening__diabetes__bp'
            ]
            available_preventive = [c for c in preventive_cols if c in df.columns]
            if available_preventive:
                households_with_preventive = 0
                for _, row in df.iterrows():
                    has_any_preventive = False
                    for col in available_preventive:
                        value = row[col] if col in row.index else None
                        if pd.notna(value) and ((isinstance(value, (int, float)) and value == 1) or (isinstance(value, str) and value.strip() == '1')):
                            has_any_preventive = True
                            break
                    if has_any_preventive:
                        households_with_preventive += 1
                
                preventive_rate = (households_with_preventive / len(df) * 100) if len(df) > 0 else 0
                st.metric(
                    "Preventive Care Uptake",
                    f"{preventive_rate:.1f}%",
                    f"{households_with_preventive} households",
                    help="Percentage of households using preventive healthcare services"
                )
        
        with col4:
            # Healthcare Accessibility Score (based on distance)
            if 'How_far_is_the_neare_healthcare_facility' in df.columns:
                distance_data = df['How_far_is_the_neare_healthcare_facility'].value_counts()
                accessible_count = 0  # Within 5km
                total_responses = distance_data.sum()
                
                for distance, count in distance_data.items():
                    if pd.notna(distance) and ('Less than 1 km' in str(distance) or '1-5 km' in str(distance)):
                        accessible_count += count
                
                accessibility_score = (accessible_count / total_responses * 100) if total_responses > 0 else 0
                st.metric(
                    "Healthcare Accessibility",
                    f"{accessibility_score:.1f}%",
                    f"{accessible_count} within 5km",
                    help="Percentage of households within 5km of healthcare facilities"
                )
        
        # Row 3: Environmental & Lifestyle Health
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Water Safety Score
            if 'Do_you_treat_water_before_drinking' in df.columns:
                water_yes = df['Do_you_treat_water_before_drinking'].str.contains('yes', case=False, na=False).sum()
                water_total = df['Do_you_treat_water_before_drinking'].count()
                water_safety_score = (water_yes / water_total * 100) if water_total > 0 else 0
                st.metric(
                    "Water Safety Practices",
                    f"{water_safety_score:.1f}%",
                    f"{water_yes} of {water_total} households",
                    help="Percentage of households treating water before drinking"
                )
        
        with col2:
            # UV Protection Rate
            if 'Do_you_use_sunscreen_ective_eyewear_daily' in df.columns:
                uv_yes = df['Do_you_use_sunscreen_ective_eyewear_daily'].str.contains('yes', case=False, na=False).sum()
                uv_total = df['Do_you_use_sunscreen_ective_eyewear_daily'].count()
                uv_protection_rate = (uv_yes / uv_total * 100) if uv_total > 0 else 0
                st.metric(
                    "UV Protection Usage",
                    f"{uv_protection_rate:.1f}%",
                    f"{uv_yes} of {uv_total} households",
                    help="Percentage of households using daily UV protection"
                )
        
        with col3:
            # Dental Health Risk - Count households with ANY dental issue
            dental_problems = [
                'Have_you_experienced_the_past_six_months/tooth_pain_or_sensitivity',
                'Have_you_experienced_the_past_six_months/bleeding_gums',
                'Have_you_experienced_the_past_six_months/cavities_or_tooth_decay'
            ]
            
            households_with_dental_issues = 0
            for _, row in df.iterrows():
                has_dental_issue = False
                for col in dental_problems:
                    if col in df.columns:
                        value = row[col] if col in row.index else None
                        if pd.notna(value) and ((isinstance(value, (int, float)) and value == 1) or (isinstance(value, str) and value.strip() == '1')):
                            has_dental_issue = True
                            break
                if has_dental_issue:
                    households_with_dental_issues += 1
            
            dental_risk_rate = (households_with_dental_issues / len(df) * 100) if len(df) > 0 else 0
            st.metric(
                "Dental Health Issues",
                f"{dental_risk_rate:.1f}%",
                f"{households_with_dental_issues} households affected",
                help="Percentage of households reporting any dental health issues"
            )
        
        with col4:
            # Mental Health Stress Level - Count households with ANY stress factor
            stress_cols = [col for col in df.columns if 'What_are_the_biggest_stress_in_your_life' in col and '/' in col]
            if stress_cols:
                households_with_stress = 0
                for _, row in df.iterrows():
                    has_stress = False
                    for col in stress_cols:
                        if col in df.columns:
                            value = row[col] if col in row.index else None
                            if pd.notna(value) and ((isinstance(value, (int, float)) and value == 1) or (isinstance(value, str) and value.strip() == '1')):
                                has_stress = True
                                break
                    if has_stress:
                        households_with_stress += 1
                
                stress_rate = (households_with_stress / len(df)) * 100
                st.metric(
                    "Community Stress Level",
                    f"{stress_rate:.1f}%",
                    f"{households_with_stress} households affected",
                    help="Percentage of households reporting significant stress factors"
                )
        
        # Health Priorities Dashboard
        st.markdown("## 🏥 Health Priorities Dashboard")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if insights['health_priorities']:
                # Top health conditions chart
                conditions = [item['condition'] for item in insights['health_priorities']]
                prevalences = [item['prevalence'] for item in insights['health_priorities']]
                severities = [item['severity'] for item in insights['health_priorities']]
                
                # Color mapping for severity
                color_map = {'High': '#d32f2f', 'Moderate': '#f57c00', 'Low': '#388e3c'}
                colors = [color_map[severity] for severity in severities]
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=prevalences,
                        y=conditions,
                        orientation='h',
                        marker_color=colors,
                        text=[f"{p:.1f}%" for p in prevalences],
                        textposition='outside',
                        name='Prevalence'
                    )
                ])
                
                fig.update_layout(
                    title="Top 5 Health Priorities by Prevalence",
                    xaxis_title="Prevalence (%)",
                    yaxis_title="Health Conditions",
                    height=400,
                    margin=dict(l=150)
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if insights['infrastructure_gaps']:
                # Healthcare barriers chart
                barriers = [item['barrier'] for item in insights['infrastructure_gaps']]
                barrier_prevalences = [item['prevalence'] for item in insights['infrastructure_gaps']]
                
                fig = px.pie(
                    values=barrier_prevalences,
                    names=barriers,
                    title="Primary Healthcare Access Barriers",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                
                fig.update_traces(textinfo='percent+label', textposition='inside')
                fig.update_layout(height=400)
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Critical Findings Alert Box
        if insights['key_findings']:
            st.markdown("## 🚨 Critical Findings")
            for finding in insights['key_findings']:
                if finding['urgency'] == 'Critical':
                    st.error(f"**{finding['finding']}**: {finding['description']}")
                elif finding['impact'] == 'High':
                    st.warning(f"**{finding['finding']}**: {finding['description']}")
                else:
                    st.info(f"**{finding['finding']}**: {finding['description']}")
        
        # Trends Analysis
        st.markdown("## 📈 Trend Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Demographic Insights")
            if trends['demographic_trends']:
                for trend in trends['demographic_trends']:
                    st.markdown(f"**{trend['category']}**: {trend['pattern']} ({trend['percentage']})")
                    st.caption(trend['insight'])
            
            if insights['demographic_patterns']:
                for pattern in insights['demographic_patterns']:
                    st.markdown(f"**{pattern['pattern']}**: {pattern['description']}")
                    st.caption(f"Top performing group: {pattern['top_group']} ({pattern['utilization_rate']})")
        
        with col2:
            st.markdown("### Geographic Insights")
            if insights['geographic_insights']:
                for geo_insight in insights['geographic_insights']:
                    st.markdown(f"**{geo_insight['insight']}**: {geo_insight['description']}")
                    if geo_insight.get('villages'):
                        villages_str = ', '.join(geo_insight['villages'])
                        st.caption(f"Affected villages: {villages_str}")
                    st.caption(f"Recommendation: {geo_insight.get('recommendation', 'Monitor situation')}")
        
        # Anomaly Detection
        if any(anomalies.values()):
            st.markdown("## ⚠️ Anomalies Detected")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if anomalies['health_anomalies']:
                    st.markdown("### Health Pattern Anomalies")
                    for anomaly in anomalies['health_anomalies']:
                        icon = "🔴" if anomaly['type'] == 'Unusually High' else "🟡"
                        st.markdown(f"{icon} **{anomaly['condition']}**: {anomaly['description']} ({anomaly['prevalence']:.1f}%)")
            
            with col2:
                if anomalies['geographic_anomalies']:
                    st.markdown("### Geographic Response Anomalies")
                    for anomaly in anomalies['geographic_anomalies']:
                        st.warning(f"**Low Participation**: {', '.join(anomaly['villages'][:3])}")
                
                # Show risk factors if any
                if insights['risk_factors']:
                    st.markdown("### Identified Risk Factors")
                    for risk in insights['risk_factors']:
                        st.error(f"🚩 **{risk['factor']}**: {risk['prevalence']:.1f}% prevalence")
        
        # Executive Recommendations
        st.markdown("## 🎯 Priority Recommendations")
        
        if insights['recommendations']:
            # Group recommendations by priority
            high_priority = [r for r in insights['recommendations'] if r['priority'] == 'High']
            critical_priority = [r for r in insights['recommendations'] if r['priority'] == 'Critical']
            medium_priority = [r for r in insights['recommendations'] if r['priority'] == 'Medium']
            
            if critical_priority:
                st.markdown("### 🔴 Critical Priority Actions")
                for rec in critical_priority:
                    with st.expander(f"{rec['title']} ({rec['timeline']})", expanded=True):
                        st.markdown(f"**Category**: {rec['category']}")
                        st.markdown(f"**Description**: {rec['description']}")
                        st.markdown("**Recommended Actions**:")
                        for action in rec['actions']:
                            st.markdown(f"• {action}")
                        st.markdown(f"**Expected Impact**: {rec['impact']}")
            
            if high_priority:
                st.markdown("### 🟠 High Priority Actions")
                for rec in high_priority:
                    with st.expander(f"{rec['title']} ({rec['timeline']})", expanded=False):
                        st.markdown(f"**Category**: {rec['category']}")
                        st.markdown(f"**Description**: {rec['description']}")
                        st.markdown("**Recommended Actions**:")
                        for action in rec['actions']:
                            st.markdown(f"• {action}")
                        st.markdown(f"**Expected Impact**: {rec['impact']}")
            
            if medium_priority:
                st.markdown("### 🟡 Medium Priority Actions")
                for rec in medium_priority:
                    with st.expander(f"{rec['title']} ({rec['timeline']})", expanded=False):
                        st.markdown(f"**Category**: {rec['category']}")
                        st.markdown(f"**Description**: {rec['description']}")
                        st.markdown("**Recommended Actions**:")
                        for action in rec['actions']:
                            st.markdown(f"• {action}")
                        st.markdown(f"**Expected Impact**: {rec['impact']}")
        
        # Summary Statistics
        st.markdown("## 📊 Executive Summary Statistics")
        
        # Create comprehensive summary chart
        col1, col2 = st.columns(2)
        
        with col1:
            # Health outcomes summary
            if insights['health_priorities']:
                health_data = pd.DataFrame(insights['health_priorities'])
                
                fig = px.treemap(
                    health_data,
                    path=['severity', 'condition'],
                    values='affected',
                    color='prevalence',
                    color_continuous_scale='Reds',
                    title="Health Conditions by Severity and Impact"
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Infrastructure needs heatmap
            if insights['infrastructure_gaps']:
                infra_data = pd.DataFrame(insights['infrastructure_gaps'])
                
                fig = px.bar(
                    infra_data,
                    x='prevalence',
                    y='barrier',
                    orientation='h',
                    color='affected',
                    color_continuous_scale='Blues',
                    title="Infrastructure Gaps by Impact",
                    labels={'prevalence': 'Prevalence (%)', 'barrier': 'Access Barrier'}
                )
                
                fig.update_layout(height=400, margin=dict(l=120))
                st.plotly_chart(fig, use_container_width=True)
        
        # Comprehensive Health Domain Analysis
        st.markdown("## 📊 Comprehensive Health Domain Analysis")
        
        # Create tabs for detailed domain analysis
        domain_tab1, domain_tab2, domain_tab3, domain_tab4 = st.columns(4)
        
        with domain_tab1:
            st.markdown("### 💧 Water & Sanitation")
            
            # Water safety metrics
            if 'Do_you_treat_water_before_drinking' in df.columns:
                water_yes = df['Do_you_treat_water_before_drinking'].str.contains('yes', case=False, na=False).sum()
                water_total = df['Do_you_treat_water_before_drinking'].count()
                water_safety_score = (water_yes / water_total * 100) if water_total > 0 else 0
                
                st.metric("Water Treatment Rate", f"{water_safety_score:.1f}%", f"{water_yes}/{water_total}")
                
                if water_safety_score < 80:
                    st.error("⚠️ Water Safety Risk")
                elif water_safety_score < 90:
                    st.warning("⚠️ Needs Improvement")
                else:
                    st.success("✅ Good Coverage")
            
            # Water sources
            water_source_cols = [col for col in df.columns if 'What_is_your_primary_rinking_water_source' in col and '/' in col]
            if water_source_cols:
                unsafe_sources = 0
                for col in water_source_cols:
                    if 'river' in col.lower() or 'lake' in col.lower():
                        unsafe_sources += df[col].sum() if df[col].dtype in ['int64', 'float64'] else df[col].value_counts().get(1, 0)
                
                if unsafe_sources > 0:
                    unsafe_pct = (unsafe_sources / len(df)) * 100
                    st.metric("Unsafe Water Sources", f"{unsafe_pct:.1f}%", f"{unsafe_sources} households")
        
        with domain_tab2:
            st.markdown("### ☀️ UV Protection & High-Altitude")
            
            # UV protection usage
            if 'Do_you_use_sunscreen_ective_eyewear_daily' in df.columns:
                uv_yes = df['Do_you_use_sunscreen_ective_eyewear_daily'].str.contains('yes', case=False, na=False).sum()
                uv_total = df['Do_you_use_sunscreen_ective_eyewear_daily'].count()
                uv_protection_rate = (uv_yes / uv_total * 100) if uv_total > 0 else 0
                
                st.metric("UV Protection Usage", f"{uv_protection_rate:.1f}%", f"{uv_yes}/{uv_total}")
                
                if uv_protection_rate < 50:
                    st.error("⚠️ Low Protection")
                elif uv_protection_rate < 75:
                    st.warning("⚠️ Needs Improvement")
                else:
                    st.success("✅ Good Protection")
            
            # UV-related health issues
            uv_issue_cols = ['Have_you_had_severe_UV_rays_or_eye_pain', 'Have_you_had_severe_indness_or_eye_pain']
            uv_problems = 0
            for col in uv_issue_cols:
                if col in df.columns:
                    uv_problems += df[col].str.contains('yes', case=False, na=False).sum() if df[col].dtype == 'object' else df[col].sum()
            
            if uv_problems > 0:
                uv_problem_rate = (uv_problems / len(df)) * 100
                st.metric("UV-Related Issues", f"{uv_problem_rate:.1f}%", f"{uv_problems} cases")
        
        with domain_tab3:
            st.markdown("### 🦷 Dental Health")
            
            # Dental health problems - Count households with ANY dental issue
            dental_data = get_dental_health_data(df)
            if dental_data:
                dental_problem_cols = [
                    'Have_you_experienced_the_past_six_months/tooth_pain_or_sensitivity',
                    'Have_you_experienced_the_past_six_months/bleeding_gums',
                    'Have_you_experienced_the_past_six_months/swollen_or_red_gums',
                    'Have_you_experienced_the_past_six_months/loose_or_missing_teeth',
                    'Have_you_experienced_the_past_six_months/bad_breath_or__halitosis',
                    'Have_you_experienced_the_past_six_months/cavities_or_tooth_decay',
                    'Have_you_experienced_the_past_six_months/mouth_sores_or_ulcers'
                ]
                
                households_with_dental_issues = 0
                for _, row in df.iterrows():
                    has_dental_issue = False
                    for col in dental_problem_cols:
                        if col in df.columns:
                            value = row[col] if col in row.index else None
                            if pd.notna(value) and ((isinstance(value, (int, float)) and value == 1) or (isinstance(value, str) and value.strip() == '1')):
                                has_dental_issue = True
                                break
                    if has_dental_issue:
                        households_with_dental_issues += 1
                
                dental_prevalence = (households_with_dental_issues / len(df) * 100) if len(df) > 0 else 0
                
                st.metric("Dental Health Issues", f"{dental_prevalence:.1f}%", f"{households_with_dental_issues} households")
                
                if dental_prevalence > 60:
                    st.error("⚠️ High Risk")
                elif dental_prevalence > 40:
                    st.warning("⚠️ Moderate Risk")
                else:
                    st.success("✅ Low Risk")
                
                # Most common dental issue
                if dental_data:
                    most_common_dental = max(dental_data.items(), key=lambda x: x[1]['count'])
                    st.caption(f"Most common: {most_common_dental[0]} ({most_common_dental[1]['percentage']:.1f}%)")
            
            # Dental care frequency
            if 'How_often_do_you_brush_your_teeth' in df.columns:
                daily_brushers = df['How_often_do_you_brush_your_teeth'].str.contains('daily', case=False, na=False).sum()
                brushing_total = df['How_often_do_you_brush_your_teeth'].count()
                daily_brushing_rate = (daily_brushers / brushing_total * 100) if brushing_total > 0 else 0
                st.metric("Daily Brushing", f"{daily_brushing_rate:.1f}%", f"{daily_brushers}/{brushing_total}")
        
        with domain_tab4:
            st.markdown("### 👶 Maternal & Child Health")
            
            # Antenatal care
            if 'Have_you_or_any_wome_d_any_antenatal_care' in df.columns:
                anc_yes = df['Have_you_or_any_wome_d_any_antenatal_care'].str.contains('yes', case=False, na=False).sum()
                anc_total = df['Have_you_or_any_wome_d_any_antenatal_care'].count()
                anc_percentage = (anc_yes / anc_total * 100) if anc_total > 0 else 0
                
                st.metric("Antenatal Care", f"{anc_percentage:.1f}%", f"{anc_yes}/{anc_total}")
                
                if anc_percentage > 90:
                    st.success("✅ Excellent")
                elif anc_percentage > 70:
                    st.success("✅ Good")
                else:
                    st.warning("⚠️ Needs Improvement")
            
            # Child vaccination
            if 'Are_children_in_your_old_fully_vaccinated' in df.columns:
                vacc_yes = df['Are_children_in_your_old_fully_vaccinated'].str.contains('yes', case=False, na=False).sum()
                vacc_total = df['Are_children_in_your_old_fully_vaccinated'].count()
                vacc_percentage = (vacc_yes / vacc_total * 100) if vacc_total > 0 else 0
                
                st.metric("Child Vaccination", f"{vacc_percentage:.1f}%", f"{vacc_yes}/{vacc_total}")
                
                if vacc_percentage > 95:
                    st.success("✅ Excellent")
                elif vacc_percentage > 85:
                    st.success("✅ Good")
                else:
                    st.error("⚠️ Critical Gap")
            
            # Home deliveries
            home_delivery_cols = [col for col in df.columns if 'Where_do_women_in_yo_household_give_birth' in col and 'home' in col.lower()]
            if home_delivery_cols:
                home_deliveries = 0
                for col in home_delivery_cols:
                    home_deliveries += df[col].sum() if df[col].dtype in ['int64', 'float64'] else df[col].value_counts().get(1, 0)
                
                if home_deliveries > 0:
                    home_delivery_pct = (home_deliveries / len(df)) * 100
                    st.metric("Home Deliveries", f"{home_delivery_pct:.1f}%", f"{home_deliveries} cases")
        
        # Action Items Export
        st.markdown("## 📋 Action Items Summary")
        
        if insights['recommendations']:
            # Create actionable summary table
            action_items = []
            for rec in insights['recommendations']:
                action_items.append({
                    'Priority': rec['priority'],
                    'Category': rec['category'],
                    'Action': rec['title'],
                    # 'Timeline': rec['timeline'],
                    'Impact': rec['impact']
                })
            
            action_df = pd.DataFrame(action_items)
            action_df = action_df.sort_values(['Priority'], key=lambda x: x.map({'Critical': 0, 'High': 1, 'Medium': 2}))
            
            st.dataframe(
                action_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Priority': st.column_config.TextColumn('Priority', width='small'),
                    'Category': st.column_config.TextColumn('Category', width='medium'),
                    'Action': st.column_config.TextColumn('Action', width='large'),
                    # 'Timeline': st.column_config.TextColumn('Timeline', width='small', hide_index=True),
                    'Impact': st.column_config.TextColumn('Impact', width='small')
                }
            )
        
        # PDF Export Section
        st.markdown("## 📥 Export Report")
        
        # Two-column layout for different PDF options
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📋 Executive Summary PDF")
            if st.button("📄 Download Executive Summary", type="secondary", use_container_width=True):
                try:
                    # Generate PDF
                    with st.spinner("Generating Executive Summary PDF..."):
                        pdf_data = generate_pdf_report(insights, trends, anomalies, df)
                    
                    # Create download button
                    b64 = base64.b64encode(pdf_data).decode()
                    current_date = datetime.now().strftime("%Y%m%d")
                    filename = f"Ladakh_Healthcare_Executive_Summary_{current_date}.pdf"
                    
                    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Click here if download does not start automatically</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                    # Auto-download using Streamlit's download_button
                    st.download_button(
                        label="📥 Download Executive PDF",
                        data=pdf_data,
                        file_name=filename,
                        mime="application/pdf",
                        type="primary",
                        use_container_width=True
                    )
                    
                    st.success(f"✅ Executive Summary PDF generated successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error generating PDF: {str(e)}")
                    st.info("💡 Tip: Make sure you have the required PDF libraries installed.")
            
            st.caption("📄 Includes executive insights, key findings, recommendations, and summary charts.")
        
        with col2:
            st.markdown("### 📊 Complete Dashboard PDF")
            if st.button("📊 Download ALL Graphs from ALL Tabs", type="secondary", use_container_width=True):
                try:
                    # Generate comprehensive PDF with ALL graphs
                    with st.spinner("Generating Complete Dashboard PDF... This may take a moment as we include ALL graphs from ALL tabs..."):
                        pdf_data = generate_all_tabs_pdf_report(insights, trends, anomalies, df)
                    
                    # Create download button
                    b64 = base64.b64encode(pdf_data).decode()
                    current_date = datetime.now().strftime("%Y%m%d")
                    filename = f"Ladakh_Healthcare_Complete_Dashboard_{current_date}.pdf"
                    
                    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Click here if download does not start automatically</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                    # Auto-download using Streamlit's download_button
                    st.download_button(
                        label="📥 Download Complete PDF",
                        data=pdf_data,
                        file_name=filename,
                        mime="application/pdf",
                        type="secondary",
                        use_container_width=True
                    )
                    
                    st.success(f"✅ Complete Dashboard PDF generated successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error generating comprehensive PDF: {str(e)}")
                    st.info("💡 Note: The complete dashboard PDF includes 25+ charts and may take longer to generate.")
            
            st.caption("📊 Includes ALL graphs from Disease, Healthcare Access, Maternal Health, Mental Health, Infrastructure, Dental, Demographic, High-Altitude, and Village Analysis tabs.")
        
    
    with tab1:
        st.markdown('<h2 class="section-header">Disease Prevalence Analysis</h2>', unsafe_allow_html=True)
        
        # Chronic conditions analysis
        col1, col2 = st.columns(2)
        
        with col1:
            chronic_conditions = get_chronic_conditions_data(df)
            if chronic_conditions:
                ascending = sort_order_control('chronic_conditions')
                names = list(chronic_conditions.keys())
                values = [v['count'] for v in chronic_conditions.values()]
                sorted_names, sorted_values = sort_names_values(names, values, ascending)
                
                fig = px.bar(
                    x=sorted_names,
                    y=sorted_values,
                    title=f"Chronic Conditions Prevalence (Total: {len(df)} households)",
                    labels={'x': 'Condition', 'y': 'Number of Cases'},
                    color=sorted_values,
                    color_continuous_scale='Reds'
                )
                
                # Create labels with correct percentages for sorted data
                labels = []
                for name, value in zip(sorted_names, sorted_values):
                    percentage = chronic_conditions[name]['percentage']
                    labels.append(f"{value}\n({percentage:.1f}%)")
                
                fig = add_bar_value_labels(fig, labels)
                fig.update_layout(xaxis_tickangle=-45, height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Recent health problems in past 6 months
            recent_health_cols = [col for col in df.columns if 'In_the_past_6_months_any_of_the_following' in col and '/' in col]
            if recent_health_cols:
                recent_health_data = {}
                total_respondents = len(df)
                
                for col in recent_health_cols:
                    raw_name = col.split('/')[-1].replace('_', ' ').title()
                    condition_name = clean_recent_health_label(raw_name)
                    count = df[col].sum() if df[col].dtype in ['int64', 'float64'] else df[col].value_counts().get(1, 0)
                    
                    if count > 0 and condition_name.lower() not in ['in the past 6 months any of the following', 'nan', '']:
                        recent_health_data[f"{condition_name}\n({count}/{total_respondents})"] = count
                
                if recent_health_data:
                    fig = px.pie(
                        values=list(recent_health_data.values()),
                        names=list(recent_health_data.keys()),
                        title=f"Recent Health Problems (Past 6 Months) - Total: {len(df)} households"
                    )
                    fig = add_pie_value_labels(fig)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Community health issues
        st.markdown("### Community Health Issues")
        community_health_cols = [col for col in df.columns if 'What_are_the_most_co_es_in_your_community' in col and '/' in col]
        if community_health_cols:
            community_data = {}
            total_respondents = len(df)
            
            for col in community_health_cols:
                issue_name = col.split('/')[-1].replace('_', ' ').title()
                count = df[col].sum() if df[col].dtype in ['int64', 'float64'] else df[col].value_counts().get(1, 0)
                
                if count > 0 and issue_name.lower() not in ['what are the most co es in your community', 'nan', '']:
                    community_data[f"{issue_name}\n({count}/{total_respondents})"] = count
            
            if community_data:
                ascending = sort_order_control('community_issues')
                names = list(community_data.keys())
                values = list(community_data.values())
                sorted_names, sorted_values = sort_names_values(names, values, ascending)
                fig = px.bar(
                    x=sorted_values,
                    y=sorted_names,
                    orientation='h',
                    title=f"Most Common Community Health Issues (Total: {len(df)} households)",
                    labels={'x': 'Number of Reports', 'y': 'Health Issue'}
                )
                fig = add_bar_value_labels(fig, [str(v) for v in sorted_values])
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown('<h2 class="section-header">Healthcare Access Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Healthcare provider preferences
            provider_data = get_healthcare_providers_data(df)
            if provider_data:
                ascending = sort_order_control('provider_prefs')
                names = list(provider_data.keys())
                values = [v['count'] for v in provider_data.values()]
                sorted_names, sorted_values = sort_names_values(names, values, ascending)
                fig = px.bar(
                    x=sorted_names,
                    y=sorted_values,
                    title=f"Healthcare Provider Preferences (Total: {len(df)} households)",
                    labels={'x': 'Provider Type', 'y': 'Number of Users'},
                    color=sorted_values,
                    color_continuous_scale='Blues'
                )
                fig = add_bar_value_labels(fig, [f"{v}\n{provider_data[n]['percentage']:.1f}%" for n, v in zip(sorted_names, sorted_values)])
                fig.update_layout(xaxis_tickangle=-45, height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Distance to healthcare facility
            if 'How_far_is_the_neare_healthcare_facility' in df.columns:
                distance_data = df['How_far_is_the_neare_healthcare_facility'].value_counts()
                total_distance = distance_data.sum()
                fig = px.pie(
                    values=distance_data.values,
                    names=[f"{name}\n({value}/{total_distance})" for name, value in zip(distance_data.index, distance_data.values)],
                    title=f"Distance to Nearest Healthcare Facility ({total_distance}/{len(df)} households)"
                )
                fig = add_pie_value_labels(fig)
                st.plotly_chart(fig, use_container_width=True)
        
        # Reasons for healthcare provider choice (moved from Health Seeking Behavior)
        st.markdown("### Reasons for Healthcare Provider Choice")
        col1, col2 = st.columns(2)
        
        with col1:
            provider_reasons = get_provider_choice_reasons_data(df)
            if provider_reasons:
                ascending = sort_order_control('provider_reasons')
                names = list(provider_reasons.keys())
                values = [v['count'] for v in provider_reasons.values()]
                sorted_names, sorted_values = sort_names_values(names, values, ascending)
                fig = px.bar(
                    x=sorted_names,
                    y=sorted_values,
                    title=f"Provider Choice Reasons (Total: {len(df)} households)",
                    labels={'x': 'Reason', 'y': 'Number of Responses'}
                )
                fig = add_bar_value_labels(fig, [f"{v}\n{provider_reasons[n]['percentage']:.1f}%" for n, v in zip(sorted_names, sorted_values)])
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Healthcare barriers
            barriers_data = get_barriers_data(df)
            if barriers_data:
                ascending = sort_order_control('barriers')
                names = list(barriers_data.keys())
                values = [v['count'] for v in barriers_data.values()]
                sorted_names, sorted_values = sort_names_values(names, values, ascending)
                fig = px.bar(
                    x=sorted_names,
                    y=sorted_values,
                    title=f"Healthcare Access Barriers (Total: {len(df)} households)",
                    labels={'x': 'Barrier Type', 'y': 'Number of Reports'},
                    color=sorted_values,
                    color_continuous_scale='Reds'
                )
                fig = add_bar_value_labels(fig, [f"{v}\n{barriers_data[n]['percentage']:.1f}%" for n, v in zip(sorted_names, sorted_values)])
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

        # Cross-check: Distance category vs barrier reporting (distance/transport)
        if 'How_far_is_the_neare_healthcare_facility' in df.columns:
            st.markdown("### Distance vs Reported Access Barriers")
            cross_cols = [
                'What_are_the_biggest_accessing_healthcare/distance',
                'What_are_the_biggest_accessing_healthcare/lack_of_transport'
            ]
            available = [c for c in cross_cols if c in df.columns]
            if available:
                x_vals = []
                y_vals = []
                for dist_cat, group in df.groupby('How_far_is_the_neare_healthcare_facility'):
                    if pd.isna(dist_cat):
                        continue
                    total = len(group)
                    if total == 0:
                        continue
                    reports = 0
                    for c in available:
                        col = group[c]
                        reports += (col.sum() if col.dtype in ['int64','float64'] else col.value_counts().get(1,0))
                    pct = (reports / total * 100)
                    x_vals.append(dist_cat)
                    y_vals.append(pct)
                ascending = sort_order_control('distance_vs_barriers')
                x_sorted, y_sorted = sort_names_values(x_vals, y_vals, ascending)
                fig = px.bar(x=x_sorted, y=y_sorted, labels={'x': 'Distance Category', 'y': 'Barrier Reports (%)'}, title='Share reporting Distance/Transport Barriers by Distance Category')
                fig = add_bar_value_labels(fig, [f"{v:.1f}%" for v in y_sorted])
                fig.update_layout(xaxis_tickangle=-30)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown('<h2 class="section-header">Health Seeking Behavior</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        # Define preventive services with better names
        preventive_cols = [
            'If_Yes_What_type/vaccination',
            'If_Yes_What_type/regular_check_ups',
            'If_Yes_What_type/screening__diabetes__bp',
            'If_Yes_What_type/maternal__child_health_services'
        ]
        
        # Service name mapping for better readability
        service_name_map = {
            'Vaccination': 'Vaccination',
            'Regular Check Ups': 'Regular Health Check-ups',
            'Screening  Diabetes  Bp': 'Health Screening (Diabetes/BP)',
            'Maternal  Child Health Services': 'Maternal & Child Health Services'
        }
        
        def get_clean_service_name(raw_name):
            """Get clean, readable service names"""
            clean_name = raw_name.replace('_', ' ').title()
            return service_name_map.get(clean_name, clean_name)
        
        # Overall Preventive Services Summary
        st.markdown("### 📋 Preventive Healthcare Services Overview")
        
        # Calculate overall statistics
        available_cols = [c for c in preventive_cols if c in df.columns]
        if available_cols:
            # Calculate overall utilization
            service_stats = []
            total_households = len(df)
            
            for col in available_cols:
                service_name = get_clean_service_name(col.split('/')[-1])
                count = df[col].sum() if df[col].dtype in ['int64', 'float64'] else df[col].value_counts().get(1, 0)
                percentage = (count / total_households * 100) if total_households > 0 else 0
                service_stats.append({
                    'Service': service_name,
                    'Count': int(count),
                    'Total': total_households,
                    'Percentage': percentage
                })
            
            # Create summary bar chart
            col1, col2 = st.columns(2)
            
            with col1:
                if service_stats:
                    ascending = sort_order_control('preventive_services_summary')
                    names = [s['Service'] for s in service_stats]
                    values = [s['Count'] for s in service_stats]
                    percentages = [s['Percentage'] for s in service_stats]
                    
                    sorted_data = sorted(zip(names, values, percentages), key=lambda x: x[1], reverse=not ascending)
                    sorted_names, sorted_values, sorted_percentages = zip(*sorted_data)
                    
                    fig = px.bar(
                        x=list(sorted_names),
                        y=list(sorted_values),
                        title=f"Preventive Services Utilization (Total: {total_households} households)",
                        labels={'x': 'Preventive Service', 'y': 'Number of Households'}
                    )
                    
                    # Set distinct colors and remove legend
                    fig.update_traces(
                        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],  # Distinct colors
                        showlegend=False
                    )
                    
                    labels = [f"{v}\n({p:.1f}%)" for v, p in zip(sorted_values, sorted_percentages)]
                    fig = add_bar_value_labels(fig, labels)
                    fig.update_layout(xaxis_tickangle=-20, height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Summary metrics
                st.markdown("#### Key Statistics")
                
                # Calculate "Any Preventive" users
                df_any = df.copy()
                has_any_summary = pd.Series(False, index=df_any.index)
                for c in available_cols:
                    col = df_any[c]
                    if col.dtype in ['int64','float64']:
                        col_bool = col.fillna(0).astype(int) == 1
                    else:
                        col_str = col.astype(str).str.strip().str.lower()
                        col_bool = col_str.isin(['1','yes','true'])
                    has_any_summary = has_any_summary | col_bool
                
                total_with_any_preventive = has_any_summary.sum()
                pct_with_any = (total_with_any_preventive / total_households * 100) if total_households > 0 else 0
                
                st.metric(
                    "Households with Any Preventive Care",
                    f"{int(total_with_any_preventive):,}",
                    f"{pct_with_any:.1f}% of all households"
                )
                
                # Most popular service
                if service_stats:
                    most_popular = max(service_stats, key=lambda x: x['Count'])
                    st.metric(
                        "Most Popular Service",
                        most_popular['Service'],
                        f"{most_popular['Percentage']:.1f}% utilization"
                    )
                
                # Least utilized service
                if service_stats:
                    least_popular = min(service_stats, key=lambda x: x['Count'])
                    st.metric(
                        "Least Utilized Service",
                        least_popular['Service'],
                        f"{least_popular['Percentage']:.1f}% utilization"
                    )
        
        # Preventive Care Uptake by Demographics
        st.markdown("### 🎓 Preventive Care Uptake by Education Level")
        
        if 'Education_Level' in df.columns and available_cols:
            col1, col2 = st.columns(2)
            
            with col1:
                # Overall preventive care by education
                df_prev = df.copy()
                has_any = pd.Series(False, index=df_prev.index)
                for c in available_cols:
                    col = df_prev[c]
                    if col.dtype in ['int64','float64']:
                        col_bool = col.fillna(0).astype(int) == 1
                    else:
                        col_str = col.astype(str).str.strip().str.lower()
                        col_bool = col_str.isin(['1','yes','true'])
                    has_any = has_any | col_bool
                
                df_prev['Has_Any_Preventive'] = has_any.astype(int)
                
                # Calculate stats by education level
                edu_stats = []
                for edu_level in df_prev['Education_Level'].unique():
                    if pd.notna(edu_level):
                        edu_group = df_prev[df_prev['Education_Level'] == edu_level]
                        if len(edu_group) > 0:
                            count = edu_group['Has_Any_Preventive'].sum()
                            total = len(edu_group)
                            percentage = (count / total * 100) if total > 0 else 0
                            edu_stats.append({
                                'Education_Level': str(edu_level),
                                'Count': int(count),
                                'Total': total,
                                'Percentage': percentage
                            })
                
                if edu_stats:
                    ascending = sort_order_control('prev_any_vs_edu')
                    sorted_stats = sorted(edu_stats, key=lambda x: x['Count'], reverse=not ascending)
                    
                    names = [s['Education_Level'] for s in sorted_stats]
                    values = [s['Count'] for s in sorted_stats]
                    percentages = [s['Percentage'] for s in sorted_stats]
                    
                    fig = px.bar(
                        x=names,
                        y=percentages,
                        title='Overall Preventive Care Utilization by Education',
                        labels={'x': 'Education Level', 'y': 'Households with Any Preventive Care (%)'}
                    )
                    
                    # Set distinct green color and remove legend
                    fig.update_traces(
                        marker_color='#2ca02c',  # Green color
                        showlegend=False
                    )
                    
                    labels = [f"{v}/{s['Total']}\n({p:.1f}%)" for v, s, p in zip(values, sorted_stats, percentages)]
                    fig = add_bar_value_labels(fig, labels)
                    fig.update_layout(xaxis_tickangle=-30, height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Detailed service breakdown by education
                service_edu_data = []
                for edu_level in df['Education_Level'].unique():
                    if pd.notna(edu_level):
                        edu_group = df[df['Education_Level'] == edu_level]
                        if len(edu_group) > 0:
                            total = len(edu_group)
                            for col in available_cols:
                                service_name = get_clean_service_name(col.split('/')[-1])
                                count = edu_group[col].sum() if edu_group[col].dtype in ['int64','float64'] else edu_group[col].value_counts().get(1,0)
                                percentage = (count / total * 100) if total > 0 else 0
                                service_edu_data.append({
                                    'Education_Level': str(edu_level),
                                    'Service': service_name,
                                    'Count': int(count),
                                    'Total': total,
                                    'Percentage': percentage
                                })
                
                if service_edu_data:
                    service_df = pd.DataFrame(service_edu_data)
                    ascending2 = sort_order_control('prev_services_vs_edu')
                    order = service_df.groupby('Education_Level')['Percentage'].mean().sort_values(ascending=ascending2).index.tolist()
                    
                    fig = px.bar(
                        service_df,
                        x='Education_Level',
                        y='Percentage',
                        color='Service',
                        category_orders={'Education_Level': order},
                        barmode='group',
                        title='Detailed Service Utilization by Education Level',
                        labels={'Percentage': 'Utilization (%)', 'Education_Level': 'Education Level'}
                    )
                    
                    # Create labels for the grouped bars
                    labels = [f"{row['Count']}\n({row['Percentage']:.1f}%)" for _, row in service_df.iterrows()]
                    fig.update_traces(text=labels, textposition='outside', cliponaxis=False)
                    fig.update_layout(xaxis_tickangle=-30, height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Healthcare expenditure analysis - moved to separate section
        st.markdown("### 💰 Healthcare Consultation Costs")
        if 'How_much_do_you_usua_d_consultation_costs' in df.columns:
            cost_data = df['How_much_do_you_usua_d_consultation_costs'].value_counts()
            total_cost_respondents = cost_data.sum()
            
            ascending = sort_order_control('consultation_costs')
            names = list(cost_data.index)
            values = list(cost_data.values)
            names, values = sort_names_values(names, values, ascending)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = px.bar(
                    x=names,
                    y=values,
                    title=f"Healthcare Consultation Cost Distribution",
                    labels={'x': 'Cost Range', 'y': 'Number of Households'}
                )
                
                # Set distinct red color and remove legend
                fig.update_traces(
                    marker_color='#d62728',  # Red color
                    showlegend=False
                )
                
                fig = add_bar_value_labels(fig, [f"{v}\n({v/total_cost_respondents*100:.1f}%)" for v in values])
                fig.update_layout(xaxis_tickangle=-30, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Cost Summary")
                st.metric("Total Respondents", f"{total_cost_respondents:,}", f"{total_cost_respondents/len(df)*100:.1f}% of all households")
                
                # Most common cost range
                most_common_cost = cost_data.idxmax()
                most_common_count = cost_data.max()
                st.metric(
                    "Most Common Range",
                    most_common_cost,
                    f"{most_common_count/total_cost_respondents*100:.1f}% of respondents"
                )
        
        # Preventive Care by Occupation
        st.markdown("### 💼 Preventive Care Uptake by Occupation")
        
        if 'Occupation' in df.columns and available_cols:
            col1, col2 = st.columns(2)
            
            with col1:
                # Overall preventive care by occupation
                occ_stats = []
                for occ in df['Occupation'].unique():
                    if pd.notna(occ):
                        occ_group = df[df['Occupation'] == occ]
                        if len(occ_group) > 0:
                            total = len(occ_group)
                            
                            # Calculate "Any Preventive" for this occupation
                            any_preventive_count = 0
                            for _, row in occ_group.iterrows():
                                has_any = False
                                for c in available_cols:
                                    r = row[c]
                                    if pd.notna(r) and ((isinstance(r, (int,float)) and r==1) or (isinstance(r,str) and r.strip()=='1')):
                                        has_any = True
                                        break
                                if has_any:
                                    any_preventive_count += 1
                            
                            percentage = (any_preventive_count / total * 100) if total > 0 else 0
                            occ_stats.append({
                                'Occupation': str(occ),
                                'Count': any_preventive_count,
                                'Total': total,
                                'Percentage': percentage
                            })
                
                if occ_stats:
                    ascending = sort_order_control('prev_any_vs_occ')
                    sorted_stats = sorted(occ_stats, key=lambda x: x['Count'], reverse=not ascending)
                    
                    names = [s['Occupation'] for s in sorted_stats]
                    values = [s['Count'] for s in sorted_stats]
                    percentages = [s['Percentage'] for s in sorted_stats]
                    
                    fig = px.bar(
                        x=names,
                        y=percentages,
                        title='Overall Preventive Care Utilization by Occupation',
                        labels={'x': 'Occupation', 'y': 'Households with Any Preventive Care (%)'}
                    )
                    
                    # Set distinct orange color and remove legend
                    fig.update_traces(
                        marker_color='#ff7f0e',  # Orange color
                        showlegend=False
                    )
                    
                    labels = [f"{v}/{s['Total']}\n({p:.1f}%)" for v, s, p in zip(values, sorted_stats, percentages)]
                    fig = add_bar_value_labels(fig, labels)
                    fig.update_layout(xaxis_tickangle=-30, height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Service breakdown by occupation
                service_occ_data = []
                for occ in df['Occupation'].unique():
                    if pd.notna(occ):
                        occ_group = df[df['Occupation'] == occ]
                        if len(occ_group) > 0:
                            total = len(occ_group)
                            for col in available_cols:
                                service_name = get_clean_service_name(col.split('/')[-1])
                                count = occ_group[col].sum() if occ_group[col].dtype in ['int64','float64'] else occ_group[col].value_counts().get(1,0)
                                percentage = (count / total * 100) if total > 0 else 0
                                service_occ_data.append({
                                    'Occupation': str(occ),
                                    'Service': service_name,
                                    'Count': int(count),
                                    'Total': total,
                                    'Percentage': percentage
                                })
                
                if service_occ_data:
                    service_df = pd.DataFrame(service_occ_data)
                    ascending2 = sort_order_control('prev_services_vs_occ')
                    order = service_df.groupby('Occupation')['Percentage'].mean().sort_values(ascending=ascending2).index.tolist()
                    
                    fig = px.bar(
                        service_df,
                        x='Occupation',
                        y='Percentage',
                        color='Service',
                        category_orders={'Occupation': order},
                        barmode='group',
                        title='Detailed Service Utilization by Occupation',
                        labels={'Percentage': 'Utilization (%)', 'Occupation': 'Occupation'}
                    )
                    
                    # Create labels for the grouped bars
                    labels = [f"{row['Count']}\n({row['Percentage']:.1f}%)" for _, row in service_df.iterrows()]
                    fig.update_traces(text=labels, textposition='outside', cliponaxis=False)
                    fig.update_layout(xaxis_tickangle=-30, height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
        
        
        # Health information seeking behavior
        st.markdown("### 📚 Health Information Needs")
        info_cols = [col for col in df.columns if 'What_topics_you_like_more_information_on' in col and '/' in col]
        if info_cols:
            info_data = {}
            total_respondents = len(df)
            
            for col in info_cols:
                topic_name = col.split('/')[-1].replace('_', ' ').title()
                count = df[col].sum() if df[col].dtype in ['int64', 'float64'] else df[col].value_counts().get(1, 0)
                
                if count > 0:
                    info_data[f"{topic_name}\n({count}/{total_respondents})"] = count
            
            if info_data:
                ascending = sort_order_control('info_topics')
                names = list(info_data.keys())
                values = list(info_data.values())
                names, values = sort_names_values(names, values, ascending)
                fig = px.bar(
                    x=names,
                    y=values,
                    title=f"Health Information Topics of Interest (Total: {len(df)} households)",
                    labels={'x': 'Topic', 'y': 'Number of Requests'}
                )
                fig = add_bar_value_labels(fig, [str(v) for v in values])
                fig.update_layout(xaxis_tickangle=-45)
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
                    total_delivery = sum(delivery_data.values())
                    fig = px.pie(
                        values=list(delivery_data.values()),
                        names=list(delivery_data.keys()),
                        title=f"Place of Delivery ({total_delivery}/{len(df)} households)"
                    )
                    fig = add_pie_value_labels(fig)
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
                    ascending = sort_order_control('reasons_home_delivery')
                    names = list(home_delivery_data.keys())
                    values = list(home_delivery_data.values())
                    names, values = sort_names_values(names, values, ascending)
                    total_home_reasons = sum(values)
                    fig = px.bar(
                        x=names,
                        y=values,
                        title=f"Reasons for Home Delivery ({total_home_reasons}/{len(df)} households)",
                        labels={'x': 'Reason', 'y': 'Number of Cases'}
                    )
                    fig = add_bar_value_labels(fig, [str(v) for v in values])
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Age-appropriate child nutrition analysis
        st.markdown("### Age-Appropriate Child Nutrition")
        nutrition_data = get_child_nutrition_by_age_data(df)
        if nutrition_data:
            ascending = sort_order_control('child_nutrition')
            names = list(nutrition_data.keys())
            values = [v['count'] for v in nutrition_data.values()]
            names, values = sort_names_values(names, values, ascending)
            fig = px.bar(
                x=names,
                y=values,
                title=f"Age-Appropriate Child Feeding Practices (Total: {len(df)} households)",
                labels={'x': 'Feeding Practice', 'y': 'Number of Cases'}
            )
            fig = add_bar_value_labels(fig, [f"{v}\n{nutrition_data[n]['percentage']:.1f}%" for n, v in zip(names, values)])
            st.plotly_chart(fig, use_container_width=True)
        
        # Maternal health indicators
        col1, col2 = st.columns(2)
        
        with col1:
            # Antenatal care utilization
            if 'Have_you_or_any_wome_d_any_antenatal_care' in df.columns:
                anc_data = df['Have_you_or_any_wome_d_any_antenatal_care'].value_counts()
                total_anc_respondents = anc_data.sum()
                
                fig = px.pie(
                    values=anc_data.values,
                    names=[f"{name}\n({value}/{total_anc_respondents})" for name, value in zip(anc_data.index, anc_data.values)],
                    title=f"Antenatal Care Utilization ({total_anc_respondents}/{len(df)} households)"
                )
                fig = add_pie_value_labels(fig)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Child vaccination status
            if 'Are_children_in_your_old_fully_vaccinated' in df.columns:
                vacc_data = df['Are_children_in_your_old_fully_vaccinated'].value_counts()
                total_vacc_respondents = vacc_data.sum()
                
                fig = px.pie(
                    values=vacc_data.values,
                    names=[f"{name}\n({value}/{total_vacc_respondents})" for name, value in zip(vacc_data.index, vacc_data.values)],
                    title=f"Child Vaccination Status ({total_vacc_respondents}/{len(df)} households)"
                )
                fig = add_pie_value_labels(fig)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.markdown('<h2 class="section-header">Mental Health & Substance Use</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Stress sources
            stress_cols = [col for col in df.columns if 'What_are_the_biggest_stress_in_your_life' in col and '/' in col]
            if stress_cols:
                stress_data = {}
                total_respondents = len(df)
                for col in stress_cols:
                    stress_name = col.split('/')[-1].replace('_', ' ').title()
                    count = df[col].sum() if df[col].dtype in ['int64', 'float64'] else df[col].value_counts().get(1, 0)
                    stress_data[stress_name] = count
                
                # Filter out zero values and invalid entries
                stress_data = {k: v for k, v in stress_data.items() if v > 0 and k.lower() not in ['what are the biggest stress in your life', 'nan', '']}
                
                if stress_data:
                    ascending = sort_order_control('stress_sources')
                    names = list(stress_data.keys())
                    values = list(stress_data.values())
                    sorted_names, sorted_values = sort_names_values(names, values, ascending)
                    fig = px.bar(
                        x=sorted_names,
                        y=sorted_values,
                        title=f"Major Sources of Stress (Total: {total_respondents} households)",
                        labels={'x': 'Stress Source', 'y': 'Number of Cases'}
                    )
                    fig = add_bar_value_labels(fig, [f"{v}\n({v/total_respondents*100:.1f}%)" for v in sorted_values])
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Substance use
            substance_cols = ['Tobacco', 'Alcohol', 'Chewing_betel_nut_Pan_Gutka_etc']
            substance_data = {}
            total_respondents = len(df)
            for col in substance_cols:
                if col in df.columns:
                    substance_name = col.replace('_', ' ').title()
                    count = df[col].value_counts().get('yes', 0) + df[col].value_counts().get('yes__frequently', 0) + df[col].value_counts().get('yes__occasionally', 0)
                    substance_data[substance_name] = count
            
            # Filter out zero values
            substance_data = {k: v for k, v in substance_data.items() if v > 0}
            
            if substance_data:
                ascending = sort_order_control('substance_use')
                names = list(substance_data.keys())
                values = list(substance_data.values())
                sorted_names, sorted_values = sort_names_values(names, values, ascending)
                fig = px.bar(
                    x=sorted_names,
                    y=sorted_values,
                    title=f"Substance Use Patterns (Total: {total_respondents} households)",
                    labels={'x': 'Substance', 'y': 'Number of Users'}
                )
                fig = add_bar_value_labels(fig, [f"{v}\n({v/total_respondents*100:.1f}%)" for v in sorted_values])
                st.plotly_chart(fig, use_container_width=True)
    
    with tab6:
        st.markdown('<h2 class="section-header">Infrastructure & Service Needs</h2>', unsafe_allow_html=True)
        
        # Healthcare service improvement needs
        improvement_cols = [col for col in df.columns if 'What_healthcare_serv_nk_needs_improvement' in col and '/' in col]
        if improvement_cols:
            improvement_data = {}
            total_respondents = len(df)
            
            for col in improvement_cols:
                service_name = col.split('/')[-1].replace('_', ' ').title()
                count = df[col].sum() if df[col].dtype in ['int64', 'float64'] else df[col].value_counts().get(1, 0)
                
                if count > 0:
                    improvement_data[f"{service_name}\n({count}/{total_respondents})"] = count
            
            if improvement_data:
                fig = px.bar(
                    x=list(improvement_data.keys()),
                    y=list(improvement_data.values()),
                    title=f"Healthcare Services Needing Improvement (Total: {len(df)} households)",
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
                    fig = add_pie_value_labels(fig)
                    st.plotly_chart(fig, use_container_width=True)

        # Water treatment analysis and cross-tabs
        st.markdown("### Water Treatment Methods")
        if 'Do_you_treat_water_before_drinking' in df.columns:
            treat_counts = df['Do_you_treat_water_before_drinking'].value_counts()
            total_treat = treat_counts.sum()
            ascending = sort_order_control('water_treatment')
            names = list(treat_counts.index.astype(str))
            values = list(treat_counts.values)
            names, values = sort_names_values(names, values, ascending)
            fig = px.bar(x=names, y=values, title=f"Water Treatment Responses ({total_treat}/{len(df)} households)", labels={'x': 'Response', 'y': 'Number of Households'})
            fig = add_bar_value_labels(fig, [str(v) for v in values])
            st.plotly_chart(fig, use_container_width=True)

            # Cross-tab vs Education
            if 'Education_Level' in df.columns:
                st.markdown("#### Treatment vs Education Level (%)")
                edu_groups = []
                for edu, g in df.groupby('Education_Level'):
                    if pd.isna(edu):
                        continue
                    total = len(g)
                    if total == 0:
                        continue
                    yes_count = 0
                    col = g['Do_you_treat_water_before_drinking']
                    if col.dtype == 'object':
                        yes_count = col.str.contains('yes', case=False, na=False).sum()
                    else:
                        yes_count = col.value_counts().get(1, 0)
                    edu_groups.append((str(edu), yes_count / total * 100))
                if edu_groups:
                    names2, values2 = zip(*edu_groups)
                    asc = sort_order_control('treat_vs_edu')
                    names2, values2 = sort_names_values(list(names2), list(values2), asc)
                    fig = px.bar(x=names2, y=values2, labels={'x': 'Education Level', 'y': 'Treats Water (%)'})
                    fig = add_bar_value_labels(fig, [f"{v:.1f}%" for v in values2])
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)

            # Cross-tab vs Occupation
            if 'Occupation' in df.columns:
                st.markdown("#### Treatment vs Occupation (%)")
                occ_groups = []
                for occ, g in df.groupby('Occupation'):
                    if pd.isna(occ):
                        continue
                    total = len(g)
                    if total == 0:
                        continue
                    col = g['Do_you_treat_water_before_drinking']
                    yes_count = col.str.contains('yes', case=False, na=False).sum() if col.dtype == 'object' else col.value_counts().get(1, 0)
                    occ_groups.append((str(occ), yes_count / total * 100))
                if occ_groups:
                    names3, values3 = zip(*occ_groups)
                    asc = sort_order_control('treat_vs_occ')
                    names3, values3 = sort_names_values(list(names3), list(values3), asc)
                    fig = px.bar(x=names3, y=values3, labels={'x': 'Occupation', 'y': 'Treats Water (%)'})
                    fig = add_bar_value_labels(fig, [f"{v:.1f}%" for v in values3])
                    fig.update_layout(xaxis_tickangle=-45)
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
                    fig = add_pie_value_labels(fig)
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab7:
        st.markdown('<h2 class="section-header">Dental Health Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Dental health problems
            dental_data = get_dental_health_data(df)
            if dental_data:
                ascending = sort_order_control('dental_problems')
                names = list(dental_data.keys())
                values = [v['count'] for v in dental_data.values()]
                names, values = sort_names_values(names, values, ascending)
                fig = px.bar(
                    x=names,
                    y=values,
                    title=f"Dental Problems (Past 6 Months) - Total: {len(df)} households",
                    labels={'x': 'Dental Problem', 'y': 'Number of Cases'},
                    color=values,
                    color_continuous_scale='Oranges'
                )
                fig = add_bar_value_labels(fig, [f"{v}\n{dental_data[n]['percentage']:.1f}%" for n, v in zip(names, values)])
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Dental care behaviors
            if 'How_often_do_you_brush_your_teeth' in df.columns:
                brushing_data = df['How_often_do_you_brush_your_teeth'].value_counts()
                total_brushing_respondents = brushing_data.sum()
                
                fig = px.pie(
                    values=brushing_data.values,
                    names=[f"{name}\n({value}/{total_brushing_respondents})" for name, value in zip(brushing_data.index, brushing_data.values)],
                    title=f"Tooth Brushing Frequency ({total_brushing_respondents}/{len(df)} households)"
                )
                fig = add_pie_value_labels(fig)
                st.plotly_chart(fig, use_container_width=True)
        
        # Last dental checkup
        if 'When_was_your_last_dental_check_up' in df.columns:
            checkup_data = df['When_was_your_last_dental_check_up'].value_counts()
            total_checkup_respondents = checkup_data.sum()
            
            ascending = sort_order_control('last_dental_checkup')
            names = list(checkup_data.index)
            values = list(checkup_data.values)
            names, values = sort_names_values(names, values, ascending)
            fig = px.bar(
                x=names,
                y=values,
                title=f"Last Dental Checkup ({total_checkup_respondents}/{len(df)} households)",
                labels={'x': 'Time Since Last Checkup', 'y': 'Number of Households'}
            )
            fig = add_bar_value_labels(fig, [str(v) for v in values])
            st.plotly_chart(fig, use_container_width=True)
    
    with tab8:
        st.markdown('<h2 class="section-header">Demographic Cross-Analysis</h2>', unsafe_allow_html=True)
        
        # Education vs Healthcare Utilization
        st.markdown("### Education Level vs Healthcare Provider Choice")
        if 'Education_Level' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Education distribution
                edu_data = df['Education_Level'].value_counts()
                total_edu_respondents = edu_data.sum()
                asc = sort_order_control('education_distribution')
                names = list(edu_data.index)
                values = list(edu_data.values)
                names, values = sort_names_values(names, values, asc)
                fig = px.bar(
                    x=names,
                    y=values,
                    title=f"Education Level Distribution ({total_edu_respondents}/{len(df)} households)",
                    labels={'x': 'Education Level', 'y': 'Number of Households'}
                )
                fig = add_bar_value_labels(fig, [str(v) for v in values])
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Cross-tabulation: Education vs Provider Choice (Grouped Bar)
                provider_cols = [
                    'where_do_you_usually_re_when_you_are_sick/government_hospital_health_post',
                    'where_do_you_usually_re_when_you_are_sick/private_clinic',
                    'where_do_you_usually_re_when_you_are_sick/pharmacy',
                    'where_do_you_usually_re_when_you_are_sick/traditional_healer',
                    'where_do_you_usually_re_when_you_are_sick/home_remedies',
                    'where_do_you_usually_re_when_you_are_sick/i_do_not_seek_healthcare'
                ]
                if any(col in df.columns for col in provider_cols):
                    records = []
                    for edu, g in df.groupby('Education_Level'):
                        if pd.isna(edu):
                            continue
                        for colp in provider_cols:
                            if colp in df.columns:
                                provider_name = colp.split('/')[-1].replace('_', ' ').title()
                                count = g[colp].sum() if g[colp].dtype in ['int64','float64'] else g[colp].value_counts().get(1,0)
                                if count > 0:
                                    records.append({'Education_Level': str(edu), 'Provider': provider_name, 'Count': int(count)})
                    if records:
                        provider_df = pd.DataFrame(records)
                        asc = sort_order_control('edu_provider')
                        edu_order = provider_df.groupby('Education_Level')['Count'].sum().sort_values(ascending=asc).index.tolist()
                        fig = px.bar(
                            provider_df,
                            x='Education_Level',
                            y='Count',
                            color='Provider',
                            category_orders={'Education_Level': edu_order},
                            barmode='group',
                            title='Education Level vs Healthcare Provider Choice'
                        )
                        fig.update_layout(xaxis_tickangle=-45)
                        fig.update_traces(text=provider_df['Count'], textposition='outside', cliponaxis=False)
                        st.plotly_chart(fig, use_container_width=True)
        
        # Income vs Health Outcomes
        st.markdown("### Income Level vs Chronic Conditions")
        if 'Total_number_of_hous_ncome_monthly_Approx' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Income distribution (sortable)
                income_data = df['Total_number_of_hous_ncome_monthly_Approx'].value_counts()
                total_income_respondents = income_data.sum()
                
                # Clean income labels for better readability
                cleaned_income_data = {}
                for income, count in income_data.items():
                    cleaned_label = clean_income_label(income)
                    cleaned_income_data[cleaned_label] = count
                
                asc = sort_order_control('income_distribution')
                names = list(cleaned_income_data.keys())
                values = list(cleaned_income_data.values())
                
                # Custom sort order for income ranges
                income_order = ['Under ₹50K', '₹50K-₹100K', '₹100K-₹150K', '₹150K-₹200K', 'Over ₹200K', 'Not Specified']
                sorted_names, sorted_values = [], []
                
                if asc:
                    for income in income_order:
                        if income in cleaned_income_data:
                            sorted_names.append(income)
                            sorted_values.append(cleaned_income_data[income])
                else:
                    for income in reversed(income_order):
                        if income in cleaned_income_data:
                            sorted_names.append(income)
                            sorted_values.append(cleaned_income_data[income])
                
                fig = px.bar(
                    x=sorted_names,
                    y=sorted_values,
                    title=f"Income Distribution ({total_income_respondents}/{len(df)} households)",
                    labels={'x': 'Monthly Household Income', 'y': 'Number of Households'}
                )
                fig = add_bar_value_labels(fig, [f"{v}\n({v/total_income_respondents*100:.1f}%)" for v in sorted_values])
                fig.update_layout(xaxis_tickangle=-20)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Income vs Chronic Conditions with most prevalent annotation
                chronic_cols = [
                    'Do_you_have_diagnose_h_chronic_conditions/hypertension',
                    'Do_you_have_diagnose_h_chronic_conditions/diabetes',
                    'Do_you_have_diagnose_h_chronic_conditions/asthma',
                    'Do_you_have_diagnose_h_chronic_conditions/arthritis'
                ]
                available_chronic = [c for c in chronic_cols if c in df.columns]
                if available_chronic:
                    df_temp = df.copy()
                    df_temp['Has_Chronic_Condition'] = 0
                    for col in available_chronic:
                        df_temp['Has_Chronic_Condition'] = df_temp['Has_Chronic_Condition'] | (df_temp[col] == 1)
                    
                    # Get both percentage and absolute counts
                    income_groups = df_temp.groupby('Total_number_of_hous_ncome_monthly_Approx').agg({
                        'Has_Chronic_Condition': ['sum', 'count', 'mean']
                    }).round(2)
                    income_groups.columns = ['chronic_count', 'total_count', 'chronic_rate']
                    income_groups['chronic_pct'] = income_groups['chronic_rate'] * 100
                    
                    # Most prevalent condition per income group
                    most_map = {}
                    for income, g in df.groupby('Total_number_of_hous_ncome_monthly_Approx'):
                        if pd.isna(income) or len(g) == 0:
                            continue
                        best_name = 'None'
                        best_count = -1
                        for col in available_chronic:
                            cnt = g[col].sum() if g[col].dtype in ['int64','float64'] else g[col].value_counts().get(1,0)
                            if cnt > best_count:
                                best_count = cnt
                                best_name = col.split('/')[-1].replace('_',' ').title() if cnt > 0 else 'None'
                        most_map[clean_income_label(income)] = best_name
                    
                    # Clean income labels and prepare data
                    cleaned_data = []
                    for income in income_groups.index:
                        cleaned_label = clean_income_label(income)
                        cleaned_data.append({
                            'income': cleaned_label,
                            'percentage': income_groups.loc[income, 'chronic_pct'],
                            'count': int(income_groups.loc[income, 'chronic_count']),
                            'total': int(income_groups.loc[income, 'total_count'])
                        })
                    
                    # Sort using custom income order
                    income_order = ['Under ₹50K', '₹50K-₹100K', '₹100K-₹150K', '₹150K-₹200K', 'Over ₹200K', 'Not Specified']
                    asc = sort_order_control('income_vs_chronic')
                    
                    if asc:
                        sorted_data = [d for income in income_order for d in cleaned_data if d['income'] == income]
                    else:
                        sorted_data = [d for income in reversed(income_order) for d in cleaned_data if d['income'] == income]
                    
                    names = [d['income'] for d in sorted_data]
                    values = [d['percentage'] for d in sorted_data]
                    
                    fig = px.bar(
                        x=names, 
                        y=values, 
                        labels={'x': 'Monthly Household Income', 'y': 'Chronic Condition Prevalence (%)'}, 
                        title='Chronic Condition Prevalence by Income Level'
                    )
                    
                    # Create better labels with absolute numbers and most common condition
                    labels = []
                    for d in sorted_data:
                        most_common = most_map.get(d['income'], 'None')
                        labels.append(f"{d['count']}/{d['total']}\n({d['percentage']:.1f}%)\nMost: {most_common}")
                    
                    fig = add_bar_value_labels(fig, labels)
                    fig.update_layout(xaxis_tickangle=-20)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Gender vs Healthcare Barriers
        st.markdown("### Gender vs Healthcare Barriers")
        if 'Gender' in df.columns:
            barrier_cols = ['What_are_the_biggest_accessing_healthcare/distance',
                          'What_are_the_biggest_accessing_healthcare/cost',
                          'What_are_the_biggest_accessing_healthcare/cultural_beliefs']
            
            gender_barrier_data = []
            for gender in df['Gender'].unique():
                if pd.notna(gender):
                    gender_df = df[df['Gender'] == gender]
                    
                    for barrier_col in barrier_cols:
                        if barrier_col in df.columns:
                            barrier_name = barrier_col.split('/')[-1].replace('_', ' ').title()
                            count = gender_df[barrier_col].sum() if gender_df[barrier_col].dtype in ['int64', 'float64'] else gender_df[barrier_col].value_counts().get(1, 0)
                            total = len(gender_df)
                            
                            gender_barrier_data.append({
                                'Gender': gender,
                                'Barrier': barrier_name,
                                'Percentage': (count / total * 100) if total > 0 else 0,
                                'Count': count,
                                'Total': total
                            })
            
            if gender_barrier_data:
                barrier_df = pd.DataFrame(gender_barrier_data)
                asc = sort_order_control('gender_barriers')
                order = barrier_df.groupby('Barrier')['Percentage'].mean().sort_values(ascending=asc).index.tolist()
                fig = px.bar(
                    barrier_df,
                    x='Barrier',
                    y='Percentage',
                    color='Gender',
                    category_orders={'Barrier': order},
                    title="Healthcare Barriers by Gender (%)",
                    labels={'Percentage': 'Percentage Reporting Barrier'},
                    barmode='group'
                )
                fig.update_traces(text=barrier_df['Percentage'].round(1).astype(str)+'%', textposition='outside', cliponaxis=False)
                st.plotly_chart(fig, use_container_width=True)
        
        # Occupation vs Health Outcomes
        st.markdown("### Occupation vs Health Outcomes")
        if 'Occupation' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Occupation distribution (bar only)
                occ_data = df['Occupation'].value_counts()
                total_occ_respondents = occ_data.sum()
                asc = sort_order_control('occupation_distribution')
                names = list(occ_data.index)
                values = list(occ_data.values)
                names, values = sort_names_values(names, values, asc)
                fig = px.bar(
                    x=names,
                    y=values,
                    title=f"Occupation Distribution ({total_occ_respondents}/{len(df)} households)",
                    labels={'x': 'Occupation', 'y': 'Number of Households'}
                )
                fig = add_bar_value_labels(fig, [str(v) for v in values])
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Occupation vs Substance Use
                substance_cols = ['Tobacco', 'Alcohol']
                
                if any(col in df.columns for col in substance_cols):
                    occ_substance_data = []
                    
                    for occ in df['Occupation'].unique():
                        if pd.notna(occ):
                            occ_df = df[df['Occupation'] == occ]
                            
                            for substance in substance_cols:
                                if substance in df.columns:
                                    # Count yes responses
                                    yes_count = 0
                                    if occ_df[substance].dtype == 'object':
                                        yes_count = occ_df[substance].str.contains('yes', case=False, na=False).sum()
                                    else:
                                        yes_count = occ_df[substance].sum()
                                    
                                    total = len(occ_df)
                                    
                                    occ_substance_data.append({
                                        'Occupation': occ,
                                        'Substance': substance,
                                        'Percentage': (yes_count / total * 100) if total > 0 else 0,
                                        'Count': yes_count,
                                        'Total': total
                                    })
                    
                    if occ_substance_data:
                        substance_df = pd.DataFrame(occ_substance_data)
                        
                        asc2 = sort_order_control('substance_by_occupation')
                        order = substance_df.groupby('Occupation')['Percentage'].mean().sort_values(ascending=asc2).index.tolist()
                        fig = px.bar(
                            substance_df,
                            x='Occupation',
                            y='Percentage',
                            color='Substance',
                            category_orders={'Occupation': order},
                            title="Substance Use by Occupation (%)",
                            labels={'Percentage': 'Percentage Using Substance'},
                            barmode='group'
                        )
                        fig.update_traces(text=substance_df['Percentage'].round(1).astype(str)+'%', textposition='outside', cliponaxis=False)
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
        
        # Age Group vs Healthcare Utilization
        st.markdown("### Age Group vs Healthcare Utilization")
        
        # Create age groups based on existing age data
        if 'Basic_Demographics' in df.columns:
            df_temp = df.copy()
            df_temp['Age_Group'] = 'Unknown'
            
            # Convert age to numeric if it's a string
            age_col = 'Basic_Demographics'
            df_temp[age_col] = pd.to_numeric(df_temp[age_col], errors='coerce')
            
            # Create age groups
            df_temp.loc[df_temp[age_col] < 18, 'Age_Group'] = 'Under 18'
            df_temp.loc[(df_temp[age_col] >= 18) & (df_temp[age_col] < 60), 'Age_Group'] = '18-59'
            df_temp.loc[df_temp[age_col] >= 60, 'Age_Group'] = '60+'
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Age group distribution
                age_data = df_temp['Age_Group'].value_counts()
                total_age_respondents = age_data.sum()
                
                asc = sort_order_control('age_group_distribution')
                names = list(age_data.index)
                values = list(age_data.values)
                names, values = sort_names_values(names, values, asc)
                fig = px.bar(
                    x=names,
                    y=values,
                    title=f"Age Group Distribution ({total_age_respondents}/{len(df)} households)",
                    labels={'x': 'Age Group', 'y': 'Number of Households'}
                )
                fig = add_bar_value_labels(fig, [str(v) for v in values])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Age vs Healthcare Provider Choice
                provider_cols = [
                    'where_do_you_usually_re_when_you_are_sick/government_hospital_health_post',
                    'where_do_you_usually_re_when_you_are_sick/private_clinic',
                    'where_do_you_usually_re_when_you_are_sick/pharmacy',
                    'where_do_you_usually_re_when_you_are_sick/traditional_healer',
                    'where_do_you_usually_re_when_you_are_sick/home_remedies',
                    'where_do_you_usually_re_when_you_are_sick/i_do_not_seek_healthcare'
                ]
                
                if any(col in df.columns for col in provider_cols):
                    age_provider_data = []
                    
                    for age_group in df_temp['Age_Group'].unique():
                        if age_group != 'Unknown':
                            age_df = df_temp[df_temp['Age_Group'] == age_group]
                            
                            for provider_col in provider_cols:
                                if provider_col in df.columns:
                                    provider_name = provider_col.split('/')[-1].replace('_', ' ').title()
                                    count = age_df[provider_col].sum() if age_df[provider_col].dtype in ['int64', 'float64'] else age_df[provider_col].value_counts().get(1, 0)
                                    total = len(age_df)
                                    
                                    age_provider_data.append({
                                        'Age_Group': age_group,
                                        'Provider': provider_name,
                                        'Percentage': (count / total * 100) if total > 0 else 0,
                                        'Count': count,
                                        'Total': total
                                    })
                    
                    if age_provider_data:
                        provider_df = pd.DataFrame(age_provider_data)
                        fig = px.bar(
                            provider_df,
                            x='Age_Group',
                            y='Percentage',
                            color='Provider',
                            title="Healthcare Provider Choice by Age Group (%)",
                            labels={'Percentage': 'Percentage Using Provider'},
                            barmode='group'
                        )
                        fig.update_traces(text=provider_df['Percentage'].round(1).astype(str)+'%', textposition='outside', cliponaxis=False)
                        st.plotly_chart(fig, use_container_width=True)
        
        # Others Specify Analysis
        st.markdown("### Analysis of 'Others Specify' Responses")
        
        # Analyze occupation others
        if 'Other_Specify' in df.columns:
            occupation_others = analyze_others_responses(df, 'Occupation')
            
            if occupation_others:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(
                        x=list(occupation_others.keys()),
                        y=list(occupation_others.values()),
                        title="Categorized 'Other' Occupation Responses",
                        labels={'x': 'Category', 'y': 'Number of Responses'}
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Display categorized responses as a table
                    others_df = pd.DataFrame(list(occupation_others.items()), columns=['Category', 'Count'])
                    others_df['Percentage'] = (others_df['Count'] / others_df['Count'].sum() * 100).round(1)
                    st.markdown("**Categorized Others Responses:**")
                    st.dataframe(others_df, use_container_width=True)
    
    with tab9:
        st.markdown('<h2 class="section-header">High-Altitude Health</h2>', unsafe_allow_html=True)
        
        # UV protection usage and eye pain
        col1, col2 = st.columns(2)
        with col1:
            if 'Do_you_use_sunscreen_ective_eyewear_daily' in df.columns:
                uv_use = df['Do_you_use_sunscreen_ective_eyewear_daily'].value_counts()
                total = uv_use.sum()
                asc = sort_order_control('uv_protection_use')
                names = list(uv_use.index.astype(str))
                values = list(uv_use.values)
                names, values = sort_names_values(names, values, asc)
                fig = px.bar(x=names, y=values, title=f"Daily UV Protection Use ({total}/{len(df)} households)", labels={'x':'Response','y':'Households'})
                fig = add_bar_value_labels(fig, [str(v) for v in values])
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            if 'Have_you_had_severe_UV_rays_or_eye_pain' in df.columns:
                uv_eye = df['Have_you_had_severe_UV_rays_or_eye_pain'].value_counts()
                total = uv_eye.sum()
                asc = sort_order_control('uv_eye_pain')
                names = list(uv_eye.index.astype(str))
                values = list(uv_eye.values)
                names, values = sort_names_values(names, values, asc)
                fig = px.bar(x=names, y=values, title=f"Severe UV Exposure or Eye Pain ({total}/{len(df)} households)", labels={'x':'Response','y':'Households'})
                fig = add_bar_value_labels(fig, [str(v) for v in values])
                st.plotly_chart(fig, use_container_width=True)
        
        # Frostbite / numbness in fingers or toes
        col1, col2 = st.columns(2)
        with col1:
            if 'Have_you_experienced_tion_in_fingers_toes' in df.columns:
                frost = df['Have_you_experienced_tion_in_fingers_toes'].value_counts()
                total = frost.sum()
                asc = sort_order_control('frostbite_numbness')
                names = list(frost.index.astype(str))
                values = list(frost.values)
                names, values = sort_names_values(names, values, asc)
                fig = px.bar(x=names, y=values, title=f"Frostbite / Numbness ({total}/{len(df)} households)", labels={'x':'Response','y':'Households'})
                fig = add_bar_value_labels(fig, [str(v) for v in values])
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            if 'Have_you_suffered_fr_ng_climbing_injuries' in df.columns:
                climb = df['Have_you_suffered_fr_ng_climbing_injuries'].value_counts()
                total = climb.sum()
                asc = sort_order_control('climbing_injuries')
                names = list(climb.index.astype(str))
                values = list(climb.values)
                names, values = sort_names_values(names, values, asc)
                fig = px.bar(x=names, y=values, title=f"Climbing Injuries ({total}/{len(df)} households)", labels={'x':'Response','y':'Households'})
                fig = add_bar_value_labels(fig, [str(v) for v in values])
                st.plotly_chart(fig, use_container_width=True)
        
        # Snow blindness
        if 'Have_you_had_severe_indness_or_eye_pain' in df.columns:
            snow = df['Have_you_had_severe_indness_or_eye_pain'].value_counts()
            total = snow.sum()
            asc = sort_order_control('snow_blindness')
            names = list(snow.index.astype(str))
            values = list(snow.values)
            names, values = sort_names_values(names, values, asc)
            fig = px.bar(x=names, y=values, title=f"Snow Blindness or Severe Eye Pain ({total}/{len(df)} households)", labels={'x':'Response','y':'Households'})
            fig = add_bar_value_labels(fig, [str(v) for v in values])
            st.plotly_chart(fig, use_container_width=True)
        
        # Sleep issues
        if 'Do_you_frequently_ex_dizziness_faintness' in df.columns:
            # assuming dizziness/faintness as proxy for sleep or altitude sickness symptoms
            dizzy = df['Do_you_frequently_ex_dizziness_faintness'].value_counts()
            total = dizzy.sum()
            asc = sort_order_control('sleep_issues')
            names = list(dizzy.index.astype(str))
            values = list(dizzy.values)
            names, values = sort_names_values(names, values, asc)
            fig = px.bar(x=names, y=values, title=f"Dizziness/Faintness ({total}/{len(df)} households)", labels={'x':'Response','y':'Households'})
            fig = add_bar_value_labels(fig, [str(v) for v in values])
            st.plotly_chart(fig, use_container_width=True)
    with tab10:
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
                    fig = add_bar_value_labels(fig, [f"{v:.1f}%" for v in condition_data['Prevalence']])
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
                        fig = add_bar_value_labels(fig, [f"{v:.1f} km" for v in distance_df['Average_Distance']])
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
        <p>Built by Aditya with Streamlit and Plotly for interactive visualization</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 
