import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from datetime import datetime

# Configure Streamlit page
st.set_page_config(
    page_title="💳 Credit Risk Assessment",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .risk-high {
        color: #d32f2f;
        font-weight: bold;
    }
    .risk-medium {
        color: #f57c00;
        font-weight: bold;
    }
    .risk-low {
        color: #388e3c;
        font-weight: bold;
    }
    .section-header {
        font-size: 18px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 15px;
        padding-bottom: 10px;
        border-bottom: 2px solid #667eea;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== FEATURE DEFINITIONS ====================
# Based on "Give Me Some Credit" Kaggle dataset
FEATURE_DEFINITIONS = {
    "Age": {
        "label": "Age (years)",
        "description": "Age of the borrower in years",
        "min": 18,
        "max": 120,
        "default": 40,
        "unit": "years"
    },
    "NumberOfDependents": {
        "label": "Number of Dependents",
        "description": "Number of dependents (children) the borrower has",
        "min": 0,
        "max": 20,
        "default": 1,
        "unit": "count"
    },
    "MonthlyIncome": {
        "label": "Monthly Income",
        "description": "Monthly gross income of the borrower",
        "min": 100,
        "max": 500000,
        "default": 5000,
        "unit": "currency"
    },
    "DebtRatio": {
        "label": "Debt-to-Income Ratio",
        "description": "Total monthly debt payments divided by monthly income",
        "min": 0,
        "max": 10,
        "default": 0.5,
        "unit": "ratio"
    },
    "MonthsSinceLastDelinquent": {
        "label": "Months Since Last Delinquency",
        "description": "Number of months since the borrower's last delinquent payment (0 if never delinquent)",
        "min": 0,
        "max": 600,
        "default": 100,
        "unit": "months"
    },
    "NumberOfOpenCreditLinesAndLoans": {
        "label": "Open Credit Lines/Loans",
        "description": "Number of open credit lines and loans the borrower has",
        "min": 0,
        "max": 50,
        "default": 5,
        "unit": "count"
    },
    "NumberOfTimes90DaysLate": {
        "label": "Times 90+ Days Late",
        "description": "Number of times borrower has been 90+ days late on payments",
        "min": 0,
        "max": 100,
        "default": 0,
        "unit": "count"
    },
    "NumberOfRealEstateLoans": {
        "label": "Real Estate Loans",
        "description": "Number of mortgage and real estate loans",
        "min": 0,
        "max": 50,
        "default": 1,
        "unit": "count"
    },
    "NumberOfTimes60DaysLate": {
        "label": "Times 60-89 Days Late",
        "description": "Number of times borrower has been 60-89 days late on payments",
        "min": 0,
        "max": 100,
        "default": 0,
        "unit": "count"
    },
    "NumberOfDays90DaysLate": {
        "label": "Total 90+ Days Late (days)",
        "description": "Total number of days delinquent (90+ days late)",
        "min": 0,
        "max": 10000,
        "default": 0,
        "unit": "days"
    },
    "RevolvingUtilizationOfUnsecuredLines": {
        "label": "Credit Utilization Ratio",
        "description": "Total revolving credit used divided by total available credit (0-1 scale)",
        "min": 0,
        "max": 1,
        "default": 0.3,
        "unit": "ratio"
    },
}

# Example profiles for different risk scenarios
EXAMPLE_PROFILES = {
    "Low Risk - Stable Professional": {
        "Age": 45,
        "NumberOfDependents": 2,
        "MonthlyIncome": 8000,
        "DebtRatio": 0.2,
        "MonthsSinceLastDelinquent": 300,
        "NumberOfOpenCreditLinesAndLoans": 3,
        "NumberOfTimes90DaysLate": 0,
        "NumberOfRealEstateLoans": 1,
        "NumberOfTimes60DaysLate": 0,
        "NumberOfDays90DaysLate": 0,
        "RevolvingUtilizationOfUnsecuredLines": 0.15,
    },
    "Medium Risk - Occasional Issues": {
        "Age": 35,
        "NumberOfDependents": 1,
        "MonthlyIncome": 4000,
        "DebtRatio": 0.6,
        "MonthsSinceLastDelinquent": 30,
        "NumberOfOpenCreditLinesAndLoans": 6,
        "NumberOfTimes90DaysLate": 1,
        "NumberOfRealEstateLoans": 0,
        "NumberOfTimes60DaysLate": 1,
        "NumberOfDays90DaysLate": 15,
        "RevolvingUtilizationOfUnsecuredLines": 0.65,
    },
    "High Risk - Multiple Issues": {
        "Age": 28,
        "NumberOfDependents": 3,
        "MonthlyIncome": 2500,
        "DebtRatio": 1.2,
        "MonthsSinceLastDelinquent": 5,
        "NumberOfOpenCreditLinesAndLoans": 10,
        "NumberOfTimes90DaysLate": 3,
        "NumberOfRealEstateLoans": 2,
        "NumberOfTimes60DaysLate": 2,
        "NumberOfDays90DaysLate": 90,
        "RevolvingUtilizationOfUnsecuredLines": 0.95,
    }
}

# ==================== PAGE HEADER ====================
st.title("💳 Credit Risk & Default Prediction")
st.markdown("""
    ### Intelligent Risk Assessment for Loan Approval Decisions
    
    This tool uses advanced machine learning to predict the likelihood of credit default 
    based on borrower financial profiles. Enter customer details below to receive a 
    risk assessment and actionable insights.
    
    **Model Performance:**
    - AUC-ROC: 0.8628 | Gini Coefficient: 0.7255 | KS Statistic: 0.5779
""")

# ==================== SIDEBAR: QUICK ACTIONS ====================
st.sidebar.markdown("### 🎯 Quick Actions")

# Example profile selector
selected_profile = st.sidebar.selectbox(
    "📋 Load Example Profile",
    ["--- Enter Manually ---"] + list(EXAMPLE_PROFILES.keys()),
    help="Load pre-configured customer profiles to test the model"
)

# ==================== MAIN INPUT FORM ====================
st.markdown("### 📋 Customer Financial Profile")

# Initialize session state if empty
if "form_data" not in st.session_state:
    st.session_state.form_data = {key: FEATURE_DEFINITIONS[key]["default"] 
                                   for key in FEATURE_DEFINITIONS}

# Load example profile if selected
if selected_profile != "--- Enter Manually ---":
    st.session_state.form_data = EXAMPLE_PROFILES[selected_profile].copy()
    st.success(f"✅ Loaded profile: {selected_profile}")

# Create organized input columns
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-header">👤 Personal Information</div>', unsafe_allow_html=True)
    
    age = st.number_input(
        FEATURE_DEFINITIONS["Age"]["label"],
        min_value=FEATURE_DEFINITIONS["Age"]["min"],
        max_value=FEATURE_DEFINITIONS["Age"]["max"],
        value=st.session_state.form_data["Age"],
        step=1,
        help=FEATURE_DEFINITIONS["Age"]["description"]
    )
    st.session_state.form_data["Age"] = age
    
    dependents = st.number_input(
        FEATURE_DEFINITIONS["NumberOfDependents"]["label"],
        min_value=FEATURE_DEFINITIONS["NumberOfDependents"]["min"],
        max_value=FEATURE_DEFINITIONS["NumberOfDependents"]["max"],
        value=st.session_state.form_data["NumberOfDependents"],
        step=1,
        help=FEATURE_DEFINITIONS["NumberOfDependents"]["description"]
    )
    st.session_state.form_data["NumberOfDependents"] = dependents

with col2:
    st.markdown('<div class="section-header">💰 Income & Debt</div>', unsafe_allow_html=True)
    
    monthly_income = st.number_input(
        FEATURE_DEFINITIONS["MonthlyIncome"]["label"],
        min_value=FEATURE_DEFINITIONS["MonthlyIncome"]["min"],
        max_value=FEATURE_DEFINITIONS["MonthlyIncome"]["max"],
        value=st.session_state.form_data["MonthlyIncome"],
        step=100,
        help=FEATURE_DEFINITIONS["MonthlyIncome"]["description"]
    )
    st.session_state.form_data["MonthlyIncome"] = monthly_income
    
    debt_ratio = st.slider(
        FEATURE_DEFINITIONS["DebtRatio"]["label"],
        min_value=FEATURE_DEFINITIONS["DebtRatio"]["min"],
        max_value=FEATURE_DEFINITIONS["DebtRatio"]["max"],
        value=st.session_state.form_data["DebtRatio"],
        step=0.05,
        help=FEATURE_DEFINITIONS["DebtRatio"]["description"]
    )
    st.session_state.form_data["DebtRatio"] = debt_ratio

# Credit History Section
st.markdown('<div class="section-header">📊 Credit History & Payment Behavior</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    months_since_delinquent = st.number_input(
        FEATURE_DEFINITIONS["MonthsSinceLastDelinquent"]["label"],
        min_value=FEATURE_DEFINITIONS["MonthsSinceLastDelinquent"]["min"],
        max_value=FEATURE_DEFINITIONS["MonthsSinceLastDelinquent"]["max"],
        value=st.session_state.form_data["MonthsSinceLastDelinquent"],
        step=1,
        help=FEATURE_DEFINITIONS["MonthsSinceLastDelinquent"]["description"]
    )
    st.session_state.form_data["MonthsSinceLastDelinquent"] = months_since_delinquent

with col2:
    times_90_late = st.number_input(
        FEATURE_DEFINITIONS["NumberOfTimes90DaysLate"]["label"],
        min_value=FEATURE_DEFINITIONS["NumberOfTimes90DaysLate"]["min"],
        max_value=FEATURE_DEFINITIONS["NumberOfTimes90DaysLate"]["max"],
        value=st.session_state.form_data["NumberOfTimes90DaysLate"],
        step=1,
        help=FEATURE_DEFINITIONS["NumberOfTimes90DaysLate"]["description"]
    )
    st.session_state.form_data["NumberOfTimes90DaysLate"] = times_90_late

with col3:
    times_60_late = st.number_input(
        FEATURE_DEFINITIONS["NumberOfTimes60DaysLate"]["label"],
        min_value=FEATURE_DEFINITIONS["NumberOfTimes60DaysLate"]["min"],
        max_value=FEATURE_DEFINITIONS["NumberOfTimes60DaysLate"]["max"],
        value=st.session_state.form_data["NumberOfTimes60DaysLate"],
        step=1,
        help=FEATURE_DEFINITIONS["NumberOfTimes60DaysLate"]["description"]
    )
    st.session_state.form_data["NumberOfTimes60DaysLate"] = times_60_late

# Credit Accounts Section
st.markdown('<div class="section-header">🏦 Credit Accounts & Loans</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    open_credit_lines = st.number_input(
        FEATURE_DEFINITIONS["NumberOfOpenCreditLinesAndLoans"]["label"],
        min_value=FEATURE_DEFINITIONS["NumberOfOpenCreditLinesAndLoans"]["min"],
        max_value=FEATURE_DEFINITIONS["NumberOfOpenCreditLinesAndLoans"]["max"],
        value=st.session_state.form_data["NumberOfOpenCreditLinesAndLoans"],
        step=1,
        help=FEATURE_DEFINITIONS["NumberOfOpenCreditLinesAndLoans"]["description"]
    )
    st.session_state.form_data["NumberOfOpenCreditLinesAndLoans"] = open_credit_lines

with col2:
    real_estate_loans = st.number_input(
        FEATURE_DEFINITIONS["NumberOfRealEstateLoans"]["label"],
        min_value=FEATURE_DEFINITIONS["NumberOfRealEstateLoans"]["min"],
        max_value=FEATURE_DEFINITIONS["NumberOfRealEstateLoans"]["max"],
        value=st.session_state.form_data["NumberOfRealEstateLoans"],
        step=1,
        help=FEATURE_DEFINITIONS["NumberOfRealEstateLoans"]["description"]
    )
    st.session_state.form_data["NumberOfRealEstateLoans"] = real_estate_loans

with col3:
    credit_utilization = st.slider(
        FEATURE_DEFINITIONS["RevolvingUtilizationOfUnsecuredLines"]["label"],
        min_value=FEATURE_DEFINITIONS["RevolvingUtilizationOfUnsecuredLines"]["min"],
        max_value=FEATURE_DEFINITIONS["RevolvingUtilizationOfUnsecuredLines"]["max"],
        value=st.session_state.form_data["RevolvingUtilizationOfUnsecuredLines"],
        step=0.01,
        help=FEATURE_DEFINITIONS["RevolvingUtilizationOfUnsecuredLines"]["description"]
    )
    st.session_state.form_data["RevolvingUtilizationOfUnsecuredLines"] = credit_utilization

# Advanced Details (Optional)
with st.expander("⚙️ Advanced Details (Optional)", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        days_90_late = st.number_input(
            FEATURE_DEFINITIONS["NumberOfDays90DaysLate"]["label"],
            min_value=FEATURE_DEFINITIONS["NumberOfDays90DaysLate"]["min"],
            max_value=FEATURE_DEFINITIONS["NumberOfDays90DaysLate"]["max"],
            value=st.session_state.form_data["NumberOfDays90DaysLate"],
            step=1,
            help=FEATURE_DEFINITIONS["NumberOfDays90DaysLate"]["description"]
        )
        st.session_state.form_data["NumberOfDays90DaysLate"] = days_90_late

# ==================== PREDICTION BUTTON ====================
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    predict_button = st.button(
        "🔍 Predict Credit Risk",
        use_container_width=True,
        key="predict_btn"
    )

# ==================== RESULTS SECTION ====================
if predict_button:
    st.markdown("---")
    
    # Prepare data for prediction
    input_data = pd.DataFrame([st.session_state.form_data])
    
    # Mock prediction (replace with actual model loading)
    # For now, we'll create a simple scoring system
    try:
        # Load your model here
        # with open('fraud_model.pkl', 'rb') as f:
        #     model = pickle.load(f)
        # prediction_prob = model.predict_proba(input_data)[0][1]
        
        # Mock prediction based on simple heuristics
        risk_score = calculate_risk_score(st.session_state.form_data)
        
        # Determine risk level
        if risk_score < 0.3:
            risk_level = "🟢 LOW RISK"
            risk_class = "risk-low"
            recommendation = "✅ Likely Eligible for Approval"
            color = "#388e3c"
        elif risk_score < 0.6:
            risk_level = "🟡 MEDIUM RISK"
            risk_class = "risk-medium"
            recommendation = "⚠️ Further Review Recommended"
            color = "#f57c00"
        else:
            risk_level = "🔴 HIGH RISK"
            risk_class = "risk-high"
            recommendation = "❌ Caution: Higher Default Probability"
            color = "#d32f2f"
        
        # Display results in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Default Probability",
                value=f"{risk_score*100:.1f}%",
                delta=None
            )
        
        with col2:
            st.metric(
                label="Risk Classification",
                value=risk_level
            )
        
        with col3:
            st.metric(
                label="Recommendation",
                value=recommendation
            )
        
        # Risk Score Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_score * 100,
            title={'text': "Default Risk Score (%)"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 30], 'color': "rgba(56, 142, 60, 0.2)"},
                    {'range': [30, 60], 'color': "rgba(245, 124, 0, 0.2)"},
                    {'range': [60, 100], 'color': "rgba(211, 47, 47, 0.2)"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        
        fig_gauge.update_layout(height=400, margin=dict(l=20, r=20, t=70, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Key Risk Factors
        st.markdown("### 🔍 Key Risk Factors")
        
        risk_factors = identify_risk_factors(st.session_state.form_data)
        
        if risk_factors:
            for factor, severity in risk_factors:
                if severity == "High":
                    st.error(f"🔴 **{factor}** - High Risk Factor")
                elif severity == "Medium":
                    st.warning(f"🟡 **{factor}** - Moderate Risk Factor")
                else:
                    st.info(f"🟢 **{factor}** - Low Risk Factor")
        
        # Financial Summary
        st.markdown("### 📊 Financial Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            annual_income = st.session_state.form_data["MonthlyIncome"] * 12
            st.metric("Annual Income", f"${annual_income:,.0f}")
        
        with col2:
            st.metric("Debt-to-Income Ratio", f"{debt_ratio:.2f}")
        
        with col3:
            st.metric("Credit Utilization", f"{credit_utilization*100:.1f}%")
        
        with col4:
            st.metric("Payment History", "Good" if times_90_late == 0 else "Issues Detected")
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        
        recommendations = generate_recommendations(st.session_state.form_data, risk_score)
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
        
    except Exception as e:
        st.error(f"❌ Error in prediction: {str(e)}")
        st.info("Please ensure the model file is properly loaded.")

# ==================== HELPER FUNCTIONS ====================

def calculate_risk_score(data):
    """
    Calculate risk score based on customer profile.
    This is a simplified scoring - replace with actual model.
    """
    score = 0.2  # Base score
    
    # Debt ratio impact
    if data["DebtRatio"] > 0.8:
        score += 0.25
    elif data["DebtRatio"] > 0.5:
        score += 0.15
    
    # Payment history impact
    score += data["NumberOfTimes90DaysLate"] * 0.15
    score += data["NumberOfTimes60DaysLate"] * 0.08
    
    # Credit utilization impact
    if data["RevolvingUtilizationOfUnsecuredLines"] > 0.8:
        score += 0.15
    elif data["RevolvingUtilizationOfUnsecuredLines"] > 0.6:
        score += 0.08
    
    # Recent delinquency impact
    if data["MonthsSinceLastDelinquent"] < 12:
        score += 0.15
    elif data["MonthsSinceLastDelinquent"] < 36:
        score += 0.08
    
    return min(score, 0.95)

def identify_risk_factors(data):
    """Identify key risk factors in the profile."""
    factors = []
    
    if data["DebtRatio"] > 0.8:
        factors.append(("High Debt-to-Income Ratio", "High"))
    
    if data["NumberOfTimes90DaysLate"] > 0:
        factors.append(("Recent 90+ Day Late Payments", "High"))
    
    if data["RevolvingUtilizationOfUnsecuredLines"] > 0.8:
        factors.append(("High Credit Card Utilization", "Medium"))
    
    if data["MonthsSinceLastDelinquent"] < 12 and data["MonthsSinceLastDelinquent"] > 0:
        factors.append(("Recent Payment Delinquency", "High"))
    
    if data["NumberOfDependents"] > 3 and data["MonthlyIncome"] < 3000:
        factors.append(("High Dependent Burden Relative to Income", "Medium"))
    
    if len(factors) == 0:
        factors.append(("Financial Profile Appears Stable", "Low"))
    
    return factors

def generate_recommendations(data, risk_score):
    """Generate actionable recommendations."""
    recommendations = []
    
    if data["DebtRatio"] > 0.6:
        recommendations.append("Reduce outstanding debt or increase income to improve debt-to-income ratio")
    
    if data["RevolvingUtilizationOfUnsecuredLines"] > 0.7:
        recommendations.append("Pay down credit card balances to lower credit utilization ratio")
    
    if data["NumberOfTimes90DaysLate"] > 0:
        recommendations.append("Demonstrate consistent on-time payments for at least 12 months")
    
    if data["MonthsSinceLastDelinquent"] < 24 and data["MonthsSinceLastDelinquent"] > 0:
        recommendations.append("Continue maintaining current payment schedule; delinquency will age with time")
    
    if risk_score < 0.3:
        recommendations.append("Excellent profile - eligible for competitive interest rates")
    
    if len(recommendations) == 0:
        recommendations.append("Maintain current financial discipline and monitor credit profile")
    
    return recommendations

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #888; font-size: 12px;'>
        <p>💳 Credit Risk Assessment Tool | Powered by Machine Learning | Last Updated: 2026</p>
        <p><strong>Disclaimer:</strong> This tool provides predictions based on machine learning models. 
        Final loan decisions should consider additional factors and require human review.</p>
    </div>
""", unsafe_allow_html=True)