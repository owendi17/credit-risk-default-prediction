import streamlit as st
import pandas as pd
import numpy as np
import pickle
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
FEATURE_DEFINITIONS = {
    "Age": {
        "label": "Age (years)",
        "description": "Age of the borrower in years",
        "min": 18,
        "max": 120,
        "default": 40,
    },
    "NumberOfDependents": {
        "label": "Number of Dependents",
        "description": "Number of dependents (children) the borrower has",
        "min": 0,
        "max": 20,
        "default": 1,
    },
    "MonthlyIncome": {
        "label": "Monthly Income",
        "description": "Monthly gross income of the borrower",
        "min": 100,
        "max": 500000,
        "default": 5000,
    },
    "DebtRatio": {
        "label": "Debt-to-Income Ratio",
        "description": "Total monthly debt payments divided by monthly income",
        "min": 0,
        "max": 10,
        "default": 0.5,
    },
    "MonthsSinceLastDelinquent": {
        "label": "Months Since Last Delinquency",
        "description": "Number of months since the borrower's last delinquent payment (0 if never delinquent)",
        "min": 0,
        "max": 600,
        "default": 100,
    },
    "NumberOfOpenCreditLinesAndLoans": {
        "label": "Open Credit Lines/Loans",
        "description": "Number of open credit lines and loans the borrower has",
        "min": 0,
        "max": 50,
        "default": 5,
    },
    "NumberOfTimes90DaysLate": {
        "label": "Times 90+ Days Late",
        "description": "Number of times borrower has been 90+ days late on payments",
        "min": 0,
        "max": 100,
        "default": 0,
    },
    "NumberOfRealEstateLoans": {
        "label": "Real Estate Loans",
        "description": "Number of mortgage and real estate loans",
        "min": 0,
        "max": 50,
        "default": 1,
    },
    "NumberOfTimes60DaysLate": {
        "label": "Times 60-89 Days Late",
        "description": "Number of times borrower has been 60-89 days late on payments",
        "min": 0,
        "max": 100,
        "default": 0,
    },
    "NumberOfDays90DaysLate": {
        "label": "Total 90+ Days Late (days)",
        "description": "Total number of days delinquent (90+ days late)",
        "min": 0,
        "max": 10000,
        "default": 0,
    },
    "RevolvingUtilizationOfUnsecuredLines": {
        "label": "Credit Utilization Ratio",
        "description": "Total revolving credit used divided by total available credit (0-1 scale)",
        "min": 0,
        "max": 1,
        "default": 0.3,
    },
}

# Example profiles
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
""")

# ==================== SIDEBAR ====================
st.sidebar.markdown("### 🎯 Quick Actions")

# Example profile selector
selected_profile = st.sidebar.selectbox(
    "📋 Load Example Profile",
    ["--- Enter Manually ---"] + list(EXAMPLE_PROFILES.keys()),
    help="Load pre-configured customer profiles to test the model"
)

# ==================== INITIALIZE SESSION STATE ====================
if "form_data" not in st.session_state:
    st.session_state.form_data = {key: FEATURE_DEFINITIONS[key]["default"] 
                                   for key in FEATURE_DEFINITIONS}

# Load example profile if selected
if selected_profile != "--- Enter Manually ---":
    st.session_state.form_data = EXAMPLE_PROFILES[selected_profile].copy()
    st.success(f"✅ Loaded profile: {selected_profile}")

# ==================== INPUT FORM ====================
st.markdown("### 📋 Customer Financial Profile")

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-header">👤 Personal Information</div>', unsafe_allow_html=True)
    
    age = st.number_input(
        FEATURE_DEFINITIONS["Age"]["label"],
        min_value=FEATURE_DEFINITIONS["Age"]["min"],
        max_value=FEATURE_DEFINITIONS["Age"]["max"],
        value=int(st.session_state.form_data["Age"]),
        step=1,
        help=FEATURE_DEFINITIONS["Age"]["description"]
    )
    st.session_state.form_data["Age"] = age
    
    dependents = st.number_input(
        FEATURE_DEFINITIONS["NumberOfDependents"]["label"],
        min_value=FEATURE_DEFINITIONS["NumberOfDependents"]["min"],
        max_value=FEATURE_DEFINITIONS["NumberOfDependents"]["max"],
        value=int(st.session_state.form_data["NumberOfDependents"]),
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
        value=int(st.session_state.form_data["MonthlyIncome"]),
        step=100,
        help=FEATURE_DEFINITIONS["MonthlyIncome"]["description"]
    )
    st.session_state.form_data["MonthlyIncome"] = monthly_income
    
    debt_ratio = st.number_input(
        FEATURE_DEFINITIONS["DebtRatio"]["label"],
        min_value=0.0,
        max_value=10.0,
        value=float(st.session_state.form_data["DebtRatio"]),
        step=0.05,
        help=FEATURE_DEFINITIONS["DebtRatio"]["description"],
        format="%.2f"
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
        value=int(st.session_state.form_data["MonthsSinceLastDelinquent"]),
        step=1,
        help=FEATURE_DEFINITIONS["MonthsSinceLastDelinquent"]["description"]
    )
    st.session_state.form_data["MonthsSinceLastDelinquent"] = months_since_delinquent

with col2:
    times_90_late = st.number_input(
        FEATURE_DEFINITIONS["NumberOfTimes90DaysLate"]["label"],
        min_value=FEATURE_DEFINITIONS["NumberOfTimes90DaysLate"]["min"],
        max_value=FEATURE_DEFINITIONS["NumberOfTimes90DaysLate"]["max"],
        value=int(st.session_state.form_data["NumberOfTimes90DaysLate"]),
        step=1,
        help=FEATURE_DEFINITIONS["NumberOfTimes90DaysLate"]["description"]
    )
    st.session_state.form_data["NumberOfTimes90DaysLate"] = times_90_late

with col3:
    times_60_late = st.number_input(
        FEATURE_DEFINITIONS["NumberOfTimes60DaysLate"]["label"],
        min_value=FEATURE_DEFINITIONS["NumberOfTimes60DaysLate"]["min"],
        max_value=FEATURE_DEFINITIONS["NumberOfTimes60DaysLate"]["max"],
        value=int(st.session_state.form_data["NumberOfTimes60DaysLate"]),
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
        value=int(st.session_state.form_data["NumberOfOpenCreditLinesAndLoans"]),
        step=1,
        help=FEATURE_DEFINITIONS["NumberOfOpenCreditLinesAndLoans"]["description"]
    )
    st.session_state.form_data["NumberOfOpenCreditLinesAndLoans"] = open_credit_lines

with col2:
    real_estate_loans = st.number_input(
        FEATURE_DEFINITIONS["NumberOfRealEstateLoans"]["label"],
        min_value=FEATURE_DEFINITIONS["NumberOfRealEstateLoans"]["min"],
        max_value=FEATURE_DEFINITIONS["NumberOfRealEstateLoans"]["max"],
        value=int(st.session_state.form_data["NumberOfRealEstateLoans"]),
        step=1,
        help=FEATURE_DEFINITIONS["NumberOfRealEstateLoans"]["description"]
    )
    st.session_state.form_data["NumberOfRealEstateLoans"] = real_estate_loans

with col3:
    credit_utilization = st.number_input(
        FEATURE_DEFINITIONS["RevolvingUtilizationOfUnsecuredLines"]["label"],
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.form_data["RevolvingUtilizationOfUnsecuredLines"]),
        step=0.01,
        help=FEATURE_DEFINITIONS["RevolvingUtilizationOfUnsecuredLines"]["description"],
        format="%.2f"
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
            value=int(st.session_state.form_data["NumberOfDays90DaysLate"]),
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
    
    try:
        # Calculate risk score (using mock model - replace with your actual model)
        risk_score = calculate_risk_score(st.session_state.form_data)
        
        # Determine risk level
        if risk_score < 0.3:
            risk_level = "🟢 LOW RISK"
            recommendation = "✅ Likely Eligible for Approval"
        elif risk_score < 0.6:
            risk_level = "🟡 MEDIUM RISK"
            recommendation = "⚠️ Further Review Recommended"
        else:
            risk_level = "🔴 HIGH RISK"
            recommendation = "❌ Caution: Higher Default Probability"
        
        # Display results in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Default Probability",
                value=f"{risk_score*100:.1f}%"
            )
        
        with col2:
            st.markdown(f"### Risk Classification\n{risk_level}")
        
        with col3:
            st.markdown(f"### Recommendation\n{recommendation}")
        
        # Risk bar
        st.markdown("### 📊 Risk Score Visualization")
        
        # Create a simple progress bar
        risk_percentage = int(risk_score * 100)
        st.progress(risk_score, text=f"{risk_percentage}% Default Risk")
        
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
            payment_status = "Good" if times_90_late == 0 else "Issues Detected"
            st.metric("Payment History", payment_status)
        
        # Recommendations
        st.markdown("### 💡 Actionable Recommendations")
        
        recommendations = generate_recommendations(st.session_state.form_data, risk_score)
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
        
        # Model Info
        st.info("ℹ️ **Note:** This prediction is based on machine learning analysis. Final loan decisions should consider additional factors and require human review.")
        
    except Exception as e:
        st.error(f"❌ Error in prediction: {str(e)}")
        st.info("Please check all input values and try again.")

# ==================== HELPER FUNCTIONS ====================

def calculate_risk_score(data):
    """Calculate risk score based on customer profile."""
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
        <p>💳 Credit Risk Assessment Tool | Powered by Machine Learning</p>
        <p><strong>Disclaimer:</strong> This tool provides predictions based on machine learning models. 
        Final loan decisions should consider additional factors and require human review.</p>
    </div>
""", unsafe_allow_html=True)