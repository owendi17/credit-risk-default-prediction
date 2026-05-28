import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import tempfile
import os

def batch_processor():
    """
    Batch prediction processor component
    Handles CSV upload, prediction, and results export
    """
    
    st.header("🚀 Batch Risk Assessment")
    st.markdown("Upload a CSV file to predict default risk for multiple customers simultaneously.")
    
    # Create two columns for upload and settings
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=['csv'],
            help="CSV must contain: age, income, credit_score, debt_ratio, employment_years, num_accounts"
        )
    
    with col2:
        st.markdown("### Settings")
        confidence_level = st.slider("Confidence Level", 0.5, 1.0, 0.95, 0.05)
    
    if uploaded_file is None:
        st.info("👈 Upload a CSV file to get started. Sample columns: age, income, credit_score, debt_ratio")
        
        # Show sample CSV format
        with st.expander("📋 View Sample CSV Format"):
            sample_data = pd.DataFrame({
                'customer_id': ['C001', 'C002', 'C003'],
                'age': [34, 45, 28],
                'income': [65000, 95000, 52000],
                'credit_score': [580, 720, 650],
                'debt_ratio': [45, 25, 55],
                'employment_years': [5, 12, 3],
                'num_accounts': [4, 6, 2]
            })
            st.dataframe(sample_data, use_container_width=True)
            
            csv_string = sample_data.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Sample CSV",
                data=csv_string,
                file_name="sample_customers.csv",
                mime="text/csv"
            )
        return
    
    # Read CSV
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(df)} customer records")
    except Exception as e:
        st.error(f"❌ Error reading CSV: {str(e)}")
        return
    
    # Validate columns
    required_cols = ['age', 'income', 'credit_score', 'debt_ratio']
    optional_cols = ['employment_years', 'num_accounts']
    
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"❌ Missing required columns: {', '.join(missing)}")
        return
    
    # Add optional columns if missing
    for col in optional_cols:
        if col not in df.columns:
            df[col] = 0
    
    # Display data preview
    st.subheader("📊 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Process predictions
    st.subheader("🔍 Running Predictions")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    predictions = []
    
    for idx, row in df.iterrows():
        # Simulate prediction (replace with your actual model)
        risk_score = predict_default_risk(row)
        predictions.append(risk_score)
        
        progress = (idx + 1) / len(df)
        progress_bar.progress(progress)
        status_text.text(f"Processing: {idx + 1}/{len(df)} customers...")
    
    # Add predictions to dataframe
    df['default_risk_score'] = predictions
    df['risk_category'] = df['default_risk_score'].apply(categorize_risk)
    df['confidence'] = np.random.uniform(0.75, 0.98, len(df))  # Replace with real confidence
    
    progress_bar.empty()
    status_text.empty()
    st.success(f"✅ Completed predictions for {len(df)} customers")
    
    # Risk Summary Metrics
    st.subheader("📈 Portfolio Risk Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_risk = df['default_risk_score'].mean()
        st.metric("Average Risk", f"{avg_risk:.1f}%")
    
    with col2:
        high_risk_count = len(df[df['risk_category'] == 'High'])
        high_risk_pct = (high_risk_count / len(df)) * 100
        st.metric("High Risk", f"{high_risk_count} ({high_risk_pct:.1f}%)", 
                  delta=f"{high_risk_pct:.1f}% of portfolio")
    
    with col3:
        med_risk_count = len(df[df['risk_category'] == 'Medium'])
        med_risk_pct = (med_risk_count / len(df)) * 100
        st.metric("Medium Risk", f"{med_risk_count} ({med_risk_pct:.1f}%)")
    
    with col4:
        low_risk_count = len(df[df['risk_category'] == 'Low'])
        low_risk_pct = (low_risk_count / len(df)) * 100
        st.metric("Low Risk", f"{low_risk_count} ({low_risk_pct:.1f}%)")
    
    # Risk Distribution Chart
    st.subheader("📊 Risk Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=df['default_risk_score'],
            nbinsx=30,
            marker_color='rgba(59, 130, 246, 0.7)',
            marker_line=dict(color='rgba(59, 130, 246, 1)', width=1)
        ))
        fig_hist.update_layout(
            title="Risk Score Distribution",
            xaxis_title="Default Risk %",
            yaxis_title="Number of Customers",
            template="plotly_dark",
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Risk Category Pie Chart
        risk_counts = df['risk_category'].value_counts()
        colors = {'Low': '#22c55e', 'Medium': '#eab308', 'High': '#ef4444'}
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            marker=dict(colors=[colors.get(cat, '#64748b') for cat in risk_counts.index]),
            textposition='inside',
            textinfo='label+percent'
        )])
        fig_pie.update_layout(
            title="Risk Category Breakdown",
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Results Table with Filtering
    st.subheader("📋 Detailed Results")
    
    # Add filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_filter = st.multiselect(
            "Filter by Risk Category",
            ['Low', 'Medium', 'High'],
            default=['Low', 'Medium', 'High']
        )
    
    with col2:
        min_risk = st.slider("Minimum Risk Score", 0.0, 100.0, 0.0)
    
    with col3:
        max_risk = st.slider("Maximum Risk Score", 0.0, 100.0, 100.0)
    
    # Apply filters
    filtered_df = df[
        (df['risk_category'].isin(risk_filter)) &
        (df['default_risk_score'] >= min_risk) &
        (df['default_risk_score'] <= max_risk)
    ].copy()
    
    # Format for display
    display_df = filtered_df[[
        'customer_id', 'age', 'income', 'credit_score', 'debt_ratio',
        'default_risk_score', 'risk_category', 'confidence'
    ]].copy() if 'customer_id' in filtered_df.columns else filtered_df[[
        'age', 'income', 'credit_score', 'debt_ratio',
        'default_risk_score', 'risk_category', 'confidence'
    ]].copy()
    
    # Round numeric columns
    numeric_cols = display_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in ['default_risk_score', 'confidence']:
            display_df[col] = display_df[col].round(2)
        elif col == 'income':
            display_df[col] = display_df[col].astype(int)
        else:
            display_df[col] = display_df[col].round(1)
    
    # Rename columns for display
    display_df.columns = [col.replace('_', ' ').title() for col in display_df.columns]
    
    st.dataframe(
        display_df,
        use_container_width=True,
        height=400
    )
    
    st.markdown(f"**Showing {len(filtered_df)} of {len(df)} customers**")
    
    # Export Results
    st.subheader("⬇️ Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV Export
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        st.download_button(
            label="📊 Download Full Results (CSV)",
            data=csv_buffer.getvalue(),
            file_name="risk_predictions.csv",
            mime="text/csv"
        )
    
    with col2:
        # High Risk Only
        high_risk_df = df[df['risk_category'] == 'High']
        high_risk_csv = high_risk_df.to_csv(index=False)
        
        st.download_button(
            label="⚠️ High Risk Only (CSV)",
            data=high_risk_csv,
            file_name="high_risk_customers.csv",
            mime="text/csv"
        )
    
    with col3:
        st.markdown("*More export formats coming soon*")
    
    # Statistical Summary
    st.subheader("📊 Statistical Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Risk Score Statistics**")
        stats = {
            'Mean': df['default_risk_score'].mean(),
            'Median': df['default_risk_score'].median(),
            'Std Dev': df['default_risk_score'].std(),
            'Min': df['default_risk_score'].min(),
            'Max': df['default_risk_score'].max()
        }
        stats_df = pd.DataFrame(list(stats.items()), columns=['Metric', 'Value'])
        stats_df['Value'] = stats_df['Value'].round(2)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.write("**Risk by Credit Score Range**")
        df['credit_band'] = pd.cut(df['credit_score'], 
                                    bins=[0, 600, 700, 800, 850],
                                    labels=['Poor', 'Fair', 'Good', 'Excellent'])
        risk_by_credit = df.groupby('credit_band')['default_risk_score'].agg(['mean', 'count'])
        risk_by_credit.columns = ['Avg Risk', 'Count']
        risk_by_credit = risk_by_credit.round(2)
        st.dataframe(risk_by_credit, use_container_width=True)


def predict_default_risk(row):
    """
    Simulate model prediction
    Replace with your actual model inference
    """
    # Normalized weights (simplified model)
    score = 0
    
    # Credit score impact (inverse - lower score = higher risk)
    credit_factor = (750 - row['credit_score']) / 10
    score += max(0, credit_factor) * 0.4
    
    # Debt ratio impact
    debt_factor = max(0, row['debt_ratio'] - 30) / 70
    score += debt_factor * 0.3
    
    # Age impact (younger = slightly higher risk)
    age_factor = max(0, (45 - row['age']) / 35)
    score += age_factor * 0.15
    
    # Income impact (lower income = higher risk)
    income_factor = max(0, (80000 - row['income']) / 80000)
    score += income_factor * 0.15
    
    # Convert to percentage and clamp
    risk_percentage = max(0, min(100, score * 100))
    
    # Add small random noise for realism
    noise = np.random.normal(0, 2)
    return max(0, min(100, risk_percentage + noise))


def categorize_risk(score):
    """Categorize risk score into Low, Medium, High"""
    if score < 30:
        return 'Low'
    elif score < 70:
        return 'Medium'
    else:
        return 'High'


if __name__ == "__main__":
    batch_processor()
