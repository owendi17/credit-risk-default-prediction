import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt

def portfolio_dashboard():
    """
    Advanced portfolio risk dashboard
    Shows portfolio-level insights and risk correlations
    """
    
    st.header("📊 Portfolio Risk Dashboard")
    st.markdown("Executive-level insights into your entire credit portfolio")
    
    # Generate or load portfolio data
    portfolio_df = generate_portfolio_data()
    
    # Dashboard Controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        date_range = st.date_input(
            "Date Range",
            value=(pd.Timestamp.now() - pd.Timedelta(days=90), pd.Timestamp.now()),
            max_value=pd.Timestamp.now()
        )
    
    with col2:
        min_loan_amount = st.number_input("Min Loan Amount ($)", value=5000, step=1000)
    
    with col3:
        risk_threshold = st.slider("Risk Threshold (%)", 0, 100, 70)
    
    # Apply filters
    filtered_df = portfolio_df[
        (portfolio_df['loan_amount'] >= min_loan_amount)
    ].copy()
    
    # KEY METRICS ROW
    st.subheader("🎯 Portfolio Health Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_portfolio = filtered_df['loan_amount'].sum()
    default_risk_total = (filtered_df['default_risk_score'] * filtered_df['loan_amount']).sum() / total_portfolio
    
    with col1:
        st.metric(
            "Total Portfolio Value",
            f"${total_portfolio/1e6:.1f}M",
            delta=f"+$120K this month"
        )
    
    with col2:
        portfolio_default_prob = filtered_df['default_risk_score'].mean()
        st.metric(
            "Portfolio Default Rate",
            f"{portfolio_default_prob:.1f}%",
            delta="+2.3% vs last month",
            delta_color="inverse"
        )
    
    with col3:
        num_customers = len(filtered_df)
        num_high_risk = len(filtered_df[filtered_df['default_risk_score'] > risk_threshold])
        st.metric(
            "High-Risk Accounts",
            f"{num_high_risk}",
            delta=f"{(num_high_risk/num_customers)*100:.1f}% of portfolio"
        )
    
    with col4:
        avg_credit_score = filtered_df['credit_score'].mean()
        st.metric(
            "Avg Credit Score",
            f"{avg_credit_score:.0f}",
            delta="-12 points from last quarter"
        )
    
    with col5:
        portfolio_health = max(0, 100 - portfolio_default_prob)
        st.metric(
            "Portfolio Health",
            f"{portfolio_health:.1f}%",
            delta="-2.3% from last month",
            delta_color="inverse"
        )
    
    # MAIN DASHBOARD GRID
    st.markdown("---")
    
    # Row 1: Risk Distribution & Heatmap
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("📈 Risk Score Distribution")
        
        fig_hist = go.Figure()
        
        # Add histogram with zones
        fig_hist.add_trace(go.Histogram(
            x=filtered_df['default_risk_score'],
            nbinsx=40,
            name='Risk Score',
            marker=dict(color='rgba(59, 130, 246, 0.7)'),
            marker_line=dict(color='rgba(59, 130, 246, 1)', width=0.5)
        ))
        
        # Add zone lines
        fig_hist.add_vline(x=30, line_dash="dash", line_color="#22c55e", 
                          annotation_text="Low/Med", annotation_position="top")
        fig_hist.add_vline(x=70, line_dash="dash", line_color="#ef4444", 
                          annotation_text="Med/High", annotation_position="top")
        
        fig_hist.update_layout(
            title="",
            xaxis_title="Default Risk Score (%)",
            yaxis_title="Number of Accounts",
            template="plotly_dark",
            height=350,
            showlegend=False,
            hovermode='x unified',
            margin=dict(l=0, r=0, t=20, b=0)
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Risk Tier Distribution")
        
        risk_tiers = pd.cut(filtered_df['default_risk_score'], 
                            bins=[0, 30, 70, 100],
                            labels=['Low', 'Medium', 'High'])
        tier_counts = risk_tiers.value_counts()
        
        colors_map = {'Low': '#22c55e', 'Medium': '#eab308', 'High': '#ef4444'}
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=tier_counts.index,
            values=tier_counts.values,
            marker=dict(colors=[colors_map.get(label, '#64748b') for label in tier_counts.index]),
            textposition='inside',
            textinfo='label+percent+value',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig_pie.update_layout(
            title="",
            template="plotly_dark",
            height=350,
            margin=dict(l=0, r=0, t=20, b=0)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Row 2: Feature Impact Heatmap
    st.subheader("🔥 Risk Driver Correlation Matrix")
    
    # Create correlation matrix
    corr_features = ['default_risk_score', 'credit_score', 'debt_ratio', 'age', 'income']
    corr_matrix = filtered_df[corr_features].corr()
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=['Risk Score', 'Credit Score', 'Debt Ratio', 'Age', 'Income'],
        y=['Risk Score', 'Credit Score', 'Debt Ratio', 'Age', 'Income'],
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig_heatmap.update_layout(
        title="",
        template="plotly_dark",
        height=400,
        margin=dict(l=0, r=0, t=20, b=0)
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Row 3: Feature vs Risk Analysis
    st.subheader("📊 Risk Drivers Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Credit Score vs Risk
        fig_scatter1 = px.scatter(
            filtered_df,
            x='credit_score',
            y='default_risk_score',
            color='default_risk_score',
            size='loan_amount',
            color_continuous_scale='RdYlGn_r',
            title="Credit Score vs Risk Score",
            labels={'credit_score': 'Credit Score', 'default_risk_score': 'Default Risk %'},
            hover_data=['age', 'income', 'debt_ratio'],
            trendline='ols',
            trendline_color_override='rgba(255,255,255,0.5)'
        )
        fig_scatter1.update_layout(
            template="plotly_dark",
            height=350,
            hovermode='closest',
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_scatter1, use_container_width=True)
    
    with col2:
        # Debt Ratio vs Risk
        fig_scatter2 = px.scatter(
            filtered_df,
            x='debt_ratio',
            y='default_risk_score',
            color='default_risk_score',
            size='loan_amount',
            color_continuous_scale='RdYlGn_r',
            title="Debt Ratio vs Risk Score",
            labels={'debt_ratio': 'Debt Ratio (%)', 'default_risk_score': 'Default Risk %'},
            hover_data=['age', 'income', 'credit_score'],
            trendline='ols',
            trendline_color_override='rgba(255,255,255,0.5)'
        )
        fig_scatter2.update_layout(
            template="plotly_dark",
            height=350,
            hovermode='closest',
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_scatter2, use_container_width=True)
    
    # Row 4: Risk Segmentation
    st.subheader("📊 Risk by Demographic Segments")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk by Age Group
        age_bins = [0, 25, 35, 45, 55, 65, 100]
        age_labels = ['<25', '25-35', '35-45', '45-55', '55-65', '65+']
        filtered_df['age_group'] = pd.cut(filtered_df['age'], bins=age_bins, labels=age_labels)
        
        risk_by_age = filtered_df.groupby('age_group', observed=True).agg({
            'default_risk_score': ['mean', 'count'],
            'loan_amount': 'sum'
        }).reset_index()
        risk_by_age.columns = ['Age Group', 'Avg Risk', 'Count', 'Total Amount']
        
        fig_age = go.Figure()
        fig_age.add_trace(go.Bar(
            x=risk_by_age['Age Group'],
            y=risk_by_age['Avg Risk'],
            marker_color='rgba(59, 130, 246, 0.7)',
            text=risk_by_age['Avg Risk'].round(1),
            textposition='outside',
            name='Avg Risk %',
            hovertemplate='<b>%{x}</b><br>Avg Risk: %{y:.1f}%<extra></extra>'
        ))
        fig_age.update_layout(
            title="Risk by Age Group",
            xaxis_title="Age Group",
            yaxis_title="Average Risk Score (%)",
            template="plotly_dark",
            height=300,
            showlegend=False,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        # Risk by Income Bracket
        income_bins = [0, 40000, 60000, 80000, 100000, 1000000]
        income_labels = ['<40K', '40-60K', '60-80K', '80-100K', '100K+']
        filtered_df['income_bracket'] = pd.cut(filtered_df['income'], bins=income_bins, labels=income_labels)
        
        risk_by_income = filtered_df.groupby('income_bracket', observed=True).agg({
            'default_risk_score': ['mean', 'count']
        }).reset_index()
        risk_by_income.columns = ['Income Bracket', 'Avg Risk', 'Count']
        
        fig_income = go.Figure()
        fig_income.add_trace(go.Bar(
            x=risk_by_income['Income Bracket'],
            y=risk_by_income['Avg Risk'],
            marker_color='rgba(34, 197, 94, 0.7)',
            text=risk_by_income['Avg Risk'].round(1),
            textposition='outside',
            name='Avg Risk %',
            hovertemplate='<b>%{x}</b><br>Avg Risk: %{y:.1f}%<extra></extra>'
        ))
        fig_income.update_layout(
            title="Risk by Income Bracket",
            xaxis_title="Income Bracket",
            yaxis_title="Average Risk Score (%)",
            template="plotly_dark",
            height=300,
            showlegend=False,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_income, use_container_width=True)
    
    # Risk Time Series (Simulated)
    st.subheader("📅 Portfolio Risk Trend")
    
    # Generate time series data
    dates = pd.date_range(start='2023-01-01', end='2024-05-28', freq='D')
    risk_trend = pd.DataFrame({
        'date': dates,
        'portfolio_risk': 45 + np.cumsum(np.random.randn(len(dates)) * 0.2)
    })
    risk_trend['portfolio_risk'] = risk_trend['portfolio_risk'].clip(20, 80)
    risk_trend['high_risk_count'] = 150 + np.cumsum(np.random.randn(len(dates)) * 0.5).astype(int)
    risk_trend['high_risk_count'] = risk_trend['high_risk_count'].clip(100, 300)
    
    fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig_trend.add_trace(
        go.Scatter(
            x=risk_trend['date'],
            y=risk_trend['portfolio_risk'],
            name='Portfolio Risk %',
            line=dict(color='#3b82f6', width=3),
            fill='tozeroy',
            fillcolor='rgba(59, 130, 246, 0.2)',
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Risk: %{y:.1f}%<extra></extra>'
        ),
        secondary_y=False
    )
    
    fig_trend.add_trace(
        go.Scatter(
            x=risk_trend['date'],
            y=risk_trend['high_risk_count'],
            name='High-Risk Accounts',
            line=dict(color='#ef4444', width=3),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Count: %{y}<extra></extra>'
        ),
        secondary_y=True
    )
    
    fig_trend.update_xaxes(title_text="Date")
    fig_trend.update_yaxes(title_text="Portfolio Risk (%)", secondary_y=False)
    fig_trend.update_yaxes(title_text="High-Risk Account Count", secondary_y=True)
    
    fig_trend.update_layout(
        title="",
        template="plotly_dark",
        height=350,
        hovermode='x unified',
        margin=dict(l=0, r=0, t=20, b=0)
    )
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Export Dashboard Data
    st.markdown("---")
    st.subheader("⬇️ Export Dashboard Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        dashboard_summary = pd.DataFrame({
            'Metric': [
                'Total Portfolio Value',
                'Portfolio Default Rate',
                'High-Risk Accounts',
                'Average Credit Score',
                'Total Accounts'
            ],
            'Value': [
                f"${total_portfolio/1e6:.1f}M",
                f"{portfolio_default_prob:.1f}%",
                f"{num_high_risk}",
                f"{avg_credit_score:.0f}",
                f"{num_customers}"
            ]
        })
        csv_data = dashboard_summary.to_csv(index=False)
        st.download_button(
            label="📊 Export Summary",
            data=csv_data,
            file_name="portfolio_summary.csv",
            mime="text/csv"
        )
    
    with col2:
        st.download_button(
            label="📈 Export Full Data",
            data=filtered_df.to_csv(index=False),
            file_name="portfolio_data.csv",
            mime="text/csv"
        )
    
    with col3:
        st.markdown("*More formats coming soon*")


def generate_portfolio_data(n_records=500):
    """Generate synthetic portfolio data for demonstration"""
    np.random.seed(42)
    
    df = pd.DataFrame({
        'customer_id': [f'C{i:05d}' for i in range(n_records)],
        'age': np.random.randint(22, 70, n_records),
        'income': np.random.lognormal(10.5, 0.8, n_records),
        'credit_score': np.random.randint(300, 850, n_records),
        'debt_ratio': np.random.uniform(0, 100, n_records),
        'loan_amount': np.random.lognormal(10, 1, n_records),
        'employment_years': np.random.randint(0, 40, n_records),
    })
    
    # Calculate default risk based on features
    df['default_risk_score'] = (
        0.3 * (750 - df['credit_score']) / 10 +
        0.3 * df['debt_ratio'] / 100 +
        0.2 * (45 - df['age']) / 35 +
        0.2 * (80000 - df['income']) / 80000
    ) * 100
    
    df['default_risk_score'] = df['default_risk_score'].clip(0, 100)
    df['default_risk_score'] += np.random.normal(0, 3, n_records)
    df['default_risk_score'] = df['default_risk_score'].clip(0, 100)
    
    return df


if __name__ == "__main__":
    portfolio_dashboard()
