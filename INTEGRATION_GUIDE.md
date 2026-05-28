# Credit Risk App Enhancement - Integration Guide

## 📋 Overview

This guide walks you through integrating the 4 new components into your existing Streamlit app:

1. **Interactive Risk Gauge Widget** - Beautiful speedometer visualization
2. **CSV Batch Processor** - Bulk prediction and analysis
3. **Portfolio Dashboard** - Enterprise analytics
4. **Dark Mode + Responsive Design** - Complete UI refresh

---

## 🚀 Quick Start (5 minutes)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Update Your Main App File

Replace your current `app.py` with the provided `app.py` file. It includes:
- Complete dark mode styling
- Page routing (Home, Single Prediction, Batch, Dashboard, Settings)
- Custom CSS for responsive design
- All color themes and animations

### Step 3: Create Supporting Modules

Copy these files to your project directory:
- `batch_processor.py` - Batch processing logic
- `portfolio_dashboard.py` - Dashboard analytics

### Step 4: Update Streamlit Config

Create `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#3b82f6"
backgroundColor = "#0f172a"
secondaryBackgroundColor = "#1e293b"
textColor = "#e2e8f0"
font = "sans serif"

[client]
showErrorDetails = false
showWarningOnDirectExecution = false

[logger]
level = "warning"

[browser]
gatherUsageStats = false
```

### Step 5: Run the App

```bash
streamlit run app.py
```

---

## 🎯 Component Details

### 1️⃣ Interactive Risk Gauge Widget

**Location:** `risk_gauge_widget.html`

**Features:**
- SVG-based speedometer gauge
- Smooth needle animation
- Dynamic risk zones (Green/Yellow/Red)
- Detail metrics display
- Action buttons

**How to Integrate:**
```python
import streamlit as st

# Option A: Display as HTML component
with open('risk_gauge_widget.html', 'r') as f:
    html_code = f.read()
    st.components.v1.html(html_code, height=800)

# Option B: Use data-driven version (Python)
from risk_gauge import update_risk_gauge
update_risk_gauge(risk_score=72, income=65000, age=34, credit_score=580)
```

**Customization:**
- Change color zones in SVG `<linearGradient>` section
- Adjust needle smoothness: `transition: transform 1.2s cubic-bezier(...)`
- Modify risk ranges: Edit `getRiskLabel()` function

---

### 2️⃣ CSV Batch Processor

**Location:** `batch_processor.py`

**Features:**
- CSV file upload with validation
- Real-time progress tracking
- Risk categorization and filtering
- Statistical summary
- Multiple export formats
- Risk-based segmentation

**How to Use:**
```python
from batch_processor import batch_processor

# In your Streamlit app:
batch_processor()  # Full component with UI
```

**API Integration:**
```python
import pandas as pd
from batch_processor import predict_default_risk, categorize_risk

# Predict single customer
df = pd.DataFrame({
    'age': [35],
    'income': [65000],
    'credit_score': [650],
    'debt_ratio': [45]
})

df['risk_score'] = df.apply(predict_default_risk, axis=1)
df['risk_category'] = df['risk_score'].apply(categorize_risk)
```

**Expected CSV Columns:**
```
customer_id, age, income, credit_score, debt_ratio, employment_years, num_accounts
```

**Output Formats:**
- Full results (CSV)
- High-risk only (CSV)
- Statistical summary (JSON)

---

### 3️⃣ Portfolio Dashboard

**Location:** `portfolio_dashboard.py`

**Features:**
- Real-time portfolio metrics
- Risk distribution histograms
- Correlation heatmaps
- Feature impact analysis
- Demographic segmentation
- Risk trend analysis
- Multiple export options

**How to Use:**
```python
from portfolio_dashboard import portfolio_dashboard

# In your Streamlit app:
portfolio_dashboard()  # Full dashboard with controls
```

**Dashboard Sections:**
1. **Health Metrics** - 5 KPI cards
   - Total portfolio value
   - Portfolio default rate
   - High-risk accounts count
   - Average credit score
   - Portfolio health percentage

2. **Risk Distribution** - Histogram + Pie chart
   - Shows spread of risk scores
   - Segmented by risk tier

3. **Correlation Matrix** - Feature relationships
   - Risk score vs. Credit score
   - Risk score vs. Debt ratio
   - Etc.

4. **Risk Drivers Analysis** - Scatter plots with trends
   - Credit score impact
   - Debt ratio impact

5. **Demographic Insights** - Bar charts
   - Risk by age group
   - Risk by income bracket

6. **Time Series Trends** - Dual-axis line chart
   - Portfolio risk over time
   - High-risk account count trend

---

### 4️⃣ Dark Mode + Responsive Design

**Location:** `app.py` (custom_css section)

**Features:**
- Complete dark theme (slate/blue palette)
- Responsive grid layouts
- Smooth animations
- Hover effects
- Mobile-optimized
- Custom scrollbars

**CSS Variables:**
```css
--primary-color: #3b82f6           /* Blue */
--success-color: #22c55e           /* Green */
--warning-color: #eab308           /* Amber */
--danger-color: #ef4444            /* Red */
--background-dark: #0f172a         /* Dark slate */
--background-secondary: #1e293b    /* Medium slate */
--text-primary: #e2e8f0            /* Light slate */
--text-secondary: #94a3b8          /* Medium slate */
```

**Responsive Breakpoints:**
```css
@media (max-width: 768px) {
    /* Tablet/mobile styles */
    h1 { font-size: 1.875rem; }
    st.columns(2) → st.columns(1)
}
```

**Customization:**
- Change primary color: Update `--primary-color` in `:root`
- Modify background: Update gradient in `.stApp`
- Adjust font sizes: Edit h1, h2, h3 rules

---

## 📊 Data Flow

```
CSV Upload → Validation → Prediction Model → Risk Scoring
                                    ↓
                           Database/Memory
                                    ↓
     ┌──────────────────┬──────────────┬──────────────┐
     ↓                  ↓              ↓              ↓
  Gauge Widget    Batch Results   Dashboard      Exports
  (Single View)    (Table)        (Analytics)    (CSV/JSON)
```

---

## 🔧 Configuration & Customization

### Custom Risk Calculation

Edit `predict_default_risk()` in `batch_processor.py`:

```python
def predict_default_risk(row):
    score = 0
    
    # Adjust weights (must sum to 1.0)
    score += (750 - row['credit_score']) / 10 * 0.4   # 40% weight
    score += max(0, row['debt_ratio'] - 30) / 70 * 0.3   # 30% weight
    score += max(0, (45 - row['age']) / 35) * 0.15      # 15% weight
    score += max(0, (80000 - row['income']) / 80000) * 0.15  # 15% weight
    
    risk_percentage = max(0, min(100, score * 100))
    return risk_percentage
```

### Custom Risk Thresholds

Edit `categorize_risk()` in `batch_processor.py`:

```python
def categorize_risk(score):
    if score < 40:        # Change from 30
        return 'Low'
    elif score < 75:      # Change from 70
        return 'Medium'
    else:
        return 'High'
```

### Custom Color Scheme

In `app.py`, update CSS variables:
```css
:root {
    --primary-color: #your-color;
    --success-color: #your-color;
    --warning-color: #your-color;
    --danger-color: #your-color;
}
```

---

## 🚀 Deployment

### Streamlit Cloud
```bash
# Push to GitHub, then:
# 1. Go to https://streamlit.io/cloud
# 2. Connect your GitHub repo
# 3. Select app.py
# 4. Deploy
```

### Docker
Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

```bash
docker build -t credit-risk-app .
docker run -p 8501:8501 credit-risk-app
```

### Heroku / Other Platforms
Create `setup.sh`:
```bash
mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"your-email@example.com\"\n\
" > ~/.streamlit/credentials.json

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
" > ~/.streamlit/config.toml
```

---

## 🧪 Testing Checklist

- [ ] App loads without errors
- [ ] All pages are accessible via sidebar
- [ ] Dark mode is active and colors are correct
- [ ] Responsive design works on mobile (test with DevTools)
- [ ] Single prediction generates gauge and metrics
- [ ] CSV upload processes file correctly
- [ ] Batch processor shows progress bar
- [ ] Dashboard displays all charts
- [ ] Exports download successfully
- [ ] Filters and sorting work

---

## 🐛 Troubleshooting

### Issue: Charts not rendering
**Solution:** Ensure Plotly is installed
```bash
pip install --upgrade plotly
```

### Issue: Dark mode not applying
**Solution:** Clear Streamlit cache
```bash
streamlit cache clear
```

### Issue: CSV upload fails
**Solution:** Verify column names match expected names:
- `age`, `income`, `credit_score`, `debt_ratio`

### Issue: Performance is slow
**Solution:** 
1. Reduce CSV size for batch processing
2. Enable Streamlit's caching:
```python
@st.cache_data
def load_data():
    return pd.read_csv('data.csv')
```

---

## 📈 Next Steps for Enhancement

1. **Connect Real Model** - Replace prediction function with your trained model
2. **Add Database** - Store predictions in PostgreSQL/MongoDB
3. **API Integration** - Create REST API for external access
4. **Advanced Analytics** - Add ML explainability (SHAP, LIME)
5. **User Management** - Add authentication and role-based access
6. **Alerts & Notifications** - Email/Slack integration for high-risk alerts
7. **A/B Testing** - Test different model versions
8. **Historical Tracking** - Compare predictions over time

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review Streamlit docs: https://docs.streamlit.io
3. File an issue on GitHub

---

## 📄 File Structure

```
credit-risk-app/
├── app.py                      # Main Streamlit app
├── batch_processor.py          # Batch processing component
├── portfolio_dashboard.py       # Dashboard component
├── risk_gauge_widget.html      # Gauge visualization
├── requirements.txt            # Python dependencies
├── .streamlit/
│   └── config.toml            # Streamlit configuration
├── data/
│   └── sample_data.csv        # Sample data for testing
└── README.md                   # Project documentation
```

---

## 🎉 You're Ready!

You now have a professional, enterprise-grade credit risk assessment platform with:
- Beautiful dark UI
- Real-time predictions
- Batch processing
- Advanced analytics
- Responsive design
- Export capabilities

Enjoy! 🚀
