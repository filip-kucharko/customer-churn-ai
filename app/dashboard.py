import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os
from dotenv import load_dotenv
from openai import OpenAI

# Page config
st.set_page_config(
    page_title="Churn Prevention System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment
load_dotenv()

# Load Custom CSS
def load_css():
    """Load custom CSS styling"""
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# Plotly Dark Theme Template
PLOTLY_TEMPLATE = {
    'layout': {
        'plot_bgcolor': '#1a1f3a',
        'paper_bgcolor': '#1a1f3a',
        'font': {'color': '#e4e6eb', 'family': 'Inter, sans-serif'},
        'title': {'font': {'size': 20, 'color': '#e4e6eb'}},
        'xaxis': {
            'gridcolor': '#2d3454',
            'linecolor': '#2d3454',
            'zerolinecolor': '#2d3454',
            'tickfont': {'color': '#b0b3c1'}
        },
        'yaxis': {
            'gridcolor': '#2d3454',
            'linecolor': '#2d3454',
            'zerolinecolor': '#2d3454',
            'tickfont': {'color': '#b0b3c1'}
        },
        'colorway': ['#4a9eff', '#00d4aa', '#ff9f43', '#a78bfa', '#fb7185']
    }
}

# Load models and data
@st.cache_resource
def load_models_and_data():
    """Load all necessary models and data."""
    model = joblib.load('../models/final_churn_model.pkl')
    scaler = joblib.load('../models/scaler.pkl')
    X_test = np.load('../data/processed/X_test.npy')
    y_test = np.load('../data/processed/y_test.npy')
    
    with open('../data/processed/feature_names.txt', 'r') as f:
        feature_names = f.read().splitlines()
    
    return model, scaler, X_test, y_test, feature_names

model, scaler, X_test, y_test, feature_names = load_models_and_data()

# Initialize OpenAI client
@st.cache_resource
def init_openai_client():
    """Initialize OpenAI client."""
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        return OpenAI(api_key=api_key)
    return None

client = init_openai_client()

def generate_strategy(customer_profile, churn_prob, top_drivers):
    """Generate AI retention strategy."""
    if not client:
        return "OpenAI API key not configured. Please set OPENAI_API_KEY in .env file."
    
    # Calculate base success rate
    if churn_prob < 0.3:
        base_rate = 60
    elif churn_prob < 0.5:
        base_rate = 50
    elif churn_prob < 0.7:
        base_rate = 40
    else:
        base_rate = 30
    
    drivers_text = "\n".join([f"• {feature}: {importance:.3f} impact" 
                              for feature, importance in top_drivers[:3]])
    
    charges_per_service = customer_profile.get('charges_per_service', 0)
    value_perception = "Poor" if charges_per_service > 30 else "Good" if charges_per_service < 20 else "Fair"
    
    system_prompt = """You are a Customer Retention Strategist for a telecommunications company.

Create actionable retention strategies with these constraints:
- Budget: €50-100 per customer
- Implementation timeframe: 7 days
- Focus on value proposition and customer needs

Provide:
1. Primary Action (one sentence)
2. Business Rationale (2-3 sentences)
3. Expected Success Rate (percentage with brief justification)
4. Estimated Cost (breakdown in euros)

Adjust the base success rate by ±10% based on strategy quality. Keep rates between 25-65%."""
    
    user_prompt = f"""Customer Profile:
- Lifecycle Stage: {customer_profile.get('lifecycle_stage', 'N/A')}
- Tenure: {customer_profile.get('tenure', 'N/A')} months
- Monthly Charges: €{customer_profile.get('MonthlyCharges', 'N/A'):.2f}
- Services Used: {customer_profile.get('service_count', 'N/A')}/8
- Value Perception: {value_perception} (€{charges_per_service:.2f}/service)
- Contract: {customer_profile.get('Contract', 'N/A')}

Churn Risk: {churn_prob:.1%}
Base Success Rate: {base_rate}%

Key Risk Drivers:
{drivers_text}

Generate retention strategy."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=350
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating strategy: {e}"

# Helper Functions
def get_customer_profile(customer_idx, X_test, feature_names):
    """Extract customer profile with real values."""
    customer_scaled = X_test[customer_idx].reshape(1, -1)
    customer_unscaled = scaler.inverse_transform(customer_scaled)[0]
    feature_dict = dict(zip(feature_names, customer_unscaled))
   
    profile = {
        'tenure': int(feature_dict.get('tenure', 0)),
        'MonthlyCharges': float(feature_dict.get('MonthlyCharges', 0)),
        'service_count': int(feature_dict.get('service_count', 0)),
        'charges_per_service': float(feature_dict.get('charges_per_service', 0)),
        'Contract': 'Month-to-month' if feature_dict.get('is_month_to_month', 0) > 0.5 else 'Long-term',
        'PaymentMethod': 'Electronic check' if feature_dict.get('PaymentMethod', 0) else 'Other',
    }
   
    tenure_val = profile['tenure']
    if tenure_val <= 12:
        profile['lifecycle_stage'] = 'New'
    elif tenure_val <= 24:
        profile['lifecycle_stage'] = 'Growing'
    elif tenure_val <= 48:
        profile['lifecycle_stage'] = 'Mature'
    else:
        profile['lifecycle_stage'] = 'Loyal'
   
    return profile, feature_dict

# ==================== SIDEBAR ====================
st.sidebar.title("Customer Churn Prevention System")
page = st.sidebar.radio(
    "Select Page",
    ["Executive Dashboard", "Customer Risk Analysis", "Strategy Generator"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.subheader("System Metrics")
total_customers = len(X_test)
predictions = model.predict_proba(X_test)[:, 1]
high_risk_count = (predictions >= 0.6).sum()

st.sidebar.metric("Total Customers", f"{total_customers:,}")
st.sidebar.metric("High Risk (≥60%)", f"{high_risk_count:,}")
st.sidebar.metric("Model Recall", "69.5%")
st.sidebar.metric("Campaign ROI", "634%")

# Helper Function for Charts
def create_gauge_chart(value, title):
    """Create a minimal gauge chart - NO TEXT"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={'suffix': "%", 'font': {'size': 48, 'color': '#e4e6eb'}},
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 0,
                'tickcolor': "#1a1f3a",
                'tickfont': {'color': '#1a1f3a'},  # Hide ticks
                'showticklabels': False
            },
            'bar': {'color': "#ff6b6b" if value > 0.7 else "#ff9f43" if value > 0.4 else "#00d4aa"},
            'bgcolor': "#1a1f3a",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 40], 'color': '#0d3b2f'},
                {'range': [40, 70], 'color': '#4a3807'},
                {'range': [70, 100], 'color': '#4a0f0f'}
            ],
        }
    ))
    
    fig.update_layout(
        plot_bgcolor='#1a1f3a',
        paper_bgcolor='#1a1f3a',
        font={'color': '#e4e6eb'},
        height=280,
        title_text="",
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False
    )
    
    return fig

# ==================== PAGE 1: EXECUTIVE DASHBOARD ====================
if page == "Executive Dashboard":
    st.title("Dashboard")
    st.markdown("ML-powered customer retention analytics")
    
    st.markdown("---")
    
    # KPIs - Single Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Recall", "69.5%", "7 of 10 churners")
    with col2:
        st.metric("Precision", "55.1%", "55% accuracy")
    with col3:
        st.metric("ROI", "634%", "6.3x return")
    with col4:
        st.metric("Avg Cost", "€68", "per strategy")
    
    st.markdown("---")
    
    # System Overview - MINIMAL
    st.subheader("System Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Predictive Analytics**
        - Gradient Boosting (69.5% recall)
        - Optimized threshold (0.35)
        - Feature importance analysis
        
        **Feature Engineering**
        - Value perception metrics
        - Contract risk indicators
        - Lifecycle segmentation
        """)
    
    with col2:
        st.markdown("""
        **AI Strategy Generation**
        - GPT-4 personalized strategies
        - Data-driven success rates
        - Cost optimization (€50-100)
        
        **Business Metrics**
        - ROI calculation
        - Campaign analytics
        - Revenue projections
        """)
    
    # Churn Distribution
    st.markdown("---")
    st.subheader("Risk Distribution")
    
    fig = px.histogram(
        x=predictions,
        nbins=25,
        labels={'x': 'Churn Probability', 'y': 'Customer Count'},
        color_discrete_sequence=['#4a9eff']
    )
    fig.update_layout(
        **PLOTLY_TEMPLATE['layout'],
        height=350,
        margin=dict(l=40, r=40, t=20, b=40),
        bargap=0.1,
        title_text="",
        xaxis_title="",
        yaxis_title="",
        showlegend=False
    )
    # NO VLINE - completely clean
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # Key Insight - SHORTER
    st.info("""
    **Key Insight:** 100% of extreme high-risk customers (85-95% churn) are in their first 12 months. 
    This indicates an onboarding problem, not individual retention issues.
    """)

# ==================== PAGE 2: CUSTOMER RISK ANALYSIS ====================
elif page == "Customer Risk Analysis":
    st.title("Customer Risk Analysis")
    
    st.markdown("---")
    
    # Customer Selection - MINIMAL
    col1, col2 = st.columns([4, 1])
    
    with col1:
        risk_filter = st.selectbox(
            "Filter by Risk Level",
            ["All Customers", "High Risk (≥60%)", "Medium Risk (40-60%)", "Low Risk (<40%)"]
        )
        
        if risk_filter == "High Risk (≥60%)":
            filtered_indices = np.where(predictions >= 0.6)[0]
        elif risk_filter == "Medium Risk (40-60%)":
            filtered_indices = np.where((predictions >= 0.4) & (predictions < 0.6))[0]
        elif risk_filter == "Low Risk (<40%)":
            filtered_indices = np.where(predictions < 0.4)[0]
        else:
            filtered_indices = np.arange(len(predictions))
        
        # In Python-Liste umwandeln
        filtered_indices_list = filtered_indices.tolist()
        
        if len(filtered_indices_list) == 0:
            st.warning("No customers match the selected risk filter.")
            st.stop()

        # WICHTIG: Session State VOR dem selectbox initialisieren!
        if 'customer_idx' not in st.session_state:
            st.session_state.customer_idx = filtered_indices_list[0]

        # Falls der aktuell gespeicherte Index nicht mehr in der gefilterten Liste ist → zurücksetzen
        if st.session_state.customer_idx not in filtered_indices_list:
            st.session_state.customer_idx = filtered_indices_list[0]

        # Jetzt sicher selectbox erstellen
        customer_idx = st.selectbox(
            "Select Customer",
            filtered_indices_list,
            index=filtered_indices_list.index(st.session_state.customer_idx),
            format_func=lambda x: f"Customer #{x} | Risk: {predictions[x]:.1%}",
            key="customer_selector"  # Dieser key sorgt automatisch für Sync!
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Random", use_container_width=True):
            high_risk_idx = np.where(predictions >= 0.7)[0]
            if len(high_risk_idx) > 0:
                # Pick random and update session state
                st.session_state.customer_idx = int(np.random.choice(high_risk_idx))
                st.rerun()
    
    st.markdown("---")
    
    # Get customer data
    customer_profile, feature_dict = get_customer_profile(customer_idx, X_test, feature_names)
    churn_prob = predictions[customer_idx]
    actual_churn = y_test[customer_idx]
    
    # Single Row Layout - Risk + Profile
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Gauge Chart
        st.plotly_chart(
            create_gauge_chart(churn_prob, "Churn Risk"),
            use_container_width=True,
            config={'displayModeBar': False}
        )
        
        # Risk Classification
        if churn_prob >= 0.7:
            st.error("**CRITICAL RISK**")
        elif churn_prob >= 0.4:
            st.warning("**MEDIUM RISK**")
        else:
            st.success("**LOW RISK**")
        
        # Prediction vs Actual - klar und übersichtlich
        st.markdown("---")
        st.markdown("### Validation")
        
        # Calculate
        pred_label = "Churn" if churn_prob >= 0.35 else "Retain"
        actual_label = "Churned" if actual_churn == 1 else "Retained"
        is_correct = pred_label.lower() == actual_label.lower()[:len(pred_label)]
        
        # Two columns for Predicted | Actual
        val_col1, val_col2 = st.columns(2)
        
        with val_col1:
            st.markdown("<p style='color: #6b7280; font-size: 0.7rem; margin-bottom: 0.3rem; text-transform: uppercase; letter-spacing: 0.05em;'>Predicted</p>", unsafe_allow_html=True)
            if pred_label == "Churn":
                st.markdown("<p style='color: #ff6b6b; font-size: 1.3rem; font-weight: 700; margin: 0;'>Churn</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color: #00d4aa; font-size: 1.3rem; font-weight: 700; margin: 0;'>Retain</p>", unsafe_allow_html=True)
        
        with val_col2:
            st.markdown("<p style='color: #6b7280; font-size: 0.7rem; margin-bottom: 0.3rem; text-transform: uppercase; letter-spacing: 0.05em;'>Actual</p>", unsafe_allow_html=True)
            if actual_churn == 1:
                st.markdown("<p style='color: #ff6b6b; font-size: 1.3rem; font-weight: 700; margin: 0;'>Churned</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color: #00d4aa; font-size: 1.3rem; font-weight: 700; margin: 0;'>Retained</p>", unsafe_allow_html=True)
        
        # Result below both
        st.markdown("<br>", unsafe_allow_html=True)
        if is_correct:
            st.success("✓ Correct")
        else:
            if pred_label == "Churn" and actual_churn == 0:
                st.error("✗ False Positive")
            else:
                st.error("✗ False Negative")
    
    with col2:
        # Customer Profile - CLEAN GRID
        st.markdown("### Customer Profile")
        
        # 3x2 Grid
        row1_col1, row1_col2, row1_col3 = st.columns(3)
        row2_col1, row2_col2, row2_col3 = st.columns(3)
        
        with row1_col1:
            st.metric("ID", f"#{customer_idx}")
        with row1_col2:
            st.metric("Stage", customer_profile['lifecycle_stage'])
        with row1_col3:
            st.metric("Tenure", f"{customer_profile['tenure']}m")
        
        with row2_col1:
            st.metric("Services", f"{customer_profile['service_count']}/8")
        with row2_col2:
            st.metric("Monthly", f"€{customer_profile['MonthlyCharges']:.0f}")
        with row2_col3:
            contract_short = customer_profile['Contract'].replace('Month-to-month', 'M2M').replace('Long-term', 'LT')
            st.metric("Contract", contract_short)
    
    # Feature Importance 
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Key Risk Drivers")
    
    feature_importance = model.feature_importances_
    top_indices = np.argsort(feature_importance)[-10:][::-1]
    top_features = [(feature_names[i], feature_importance[i]) for i in top_indices]
    
    fig = px.bar(
        x=[imp for _, imp in top_features],
        y=[feat for feat, _ in top_features],
        orientation='h',
        labels={'x': 'Feature Importance', 'y': ''},
        color_discrete_sequence=['#4a9eff']
    )
    fig.update_layout(
        **PLOTLY_TEMPLATE['layout'],
        height=400,
        title_text="",
        margin=dict(l=150, r=40, t=40, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE 3: STRATEGY GENERATOR ====================
elif page == "Strategy Generator":
    st.title("AI-Powered Strategy Generator")
    st.markdown("Generate personalized retention strategies using machine learning and generative AI")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Customer Selection
    high_risk_indices = np.where(predictions >= 0.5)[0]
    
    customer_idx = st.selectbox(
        "Select High-Risk Customer",
        high_risk_indices,
        format_func=lambda x: f"Customer #{x} | Risk: {predictions[x]:.1%} | {get_customer_profile(x, X_test, feature_names)[0]['lifecycle_stage']}"
    )
    
    # Get customer data
    customer_profile, feature_dict = get_customer_profile(customer_idx, X_test, feature_names)
    churn_prob = predictions[customer_idx]
    
    # Customer Summary
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Churn Risk", f"{churn_prob:.1%}")
    col2.metric("Lifecycle", customer_profile['lifecycle_stage'])
    col3.metric("Tenure", f"{customer_profile['tenure']}m")
    col4.metric("Monthly Charges", f"€{customer_profile['MonthlyCharges']:.2f}")
    
    st.markdown("---")
    
    # Generate Strategy
    if st.button("Generate Retention Strategy", type="primary", use_container_width=True):
        if not client:
            st.error("OpenAI API not configured. Please set OPENAI_API_KEY in environment variables.")
        else:
            with st.spinner("Analyzing customer profile and generating strategy..."):
                feature_importance = model.feature_importances_
                top_indices = np.argsort(feature_importance)[-5:][::-1]
                top_drivers = [(feature_names[i], feature_importance[i]) for i in top_indices]
                
                strategy = generate_strategy(customer_profile, churn_prob, top_drivers)
                
                st.success("Strategy Generated Successfully")
                
                st.subheader("Recommended Retention Strategy")
                st.markdown(strategy)
                
                st.markdown("---")
                st.info("""
                **Note:** Success rates are AI-estimated based on customer risk profile and strategy effectiveness. 
                These estimates should be validated through A/B testing and updated with actual campaign performance data.
                """)
                
                # Export
                st.download_button(
                    label="Export Strategy as Text",
                    data=strategy,
                    file_name=f"retention_strategy_customer_{customer_idx}.txt",
                    mime="text/plain"
                )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; font-size: 0.875rem;'>
    <p>Intelligent Customer Churn Prevention System | Machine Learning & GenAI Solution</p>
    <p>Made by Filip Kucharko</p>
</div>
""", unsafe_allow_html=True)