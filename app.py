import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import warnings

warnings.filterwarnings("ignore")

# ─── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="HR Attrition Prediction",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {font-size:2.2rem; font-weight:800; color:#1E3A5F; margin-bottom:0.2rem;}
    .sub-header  {font-size:1.0rem; color:#666666; margin-bottom:1.5rem;}
    .metric-card {background:#414141; border-radius:12px; padding:1rem; 
                  border-left:5px solid #1565C0; margin-bottom:0.5rem;}
    .risk-high   {color:#ffff00; font-weight:bold; font-size:1.5rem;}
    .risk-low    {color:#ffff00; font-weight:bold; font-size:1.5rem;}
    .insight-box {background:#414141; border-radius:8px; padding:1rem; 
                  border-left:4px solid #FF6F00; margin:0.5rem 0;}
</style>
""", unsafe_allow_html=True)

# ─── Load Model & Data ───────────────────────────────────────
@st.cache_resource
def load_assets():
    # Loading the saved model artifacts
    # Note: Using the path provided by the user
    artifacts = joblib.load(r"saved_model\logistic_regression_model.pkl")
    
    # Based on our previous conversation, artifacts is expected to be a dict
    # If the user saved JUST the model, we load it directly
    if isinstance(artifacts, dict):
        model = artifacts['model']
        scaler = artifacts.get('scaler')
        feature_cols = artifacts.get('features')
    else:
        model = artifacts
        scaler = None
        # Use the feature columns defined in your training script
        feature_cols = ['Age', 'MonthlyIncome', 'OverTime', 'DistanceFromHome', 'YearsAtCompany', 'YearsSinceLastPromotion', 'JobSatisfaction', 'EnvironmentSatisfaction', 'StockOptionLevel', 'PromotionGap', 'TenureRatio', 'IncomePerYearExp', 'Department_Sales', 'Department_Research & Development', 'Department_Human Resources', 'JobRole_Sales Executive', 'JobRole_Research Scientist', 'JobRole_Laboratory Technician', 'JobRole_Manufacturing Director', 'JobRole_Healthcare Representative', 'JobRole_Manager', 'JobRole_Sales Representative', 'JobRole_Research Director', 'JobRole_Human Resources', 'MaritalStatus_Single', 'MaritalStatus_Married', 'MaritalStatus_Divorced', 'BusinessTravel_Non-Travel', 'BusinessTravel_Travel_Rarely', 'BusinessTravel_Travel_Frequently']
        
    # Load raw data for HR Insights calculations
    df = pd.read_csv(r"Dataset\Employee-Attrition-dataset.csv")
    return model, scaler, feature_cols, df

try:
    model, scaler, feature_cols, df = load_assets()
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.stop()

# ─── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.title("HR Predictor")
    page = st.radio("Go to", ["🔮 Predict Attrition", "💡 HR Insights","📊 Overview & EDA"])

if page == "📊 Overview & EDA":
    st.markdown('<p class="main-header">👥 HR Employee Attrition Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Understand, predict, and reduce employee turnover</p>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Employees", f"{len(df):,}")
    c2.metric("Employees Left", df["Attrition"].eq("Yes").sum())
    c3.metric("Attrition Rate", f"{df['Attrition'].eq('Yes').mean()*100:.1f}%")
    c4.metric("Avg Monthly Income", f"${df['MonthlyIncome'].mean():,.0f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(df, names="Attrition", title="Attrition Distribution",
                     color="Attrition", color_discrete_map={"Yes":"#EF5350","No":"#42A5F5"},
                     hole=0.4)
        st.plotly_chart(fig,width='stretch')

    with col2:
        feature = st.selectbox("Select a feature to analyse vs Attrition:",
                               ["Department","JobRole","OverTime","BusinessTravel",
                                "MaritalStatus","EducationField","Gender"])
        temp = (df.groupby(feature)["Attrition"]
                  .apply(lambda x: (x=="Yes").mean()*100)
                  .reset_index()
                  .rename(columns={"Attrition":"Attrition Rate (%)"}))
        fig2 = px.bar(temp, x=feature, y="Attrition Rate (%)",
                      title=f"Attrition Rate by {feature}",
                      color="Attrition Rate (%)", color_continuous_scale="RdYlGn_r")
        st.plotly_chart(fig2,width='stretch')

    # Scatter
    st.subheader("Income vs Age coloured by Attrition")
    fig3 = px.scatter(df, x="Age", y="MonthlyIncome", color="Attrition",
                      color_discrete_map={"Yes":"#EF5350","No":"#42A5F5"},
                      opacity=0.6, hover_data=["JobRole","Department","OverTime"])
    st.plotly_chart(fig3, width='stretch')
# ─── Page 1: Predict Attrition ─────────────────────────────────
if page == "🔮 Predict Attrition":
    st.markdown('<p class="main-header">🔮 Employee Attrition Risk Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Input employee attributes to estimate turnover probability</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        age            = st.slider("Age", 18, 60, 30)
        monthly_income = st.number_input("Monthly Income ($)", 1000, 20000, 5000, 500)
        overtime       = st.selectbox("Works OverTime?", ["No","Yes"])
        distance       = st.slider("Distance from Home (km)", 1, 30, 10)
    with col2:
        dept           = st.selectbox("Department", ["Sales","Research & Development","Human Resources"])
        job_role       = st.selectbox("Job Role", 
                                       ["Sales Executive","Research Scientist","Laboratory Technician",
                                        "Manufacturing Director","Healthcare Representative",
                                        "Manager","Sales Representative","Research Director","Human Resources"])
        marital_status = st.selectbox("Marital Status", ["Single","Married","Divorced"])
        business_travel= st.selectbox("Business Travel", ["Non-Travel","Travel_Rarely","Travel_Frequently"])
    with col3:
        yrs_company    = st.slider("Years at Company", 0, 40, 5)
        yrs_promotion  = st.slider("Years Since Last Promotion", 0, 15, 2)
        job_satisfaction=st.selectbox("Job Satisfaction (1=Low, 4=High)", [1,2,3,4], index=2)
        env_satisfaction=st.selectbox("Environment Satisfaction (1=Low, 4=High)", [1,2,3,4], index=2)
        stock_option   = st.selectbox("Stock Option Level", [0,1,2,3], index=1)

    if st.button("🔮 Predict Attrition Risk", type="primary"):
        # Initialize input dictionary with zeros
        input_dict = {col: 0 for col in feature_cols}
        
        # Numeric & Manual Binary mapping
        input_dict["Age"]                     = age
        input_dict["MonthlyIncome"]            = monthly_income
        input_dict["OverTime"]                = 1 if overtime=="Yes" else 0
        input_dict["DistanceFromHome"]         = distance
        input_dict["YearsAtCompany"]           = yrs_company
        input_dict["YearsSinceLastPromotion"]  = yrs_promotion
        input_dict["JobSatisfaction"]          = job_satisfaction
        input_dict["EnvironmentSatisfaction"]  = env_satisfaction
        input_dict["StockOptionLevel"]         = stock_option
        
        # Engineering the same features as training
        input_dict["PromotionGap"]             = yrs_company - yrs_promotion
        input_dict["TenureRatio"]              = min(yrs_company/max(age-22,1), 1)
        input_dict["IncomePerYearExp"]         = monthly_income/max(age-22,1)

        # One-Hot Encoding Mappings
        for cat_val, prefix in [(dept, "Department"), (job_role, "JobRole"), 
                                (marital_status, "MaritalStatus"), (business_travel, "BusinessTravel")]:
            col_name = f"{prefix}_{cat_val}"
            if col_name in input_dict:
                input_dict[col_name] = 1

        # Create DataFrame
        input_df = pd.DataFrame([input_dict])[feature_cols]

        # Apply Scaling if the scaler was saved
        if scaler:
            input_processed = scaler.transform(input_df)
        else:
            input_processed = input_df

        # Prediction using Logistic Regression probabilities
        prob     = model.predict_proba(input_processed)[0][1]
        risk_pct = prob * 100

        st.markdown("---")
        col_r1, col_r2 = st.columns([1, 2])
        with col_r1:
            risk_class = "risk-high" if risk_pct > 40 else "risk-low"
            st.markdown(f"""
            <div class="metric-card">
                <h3>Attrition Risk Score</h3>
                <p class="{risk_class}">{risk_pct:.1f}%</p>
                <p>{'🔴 HIGH RISK — Immediate action advised' if risk_pct > 40 
                    else '🟡 MEDIUM RISK — Monitor closely' if risk_pct > 20 
                    else '🟢 LOW RISK — Employee likely to stay'}</p>
            </div>
            """, unsafe_allow_html=True)

        with col_r2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_pct,
                domain={"x":[0,1],"y":[0,1]},
                title={"text":"Attrition Risk (%)"},
                gauge={
                    "axis":{"range":[0,100]},
                    "bar":{"color":"darkred" if risk_pct>40 else "orange" if risk_pct>20 else "green"},
                    "steps":[
                        {"range":[0,20],  "color":"#67FC49"},
                        {"range":[20,40], "color":"#E4D86B"},
                        {"range":[40,100],"color":"#FFCDD2"}
                    ],
                    "threshold":{"line":{"color":"red","width":4},"value":40}
                }
            ))
            fig_gauge.update_layout(height=280)
            st.plotly_chart(fig_gauge,width='stretch')

        # Recommendations
        st.subheader("📋 HR Recommendations")
        recs = []
        if overtime == "Yes":       recs.append("⚠️ Employee works overtime — review workload and staffing.")
        if monthly_income < 3000:   recs.append("💰 Below-market compensation — consider salary review.")
        if yrs_promotion > 3:      recs.append("📈 Stagnant growth (3+ years since promotion) — discuss career path.")
        if job_satisfaction <= 2:  recs.append("😟 Low job satisfaction — conduct 1-on-1 feedback session.")
        if distance > 20:          recs.append("🚗 Long commute — consider remote/hybrid options.")
        
        for r in recs: st.markdown(f'<div class="insight-box">{r}</div>', unsafe_allow_html=True)

# ─── Page 2: HR Insights ──────────────────────────────────────
elif page == "💡 HR Insights":
    st.markdown('<p class="main-header">💡 Key HR Insights & Action Plan</p>', unsafe_allow_html=True)
    
    # Calculate live insights from the dataset
    insights = [
        ("🔴","OverTime is the Major Driver", 
         f"Employees on overtime leave at {df[df['OverTime']=='Yes']['Attrition'].eq('Yes').mean()*100:.1f}% rate vs {df[df['OverTime']=='No']['Attrition'].eq('Yes').mean()*100:.1f}% for others."),
        ("🔴","Salary Benchmarking Required", 
         f"The median income for employees who left was ${df[df['Attrition']=='Yes']['MonthlyIncome'].median():,.0f}."),
        ("🟢","Stock Options Retention Power", 
         f"Attrition drops to {df[df['StockOptionLevel']>0]['Attrition'].eq('Yes').mean()*100:.1f}% when stock options are provided."),
    ]

    for emoji, title, desc in insights:
        st.markdown(f"""
        <div class="insight-box">
            <strong>{emoji} {title}</strong><br>
            {desc}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📊 Attrition Trends")
    factor = st.selectbox("Select Factor to Visualize:", ["OverTime", "JobRole", "StockOptionLevel", "BusinessTravel"])
    
    rate_df = (df.groupby(factor)["Attrition"]
                 .apply(lambda x: (x=="Yes").mean()*100)
                 .reset_index()
                 .rename(columns={"Attrition":"Attrition Rate (%)"}))
    
    fig = px.bar(rate_df, x=factor, y="Attrition Rate (%)", 
                 color="Attrition Rate (%)", color_continuous_scale="RdYlGn_r",
                 title=f"Attrition Rate by {factor}")
    st.plotly_chart(fig,width='stretch')
