"""
Churn Prediction Dashboard
============================
Interactive Streamlit app for churn predictions and analytics.

Run:  streamlit run streamlit_app/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="wide")

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "telecom_churn.csv")

@st.cache_resource
def load_artifacts():
    try:
        m = joblib.load(os.path.join(MODEL_DIR, "best_model.joblib"))
        s = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
        e = joblib.load(os.path.join(MODEL_DIR, "label_encoders.joblib"))
        with open(os.path.join(MODEL_DIR, "feature_names.json")) as f: fn = json.load(f)
        with open(os.path.join(MODEL_DIR, "model_metadata.json")) as f: md = json.load(f)
        return m, s, e, fn, md
    except:
        return None, None, None, None, None

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH) if os.path.exists(DATA_PATH) else None

model, scaler, encoders, feature_names, metadata = load_artifacts()
df = load_data()

if model is None:
    st.error("Model not found! Run `python src/pipeline.py` first.")
    st.stop()

def make_prediction(inp):
    d = pd.DataFrame([inp])
    d["tenure_bucket"] = pd.cut(d["tenure"], bins=[0,6,12,24,48,72],
                                labels=["0-6m","6-12m","1-2y","2-4y","4-6y"]).astype(str)
    d["charges_per_tenure"] = np.where(d["tenure"]>0, d["TotalCharges"]/d["tenure"], d["MonthlyCharges"])
    d["high_value"] = (d["MonthlyCharges"]>70).astype(int)
    d["contract_risk"] = (d["Contract"]=="Month-to-month").astype(int)
    d["service_count"] = (d["PhoneService"]=="Yes").astype(int) + (d["InternetService"]!="No").astype(int)
    for col, le in encoders.items():
        if col in d.columns:
            try: d[col] = le.transform(d[col])
            except: d[col] = 0
    return float(model.predict_proba(scaler.transform(d[feature_names]))[0][1])

# ── Sidebar ────────────────────────────────────────────────────
st.sidebar.title("📊 Churn Predictor")
page = st.sidebar.radio("Navigate", ["🔮 Predict", "📈 Analytics", "ℹ️ Model Info"])

# ── PREDICT PAGE ───────────────────────────────────────────────
if page == "🔮 Predict":
    st.title("Customer Churn Prediction")
    st.markdown("Enter customer details to predict churn risk.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Male","Female"])
        senior = st.selectbox("Senior Citizen", [0,1], format_func=lambda x: "Yes" if x else "No")
        partner = st.selectbox("Partner", ["Yes","No"])
        dependents = st.selectbox("Dependents", ["Yes","No"])
    with c2:
        st.subheader("Services")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone = st.selectbox("Phone Service", ["Yes","No"])
        internet = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])
        contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
    with c3:
        st.subheader("Billing")
        paperless = st.selectbox("Paperless Billing", ["Yes","No"])
        payment = st.selectbox("Payment Method", ["Electronic check","Mailed check","Bank transfer","Credit card"])
        monthly = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, step=5.0)
        total = st.number_input("Total Charges ($)", 0.0, 9000.0, float(tenure*monthly*0.95), step=50.0)

    st.markdown("---")
    if st.button("🔍 Predict Churn Risk", type="primary", use_container_width=True):
        inp = {"gender":gender,"SeniorCitizen":senior,"Partner":partner,"Dependents":dependents,
               "tenure":tenure,"PhoneService":phone,"InternetService":internet,"Contract":contract,
               "PaperlessBilling":paperless,"PaymentMethod":payment,"MonthlyCharges":monthly,"TotalCharges":total}
        prob = make_prediction(inp)
        risk = "High" if prob>0.7 else "Medium" if prob>0.4 else "Low"
        color = "#ef4444" if risk=="High" else "#f59e0b" if risk=="Medium" else "#22c55e"

        r1,r2,r3 = st.columns(3)
        with r1:
            st.markdown(f'<div style="background:{color}15;border:2px solid {color};border-radius:12px;padding:24px;text-align:center"><h1 style="color:{color};margin:0">{prob*100:.1f}%</h1><p>Churn Probability</p></div>', unsafe_allow_html=True)
        with r2:
            emoji = {"High":"🔴","Medium":"🟡","Low":"🟢"}[risk]
            st.markdown(f'<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:24px;text-align:center"><h1 style="margin:0">{emoji} {risk}</h1><p>Risk Level</p></div>', unsafe_allow_html=True)
        with r3:
            action = {"High":"Retain immediately","Medium":"Monitor closely","Low":"Standard service"}[risk]
            st.markdown(f'<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:24px;text-align:center"><h2 style="margin:0">📋</h2><p style="font-weight:600">{action}</p></div>', unsafe_allow_html=True)

        fig = go.Figure(go.Indicator(mode="gauge+number", value=prob*100, title={"text":"Churn Risk Score"},
            gauge={"axis":{"range":[0,100]},"bar":{"color":color},
                   "steps":[{"range":[0,40],"color":"#dcfce7"},{"range":[40,70],"color":"#fef9c3"},{"range":[70,100],"color":"#fee2e2"}]}))
        fig.update_layout(height=300, margin=dict(t=60,b=20))
        st.plotly_chart(fig, use_container_width=True)

        # Risk factors
        factors = []
        if contract=="Month-to-month": factors.append(("Month-to-month contract",0.85))
        if tenure<12: factors.append((f"Short tenure ({tenure}mo)",0.75))
        if monthly>70: factors.append((f"High charges (${monthly})",0.60))
        if internet=="Fiber optic": factors.append(("Fiber optic service",0.50))
        if payment=="Electronic check": factors.append(("Electronic check",0.45))
        if factors:
            st.subheader("Contributing Risk Factors")
            fig2 = px.bar(x=[f[1] for f in factors], y=[f[0] for f in factors], orientation="h",
                         color=[f[1] for f in factors], color_continuous_scale=["#22c55e","#f59e0b","#ef4444"])
            fig2.update_layout(height=250, showlegend=False, coloraxis_showscale=False, margin=dict(l=20,r=20,t=10,b=10))
            st.plotly_chart(fig2, use_container_width=True)

# ── ANALYTICS PAGE ─────────────────────────────────────────────
elif page == "📈 Analytics":
    st.title("Dataset Analytics")
    if df is None: st.warning("No data. Run pipeline.py first."); st.stop()

    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Total Customers", f"{len(df):,}")
    k2.metric("Churn Rate", f"{df['Churn'].mean()*100:.1f}%")
    k3.metric("Avg Tenure", f"{df['tenure'].mean():.0f} mo")
    k4.metric("Avg Monthly", f"${df['MonthlyCharges'].mean():.0f}")
    st.markdown("---")

    c1,c2 = st.columns(2)
    with c1:
        st.subheader("Churn by Contract")
        cd = df.groupby("Contract")["Churn"].mean().reset_index()
        cd["Churn"] *= 100
        st.plotly_chart(px.bar(cd, x="Contract", y="Churn", color="Churn",
                               color_continuous_scale=["#22c55e","#ef4444"]).update_layout(
                               showlegend=False, coloraxis_showscale=False), use_container_width=True)
    with c2:
        st.subheader("Churn by Internet")
        ci = df.groupby("InternetService")["Churn"].mean().reset_index()
        ci["Churn"] *= 100
        st.plotly_chart(px.bar(ci, x="InternetService", y="Churn", color="Churn",
                               color_continuous_scale=["#22c55e","#ef4444"]).update_layout(
                               showlegend=False, coloraxis_showscale=False), use_container_width=True)

    c3,c4 = st.columns(2)
    with c3:
        st.subheader("Monthly Charges Distribution")
        st.plotly_chart(px.histogram(df, x="MonthlyCharges", color=df["Churn"].map({0:"Retained",1:"Churned"}),
                                     nbins=40, barmode="overlay", opacity=0.7,
                                     color_discrete_map={"Retained":"#22c55e","Churned":"#ef4444"}), use_container_width=True)
    with c4:
        st.subheader("Tenure Distribution")
        st.plotly_chart(px.histogram(df, x="tenure", color=df["Churn"].map({0:"Retained",1:"Churned"}),
                                     nbins=36, barmode="overlay", opacity=0.7,
                                     color_discrete_map={"Retained":"#22c55e","Churned":"#ef4444"}), use_container_width=True)

# ── MODEL INFO PAGE ────────────────────────────────────────────
elif page == "ℹ️ Model Info":
    st.title("Model Performance")
    if metadata:
        st.subheader(f"Best Model: {metadata['model_name']}")
        if "all_results" in metadata:
            rdf = pd.DataFrame(metadata["all_results"]).T
            st.dataframe(rdf.style.highlight_max(axis=0, color="#dcfce7").format("{:.4f}"), use_container_width=True)

            cats = ["accuracy","precision","recall","f1_score","roc_auc"]
            fig = go.Figure()
            for mn, met in metadata["all_results"].items():
                vals = [met[c] for c in cats]
                fig.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=cats+[cats[0]], fill="toself", name=mn, opacity=0.6))
            fig.update_layout(polar=dict(radialaxis=dict(range=[0,1])), height=450)
            st.plotly_chart(fig, use_container_width=True)
