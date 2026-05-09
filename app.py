import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle

st.set_page_config(
    page_title="SportsPulse — Football Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

    .stApp {
        background-color: #E8ECF2 !important;
        background-image:
            linear-gradient(rgba(37,99,235,0.08) 1px, transparent 1px),
            linear-gradient(90deg, rgba(37,99,235,0.08) 1px, transparent 1px) !important;
        background-size: 36px 36px !important;
        color: #0F1923;
    }
    .main { background-color: transparent !important; }
    [data-testid="stAppViewContainer"] {
        background-color: #E8ECF2 !important;
        background-image:
            linear-gradient(rgba(37,99,235,0.08) 1px, transparent 1px),
            linear-gradient(90deg, rgba(37,99,235,0.08) 1px, transparent 1px) !important;
        background-size: 36px 36px !important;
    }
    [data-testid="stAppViewBlockContainer"] {
        background: transparent !important;
    }
    .block-container { padding-top: 1.5rem; max-width: 1200px; background: transparent !important; }

    section[data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1.5px solid #E2E8F0;
        box-shadow: 2px 0 12px rgba(0,0,0,0.04);
    }
    section[data-testid="stSidebar"] .stSelectbox > div > div {
        background: #F8FAFC !important;
        border: 1.5px solid #E2E8F0 !important;
        color: #0F1923 !important;
        border-radius: 8px !important;
    }

    h1,h2,h3 {
        font-family: 'Inter', sans-serif !important;
        color: #0F1923 !important;
        font-weight: 800 !important;
        letter-spacing: -0.03em !important;
    }
    p, span, div, li { font-family: 'Inter', sans-serif !important; }

    .hero-banner {
        background: #FFFFFF;
        border: 1.5px solid #E2E8F0;
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 20px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
        background: linear-gradient(90deg, #1E40AF, #2563EB, #3B82F6);
    }
    .hero-title {
        font-family: 'Inter', sans-serif;
        font-size: 36px;
        font-weight: 900;
        color: #0F1923;
        letter-spacing: -0.04em;
        margin-bottom: 4px;
    }
    .hero-title span { color: #2563EB; }
    .hero-sub { font-size: 14px; color: #64748B; font-weight: 500; }

    .pred-card {
        background: linear-gradient(135deg, #1E40AF, #2563EB);
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        margin-top: 16px;
        box-shadow: 0 8px 32px rgba(37,99,235,0.25);
    }
    .pred-value {
        font-family: 'Inter', sans-serif;
        font-size: 56px;
        font-weight: 900;
        color: #FFFFFF;
        letter-spacing: -0.04em;
    }
    .pred-label { font-size: 13px; color: rgba(255,255,255,0.7); margin-top: 6px; }

    .stButton > button {
        background: linear-gradient(135deg, #2563EB, #1D4ED8) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 28px !important;
        font-family: 'Inter', sans-serif !important;
        box-shadow: 0 4px 14px rgba(37,99,235,0.3) !important;
    }

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1.5px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    div[data-testid="stMetric"] label {
        color: #64748B !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 10px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #2563EB !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 800 !important;
        letter-spacing: -0.03em !important;
    }

    .stTextInput > div > div > input {
        background: #FFFFFF !important;
        border: 1.5px solid #E2E8F0 !important;
        border-radius: 10px !important;
        color: #0F1923 !important;
    }
    .stSlider > div > div > div { background: #2563EB !important; }

    .sp-divider { height: 1.5px; background: linear-gradient(90deg,transparent,#E2E8F0,transparent); margin: 20px 0; }
    .sp-divider-blue { height: 1.5px; background: linear-gradient(90deg,transparent,#BFDBFE,transparent); margin: 20px 0; }

    .tech-badge {
        display: inline-block;
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 6px;
        padding: 4px 10px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: #1D4ED8;
        margin: 3px;
        font-weight: 500;
    }

    .stDataFrame { border-radius: 12px; overflow: hidden; }
    #MainMenu,footer,header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

BG     = "#F4F6F9"
CARD   = "#FFFFFF"
BLUE   = "#2563EB"
DARK   = "#0F1923"
GRAY   = "#64748B"
FONT   = "Inter, sans-serif"

color_map = {
    "Forward":    "#2563EB",
    "Midfielder": "#0EA5E9",
    "Defender":   "#10B981",
    "Goalkeeper": "#F59E0B",
}

def sp_layout(fig, title, height=420):
    fig.update_layout(
        title=dict(text=title, font=dict(size=18,color=DARK,family=FONT), x=0.02),
        font=dict(color=DARK, family=FONT, size=13),
        plot_bgcolor=CARD, paper_bgcolor=CARD,
        height=height,
        margin=dict(t=60,l=60,r=30,b=60),
        legend=dict(font=dict(size=12,color=DARK), bgcolor="rgba(255,255,255,0.95)", bordercolor="#E2E8F0", borderwidth=1),
    )
    fig.update_xaxes(tickfont=dict(size=12,color=GRAY,family=FONT), title_font=dict(size=13,color=GRAY,family=FONT), gridcolor="#F1F5F9", linecolor="#E2E8F0")
    fig.update_yaxes(tickfont=dict(size=12,color=GRAY,family=FONT), title_font=dict(size=13,color=GRAY,family=FONT), gridcolor="#F1F5F9", linecolor="#E2E8F0")
    return fig

@st.cache_data
def load_data():
    return pd.read_csv("data/fifa23_clean.csv")

@st.cache_resource
def load_model():
    model    = pickle.load(open("models/xgb_model.pkl","rb"))
    le_pos   = pickle.load(open("models/le_pos.pkl","rb"))
    features = pickle.load(open("models/features.pkl","rb"))
    return model, le_pos, features

with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 14px;border-bottom:1.5px solid #E2E8F0;margin-bottom:16px;'>
        <div style='font-family:Inter,sans-serif;font-size:22px;font-weight:900;color:#0F1923;letter-spacing:-0.04em;'>Sport<span style='color:#2563EB;'>Pulse</span></div>
        <div style='font-size:11px;color:#64748B;margin-top:2px;font-weight:500;'>Football Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.selectbox("NAVIGATE",[
        "🔮 Value Predictor",
        "📊 Analytics Dashboard",
        "🔍 Player Explorer",
    ])

    st.markdown('<div class="sp-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#F8FAFC;border:1.5px solid #E2E8F0;border-radius:12px;padding:14px;'>
        <div style='font-family:JetBrains Mono,monospace;font-size:9px;color:#94A3B8;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:10px;'>Model info</div>
        <div style='font-size:13px;color:#0F1923;line-height:2.2;font-weight:500;'>
            <span style='color:#94A3B8;font-weight:400;'>Algorithm:</span> XGBoost<br>
            <span style='color:#94A3B8;font-weight:400;'>R² Score:</span> <span style='color:#2563EB;font-weight:700;'>0.9994</span><br>
            <span style='color:#94A3B8;font-weight:400;'>Players:</span> 165,854<br>
            <span style='color:#94A3B8;font-weight:400;'>MAE:</span> €0.08M<br>
            <span style='color:#94A3B8;font-weight:400;'>Features:</span> 13
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sp-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center;'>
        <div style='font-family:JetBrains Mono,monospace;font-size:9px;color:#94A3B8;letter-spacing:0.08em;text-transform:uppercase;'>Built by</div>
        <div style='font-family:Inter,sans-serif;font-size:14px;font-weight:700;color:#0F1923;margin-top:4px;letter-spacing:-0.02em;'>Vishaal Pedapatnam</div>
        <div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#94A3B8;margin-top:2px;'>MS Data Science · SUNY Buffalo</div>
    </div>
    """, unsafe_allow_html=True)

if page == "🔮 Value Predictor":
    st.markdown("""
    <div class='hero-banner'>
        <div class='hero-title'>⚽ Sport<span>Pulse</span></div>
        <div class='hero-sub'>Predict any football player's market value using XGBoost AI · R²=0.9994</div>
    </div>
    """, unsafe_allow_html=True)

    model, le_pos, features = load_model()
    st.markdown("### Set player attributes")
    c1,c2,c3 = st.columns(3)
    with c1:
        overall   = st.slider("Overall rating",   40,99,75)
        potential = st.slider("Potential rating", 40,99,80)
        age       = st.slider("Age",              16,45,24)
        pace      = st.slider("Pace",             20,99,70)
    with c2:
        shooting  = st.slider("Shooting",  20,99,65)
        passing   = st.slider("Passing",   20,99,65)
        dribbling = st.slider("Dribbling", 20,99,70)
        defending = st.slider("Defending", 10,99,40)
    with c3:
        physic       = st.slider("Physicality", 20,99,65)
        height_cm    = st.slider("Height (cm)", 155,210,180)
        weight_kg    = st.slider("Weight (kg)", 55,110,75)
        league_level = st.selectbox("League level",[1,2,3,4,5],index=0)
        position     = st.selectbox("Position group",["Forward","Midfielder","Defender","Goalkeeper"])

    if st.button("⚽ Predict Market Value"):
        pos_encoded = le_pos.transform([position])[0]
        input_data  = pd.DataFrame([[
            overall,potential,age,height_cm,weight_kg,
            pace,shooting,passing,dribbling,defending,
            physic,pos_encoded,league_level
        ]],columns=features)
        prediction = max(0.01, model.predict(input_data)[0])

        st.markdown(f"""
        <div class='pred-card'>
            <div style='font-family:JetBrains Mono,monospace;font-size:11px;color:rgba(255,255,255,0.6);letter-spacing:0.12em;text-transform:uppercase;margin-bottom:10px;'>Predicted Market Value</div>
            <div class='pred-value'>€{prediction:.1f}M</div>
            <div class='pred-label'>XGBoost · R²=0.9994 · 165,854 FIFA 23 players</div>
        </div>
        """, unsafe_allow_html=True)

        df = load_data()
        pct = (df["value_eur_m"] < prediction).mean() * 100
        st.markdown(f"""
        <div style='background:#EFF6FF;border:1.5px solid #BFDBFE;border-radius:12px;padding:16px;margin-top:12px;text-align:center;'>
            <div style='font-size:14px;color:#0F1923;font-weight:500;'>
                This player is worth more than
                <span style='color:#2563EB;font-weight:800;font-size:20px;'> {pct:.0f}%</span>
                of all FIFA 23 players
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sp-divider-blue"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center;padding:8px;'>
        <span class='tech-badge'>XGBoost</span>
        <span class='tech-badge'>scikit-learn</span>
        <span class='tech-badge'>165,854 players</span>
        <span class='tech-badge'>R²=0.9994</span>
        <span class='tech-badge'>FIFA 23</span>
    </div>
    """, unsafe_allow_html=True)

elif page == "📊 Analytics Dashboard":
    st.markdown("""
    <div class='hero-banner'>
        <div class='hero-title'>📊 Analytics Dashboard</div>
        <div class='hero-sub'>Market intelligence across 165,854 FIFA 23 players</div>
    </div>
    """, unsafe_allow_html=True)

    df = load_data()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Players", f"{len(df):,}")
    c2.metric("Avg Value",     f"€{df['value_eur_m'].mean():.2f}M")
    c3.metric("Highest Value", f"€{df['value_eur_m'].max():.1f}M")
    c4.metric("Avg Overall",   f"{df['overall'].mean():.1f}")

    st.markdown('<div class="sp-divider"></div>', unsafe_allow_html=True)

    top20  = df.nlargest(20,"value_eur_m")[["short_name","value_eur_m","position_group"]].sort_values("value_eur_m")
    colors = [color_map.get(p,"#64748B") for p in top20["position_group"]]
    fig1   = go.Figure(go.Bar(y=top20["short_name"],x=top20["value_eur_m"],orientation="h",marker_color=colors,text=[f"€{v:.1f}M" for v in top20["value_eur_m"]],textposition="outside",textfont=dict(size=11,color=DARK,family=FONT)))
    fig1   = sp_layout(fig1,"💰 Top 20 Most Valuable Players — FIFA 23",560)
    fig1.update_xaxes(title="Market Value (€M)")
    st.plotly_chart(fig1,use_container_width=True)

    col1,col2 = st.columns(2)
    with col1:
        pos_s = df.groupby("position_group")["value_eur_m"].agg(["mean","median"]).reset_index()
        pos_s.columns = ["Position","Avg","Median"]
        fig2  = go.Figure()
        fig2.add_trace(go.Bar(name="Avg",    x=pos_s["Position"],y=pos_s["Avg"],    marker_color=BLUE,     text=[f"€{v:.1f}M" for v in pos_s["Avg"]],    textposition="outside",textfont=dict(size=12,color=DARK,family=FONT)))
        fig2.add_trace(go.Bar(name="Median", x=pos_s["Position"],y=pos_s["Median"], marker_color="#0EA5E9", text=[f"€{v:.1f}M" for v in pos_s["Median"]], textposition="outside",textfont=dict(size=12,color=DARK,family=FONT)))
        fig2.update_layout(barmode="group")
        fig2 = sp_layout(fig2,"📊 Value by Position Group",400)
        st.plotly_chart(fig2,use_container_width=True)
    with col2:
        age_s = df.groupby("age")["value_eur_m"].agg(["mean","count"]).reset_index()
        age_s = age_s[age_s["count"]>=50]
        fig4  = go.Figure()
        fig4.add_trace(go.Scatter(x=age_s["age"],y=age_s["mean"],mode="lines+markers",line=dict(color=BLUE,width=3),marker=dict(size=8,color=BLUE,line=dict(color="#FFFFFF",width=2)),fill="tozeroy",fillcolor="rgba(37,99,235,0.06)"))
        fig4  = sp_layout(fig4,"📈 Player Value by Age — Career Curve",400)
        fig4.update_xaxes(title="Age")
        fig4.update_yaxes(title="Avg Value (€M)")
        st.plotly_chart(fig4,use_container_width=True)

    df_s = df[df["value_eur_m"]>1].sample(min(3000,len(df)),random_state=42)
    fig3  = px.scatter(df_s,x="overall",y="value_eur_m",color="position_group",hover_data=["short_name"],opacity=0.65,color_discrete_map=color_map,labels={"overall":"Overall Rating","value_eur_m":"Market Value (€M)","position_group":"Position"})
    fig3.update_traces(marker=dict(size=7,line=dict(width=0.5,color="#FFFFFF")))
    fig3  = sp_layout(fig3,"⭐ Overall Rating vs Market Value",460)
    st.plotly_chart(fig3,use_container_width=True)

    model,_,features = load_model()
    imp  = pd.DataFrame({"Feature":features,"Importance":model.feature_importances_}).sort_values("Importance",ascending=True).tail(10)
    cols = [BLUE if v>=imp["Importance"].quantile(0.7) else "#0EA5E9" for v in imp["Importance"]]
    fig5 = go.Figure(go.Bar(y=imp["Feature"],x=imp["Importance"],orientation="h",marker_color=cols,text=[f"{v:.3f}" for v in imp["Importance"]],textposition="outside",textfont=dict(size=12,color=DARK,family=FONT)))
    fig5 = sp_layout(fig5,"🎯 What Determines a Player's Value?",400)
    st.plotly_chart(fig5,use_container_width=True)

    lg   = df[df["league_name"].notna()].groupby("league_name").agg(avg=("value_eur_m","mean"),cnt=("short_name","count")).reset_index()
    lg   = lg[lg["cnt"]>=100].nlargest(15,"avg").sort_values("avg")
    fig6 = go.Figure(go.Bar(y=lg["league_name"],x=lg["avg"],orientation="h",marker=dict(color=lg["avg"],colorscale=[[0,"#BFDBFE"],[0.5,"#2563EB"],[1,"#1E40AF"]],showscale=False),text=[f"€{v:.1f}M" for v in lg["avg"]],textposition="outside",textfont=dict(size=11,color=DARK,family=FONT)))
    fig6 = sp_layout(fig6,"🏆 Top 15 Leagues by Average Player Value",520)
    st.plotly_chart(fig6,use_container_width=True)

elif page == "🔍 Player Explorer":
    st.markdown("""
    <div class='hero-banner'>
        <div class='hero-title'>🔍 Player Explorer</div>
        <div class='hero-sub'>Search and compare 165,854 FIFA 23 players</div>
    </div>
    """, unsafe_allow_html=True)

    df = load_data()
    c1,c2 = st.columns(2)
    with c1:
        pos_filter  = st.selectbox("Filter by position",["All"]+sorted(df["position_group"].unique().tolist()))
    with c2:
        min_overall = st.slider("Minimum overall rating",40,99,70)

    search = st.text_input("Search player name",placeholder="e.g. Messi, Ronaldo, Mbappé...")
    st.markdown('<div class="sp-divider"></div>', unsafe_allow_html=True)

    filtered = df.copy()
    if pos_filter != "All":
        filtered = filtered[filtered["position_group"]==pos_filter]
    filtered = filtered[filtered["overall"]>=min_overall]
    if search:
        filtered = filtered[filtered["short_name"].str.contains(search,case=False,na=False)]
    filtered = filtered.sort_values("value_eur_m",ascending=False)

    c1,c2,c3 = st.columns(3)
    c1.metric("Players Found",f"{len(filtered):,}")
    c2.metric("Avg Value",    f"€{filtered['value_eur_m'].mean():.2f}M" if len(filtered)>0 else "—")
    c3.metric("Avg Overall",  f"{filtered['overall'].mean():.1f}"        if len(filtered)>0 else "—")

    st.markdown("###")
    display = ["short_name","position_group","overall","potential","age","value_eur_m","league_name"]
    st.dataframe(
        filtered[display].rename(columns={"short_name":"Player","position_group":"Position","overall":"Overall","potential":"Potential","age":"Age","value_eur_m":"Value (€M)","league_name":"League"}).reset_index(drop=True),
        use_container_width=True,height=500,
    )

st.markdown("""
<div style='background:#FFFFFF;border-top:1.5px solid #E2E8F0;padding:16px;text-align:center;margin-top:40px;border-radius:12px;box-shadow:0 -1px 6px rgba(0,0,0,0.03);'>
    <div style='font-family:JetBrains Mono,monospace;font-size:11px;color:#94A3B8;letter-spacing:0.04em;line-height:1.8;'>
        <b style='color:#0F1923;'>SPORTSPULSE</b> · Vishaal Pedapatnam · MS Data Science, SUNY Buffalo<br>
        XGBoost · R²=0.9994 · 165,854 FIFA 23 Players · scikit-learn · Plotly
    </div>
</div>
""", unsafe_allow_html=True)