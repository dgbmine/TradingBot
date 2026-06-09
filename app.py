import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from backtest_engine import BacktestEngine, BacktestConfig # ייבוא המנוע החדש

st.set_page_config(layout="wide", page_title="Institutional Scout Pro")

# --- CSS נשאר זהה למה שהיה לך ---
st.markdown("""
<style>
/* ... (השארתי את העיצוב שלך) ... */
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE & ROUTER
# ============================================================
for k,v in [("mode","wyckoff"),("w_sub","specific"),("vp_sub","specific"),
            ("vw_sub","specific"),("comp_sub","specific")]:
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# BACKTESTING SCREEN (חדש)
# ============================================================
@st.cache_resource
def get_engine():
    return BacktestEngine()

def screen_backtest():
    st.title("⚖️ מנוע Backtesting מוסדי")
    st.markdown("מנוע הרצה היסטורי המבוסס על 35 פקטורים מוסדיים.")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        ticker = st.text_input("הכנס טיקר לבדיקה:", value="NVDA").upper()
        if st.button("🚀 הרץ סימולציה", type="primary", use_container_width=True):
            with st.spinner("מחשב..."):
                engine = get_engine()
                results = engine.run(ticker)
                
                if "error" in results:
                    st.error(results["error"])
                else:
                    st.session_state.bt_results = results
    
    if "bt_results" in st.session_state:
        res = st.session_state.bt_results
        m = res["metrics"]
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Win Rate", f"{m['win_rate']:.1%}")
        c2.metric("Sharpe", m['sharpe'])
        c3.metric("Max Drawdown", f"{m['max_drawdown']:.1%}")
        c4.metric("Total Return", f"{m['total_return']:.1%}")
        
        st.plotly_chart(go.Figure(data=go.Scatter(y=res["equity"], mode='lines', name="Equity")), use_container_width=True)
        st.write("### פירוט עסקאות", res["trades"])

# ============================================================
# TOP NAV (מעודכן)
# ============================================================
st.markdown("# INSTITUTIONAL SCOUT PRO")
c1,c2,c3,c4,c5 = st.columns(5)
nav = [("wyckoff","⬛ Wyckoff"),("vp","🔮 Volume Profile"),
       ("vwap","📐 VWAP"),("composite","🏆 Composite"),("backtest","⚖️ Backtest")]
cols = [c1,c2,c3,c4,c5]
for col,(mode_key,label) in zip(cols,nav):
    with col:
        if st.button(label, use_container_width=True,
                     type="primary" if st.session_state.mode==mode_key else "secondary"):
            st.session_state.mode = mode_key; st.rerun()
st.markdown("---")

# ============================================================
# ROUTER (כולל הבאק-טסט)
# ============================================================
# (כאן הוסף את שאר הפונקציות שלך: screen_wyckoff, screen_vp וכו')
# והוסף את הבאק-טסט לתוך ה-Routes:

routes = {
    "wyckoff": screen_wyckoff, 
    "vp": screen_vp, 
    "vwap": screen_vwap, 
    "composite": screen_composite,
    "backtest": screen_backtest
}
routes[st.session_state.mode]()
