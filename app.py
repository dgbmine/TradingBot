import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import backtest_engine

st.set_page_config(layout="wide", page_title="Institutional Scout Pro")

# ============================================================
# SCAN UNIVERSE
# ============================================================
SCAN_UNIVERSE = list(dict.fromkeys([
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","JPM","JNJ",
    "V","UNH","XOM","PG","MA","HD","CVX","MRK","ABBV","PEP",
    "KO","AVGO","COST","WMT","LLY","TMO","MCD","ACN","BAC","CRM",
    "NFLX","AMD","ADBE","CSCO","ABT","TXN","NEE","DHR","RTX","QCOM",
    "HON","NKE","INTC","AMGN","PM","IBM","SBUX","INTU","GS","CAT",
    "BA","GE","SPGI","AXP","BLK","DE","ISRG","MDLZ","ADI","GILD",
    "REGN","SYK","ZTS","MMC","AON","TJX","SCHW","CB","USB","WFC",
    "C","MS","CVS","CI","SLB","EOG","OXY","COP","PSX","VLO",
    "AMT","PLD","CCI","EQIX","SPG","O","WELL","DLR",
    "FCX","NEM","GOLD","AEM","WPM","FNV","PAAS","AG",
    "PANW","CRWD","FTNT","ZS","DDOG","SNOW","MDB","NET","PLTR",
    "UBER","ABNB","COIN","SOFI","UPST",
    "F","GM","RIVN","NIO",
    "ONTO","KLAC","LRCX","AMAT","MRVL","SMCI","DELL","HPQ",
    "DIS","CMCSA","NFLX","RBLX","U","TTWO","EA",
    "DAL","UAL","AAL","LUV","FDX","UPS","XPO","ODFL",
    "DKNG","MGM","CZR","RCL","CCL","MAR","HLT",
]))

# ============================================================
# CSS DESIGN
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+Hebrew:wght@300;400;600&display=swap');
html,body,[class*="css"]{font-family:'IBM Plex Sans Hebrew',sans-serif;direction:rtl;}
h1,h2,h3,h4{font-family:'IBM Plex Mono',monospace;direction:ltr;}

.header-box{border-radius:12px;padding:24px 32px;margin-bottom:28px;color:#e0eaf4;direction:rtl;line-height:1.9;}
.header-box.wyckoff{background:linear-gradient(135deg,#0f1923,#1a2a3a);border:1px solid #2a4a6a;}
.header-box.vp     {background:linear-gradient(135deg,#160f23,#251535);border:1px solid #4a2a6a;}
.header-box.vwap   {background:linear-gradient(135deg,#0f2318,#1a3528);border:1px solid #2a6a4a;}
.header-box.composite{background:linear-gradient(135deg,#1a1208,#2a1e08);border:1px solid #6a4a1a;}
.header-box h2{font-family:'IBM Plex Mono',monospace;font-size:1.05rem;margin-bottom:12px;direction:ltr;}
.header-box.wyckoff   h2{color:#4fc3f7;}
.header-box.vp        h2{color:#ce93d8;}
.header-box.vwap      h2{color:#4caf7d;}
.header-box.composite h2{color:#ffa726;}
.header-box p{color:#b0c8e0;font-size:0.92rem;margin:6px 0;}
.tag{display:inline-block;font-family:'IBM Plex Mono',monospace;font-size:0.75rem;padding:2px 8px;border-radius:4px;margin:3px 2px;}
.tag-w{background:#1e3a5f;border:1px solid #4fc3f7;color:#4fc3f7;}
.tag-v{background:#2a1a4a;border:1px solid #ab47bc;color:#ce93d8;}
.tag-vw{background:#0f2a1a;border:1px solid #4caf7d;color:#4caf7d;}
.tag-c{background:#2a1e08;border:1px solid #ffa726;color:#ffa726;}

.score-reason-box{background:#0d1b2a;border-left:4px solid #4fc3f7;border-radius:8px;
                  padding:18px 22px;margin:10px 0;direction:rtl;color:#cde3f5;font-size:0.88rem;line-height:1.8;}
.score-reason-box.positive   {border-left-color:#26a69a;}
.score-reason-box.negative   {border-left-color:#ef5350;}
.score-reason-box.vp-positive{background:#150d20;border-left-color:#ab47bc;}
.score-reason-box.vp-negative{background:#150d20;border-left-color:#ef5350;}
.score-reason-box.vw-positive{background:#0a1a10;border-left-color:#4caf7d;}
.score-reason-box.vw-negative{background:#0a1a10;border-left-color:#ef5350;}
.score-reason-box strong{color:#fff;}

.criteria-row{display:flex;justify-content:space-between;align-items:center;
              padding:6px 0;border-bottom:1px solid #1e3040;font-size:0.84rem;}
.hit {color:#26a69a;font-weight:600;}
.miss{color:#ef5350;}

.overview-card{background:#0d1b2a;border:1px solid #2a4a6a;border-radius:10px;
               padding:18px 20px;text-align:center;direction:ltr;}
.overview-card.vp-card  {border-color:#4a2a6a;background:#120d1e;}
.overview-card.vw-card  {border-color:#1a4a2a;background:#0a150e;}
.overview-card.comp-card{border-color:#4a3a0a;background:#150f02;}
.ticker-label{font-family:'IBM Plex Mono',monospace;font-size:1.1rem;font-weight:600;margin-bottom:4px;}
.score-big{font-family:'IBM Plex Mono',monospace;font-size:2.2rem;font-weight:600;margin:6px 0;}
.verdict-label{font-size:0.78rem;color:#b0c8e0;margin-top:4px;}
.bar-bg{background:#1e3040;border-radius:4px;height:8px;margin-top:10px;overflow:hidden;}
.bar-fill{height:8px;border-radius:4px;}

.comp-breakdown{background:#0d1208;border:1px solid #2a2008;border-radius:8px;
                padding:14px 18px;margin:8px 0;direction:ltr;}
.comp-row{display:flex;align-items:center;gap:10px;margin:6px 0;}
.comp-label{font-family:'IBM Plex Mono',monospace;font-size:0.78rem;color:#b0c8e0;min-width:160px;}
.comp-bar-bg{flex:1;background:#1e2010;border-radius:4px;height:10px;overflow:hidden;}
.comp-bar-fill{height:10px;border-radius:4px;}
.comp-score{font-family:'IBM Plex Mono',monospace;font-size:0.78rem;min-width:50px;text-align:right;}

.signal-box{border-radius:10px;padding:16px 22px;text-align:center;direction:rtl;margin:12px 0;}
.signal-strong {background:#0a2a10;border:2px solid #26a69a;color:#4caf50;}
.signal-medium {background:#2a1e08;border:2px solid #ffa726;color:#ffb74d;}
.signal-weak   {background:#2a0808;border:2px solid #ef5350;color:#ef9a9a;}
.signal-box .signal-title{font-family:'IBM Plex Mono',monospace;font-size:1.1rem;font-weight:600;}
.signal-box .signal-sub  {font-size:0.82rem;margin-top:6px;opacity:0.85;}

.scan-result-row{background:#0a1520;border:1px solid #1e3040;border-radius:8px;
                 padding:12px 18px;margin:6px 0;direction:ltr;
                 font-family:'IBM Plex Mono',monospace;font-size:0.88rem;}
.scan-badge{display:inline-block;padding:3px 10px;border-radius:20px;
            font-size:0.75rem;font-weight:600;font-family:'IBM Plex Mono',monospace;}
.badge-green {background:#0d2a20;color:#26a69a;border:1px solid #26a69a;}
.badge-yellow{background:#2a1e08;color:#ffa726;border:1px solid #ffa726;}
.badge-red   {background:#2a0d0d;color:#ef5350;border:1px solid #ef5350;}
.badge-purple{background:#1e0d2a;color:#ce93d8;border:1px solid #ce93d8;}
.badge-teal  {background:#0a2010;color:#4caf7d;border:1px solid #4caf7d;}
.badge-orange{background:#2a1500;color:#ffa726;border:1px solid #ffa726;}

.disclaimer{background:#1a1206;border:1px solid #5a4010;border-radius:8px;
            padding:10px 16px;color:#a08040;font-size:0.78rem;direction:rtl;margin-top:18px;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
for k,v in [("mode","wyckoff"),("w_sub","specific"),("vp_sub","specific"),
            ("vw_sub","specific"),("comp_sub","specific"),("backtest_sub","specific")]:
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# TOP NAV (5 Columns with Backtest)
# ============================================================
st.markdown("# INSTITUTIONAL SCOUT PRO")
c1,c2,c3,c4,c5 = st.columns(5)
nav = [("wyckoff","⬛  Wyckoff"),("vp","🔮  Volume Profile"),
       ("vwap","📐  VWAP Deviation"),("composite","🏆  Composite Score"),
       ("backtest","📈  Backtest")]
cols = [c1,c2,c3,c4,c5]
for col,(mode_key,label) in zip(cols,nav):
    with col:
        if st.button(label, use_container_width=True,
                     type="primary" if st.session_state.mode==mode_key else "secondary",
                     key=f"nav_{mode_key}"):
            st.session_state.mode = mode_key; st.rerun()
st.markdown("---")

# ============================================================
# DATA INTAKE
# ============================================================
@st.cache_data(ttl=3600)
def get_data(ticker):
    try:
        df = yf.Ticker(ticker).history(period="1y")
    except Exception:
        return None
    if df is None or len(df) < 100:
        return None
    df["VOL_MEAN"]     = df["Volume"].rolling(20).mean()
    df["BODY"]         = abs(df["Close"] - df["Open"])
    df["LOWER_SHADOW"] = df[["Open","Close"]].min(axis=1) - df["Low"]
    df["RANGE"]        = df["High"] - df["Low"]
    return df

# ============================================================
# SHARED RENDERERS
# ============================================================
def gauge_color(score, mode):
    if mode=="wyckoff":   return "#26a69a" if score>=75 else "#ffa726" if score>=45 else "#ef5350"
    if mode=="vp":        return "#ab47bc" if score>=75 else "#ffa726" if score>=45 else "#ef5350"
    if mode=="vwap":      return "#4caf7d" if score>=75 else "#ffa726" if score>=45 else "#ef5350"
    return "#ffa726" if score>=75 else "#ff7043" if score>=45 else "#ef5350"

def render_gauge(score, verdict, verdict_color, mode="wyckoff"):
    bc = gauge_color(score, mode)
    steps_map = {
        "wyckoff":   [{'range':[0,44],'color':'#1a0d0d'},{'range':[44,74],'color':'#1a1206'},{'range':[74,100],'color':'#0d1a18'}],
        "vp":        [{'range':[0,44],'color':'#1a0d18'},{'range':[44,74],'color':'#1a0f2a'},{'range':[74,100],'color':'#1a0d25'}],
        "vwap":      [{'range':[0,44],'color':'#0d1a0d'},{'range':[44,74],'color':'#0f2010'},{'range':[74,100],'color':'#0a1a10'}],
        "composite": [{'range':[0,44],'color':'#1a0d08'},{'range':[44,74],'color':'#1a1008'},{'range':[74,100],'color':'#1a1205'}],
    }
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        title={'text':f"<b>Institutional Score</b><br><span style='font-size:0.82em;color:{verdict_color}'>{verdict}</span>",'font':{'size':13}},
        gauge={'axis':{'range':[0,100],'tickwidth':1,'tickcolor':"#4a6a8a"},
               'bar':{'color':bc,'thickness':0.3},
               'bgcolor':"#0d1b2a",'borderwidth':1,'bordercolor':"#2a4a6a",
               'steps':steps_map.get(mode, steps_map["wyckoff"]),
               'threshold':{'line':{'color':"#ffffff",'width':2},'thickness':0.75,'value':score}},
        number={'font':{'size':48,'color':bc},'suffix':'/100'}
    ))
    fig.update_layout(height=300,margin=dict(t=80,b=10,l=20,r=20),
                      paper_bgcolor="#0a1520",font_color="#e0eaf4")
    return fig

def render_comparison_chart(valid, mode):
    st_sorted = sorted(valid.keys(), key=lambda t: valid[t]["score"], reverse=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=st_sorted, y=[valid[t]["score"] for t in st_sorted],
        marker_color=[valid[t]["verdict_color"] for t in st_sorted],
        text=[str(valid[t]["score"]) for t in st_sorted],
        textposition="outside",
        textfont=dict(color="#e0eaf4",family="IBM Plex Mono",size=14)
    ))
    fig.update_layout(height=280,paper_bgcolor="#0a1520",plot_bgcolor="#0d1b2a",
                      font_color="#e0eaf4",font_family="IBM Plex Mono",
                      yaxis=dict(range=[0,115],gridcolor="#1e3040",title="Score"),
                      xaxis=dict(gridcolor="#1e3040"),
                      margin=dict(t=20,b=20,l=20,r=20),showlegend=False)
    return fig, st_sorted

# ============================================================
# WYCKOFF METHOD
# ============================================================
def analyze_wyckoff(df):
    score=0; criteria=[]
    high_3m=df["Close"].iloc[-65:].max(); cur=df["Close"].iloc[-1]
    dd=(high_3m-cur)/high_3m; prereq=dd>=0.12

    sc_win=df.iloc[-30:]
    sc_c=sc_win[(sc_win["Volume"]>=sc_win["VOL_MEAN"]*2.0)&(sc_win["LOWER_SHADOW"]>sc_win["BODY"]*1.2)]
    sc_found=len(sc_c)>0; sc_pts=25 if (sc_found and prereq) else 0; score+=sc_pts
    sc_idx=sc_c.index[-1] if sc_found else None
    criteria.append({"name":"Selling Climax (SC)","hit":sc_found and prereq,"points":25,"earned":sc_pts,
        "explanation":(f"זוהה SC ב-{sc_idx.strftime('%d/%m/%Y') if sc_idx else '—'}: ווליום פי {sc_c['Volume'].iloc[-1]/sc_c['VOL_MEAN'].iloc[-1]:.1f} מהממוצע עם זנב תחתון."
            if sc_found and prereq else "לא זוהה SC. "+(f"ירידה {dd*100:.1f}% מהשיא — נדרש ≥12%." if not prereq else "לא נמצא נר עם ווליום חריג וזנב תחתון ב-30 הימים."))})

    ar_found=False; ar_pts=0; ar_exp="לא זוהה AR — נדרש SC קודם."
    if sc_found and sc_idx is not None:
        post=df.loc[sc_idx:].iloc[1:11]
        if len(post)>=2:
            rally=(post["Close"].max()-df.loc[sc_idx,"Close"])/df.loc[sc_idx,"Close"]
            ar_found=rally>=0.04; ar_pts=20 if ar_found else 0; score+=ar_pts
            ar_exp=(f"זוהה AR: עלייה {rally*100:.1f}% תוך 10 ימים." if ar_found else f"לא זוהה AR: עלייה {rally*100:.1f}% בלבד (נדרש ≥4%).")
    criteria.append({"name":"Automatic Rally (AR)","hit":ar_found,"points":20,"earned":ar_pts,"explanation":ar_exp})

    avg10=df.iloc[-10:]["Volume"].mean(); gm=df["VOL_MEAN"].iloc[-1]
    ns=avg10<gm*0.7; ns_pts=20 if ns else 0; score+=ns_pts
    criteria.append({"name":"No Supply","hit":ns,"points":20,"earned":ns_pts,
        "explanation":f"ווליום ממוצע 10 ימים: {avg10/gm*100:.0f}% מהממוצע — "+("המוכרים נעלמו." if ns else "ווליום עדיין גבוה.")})

    l20=df.iloc[-20:]; pc=(l20["Close"].iloc[-1]-l20["Close"].iloc[0])/l20["Close"].iloc[0]
    vc=(l20["Volume"].iloc[-5:].mean()-l20["Volume"].iloc[:5].mean())/l20["Volume"].iloc[:5].mean()
    div=(pc<0)and(vc<-0.25); div_pts=20 if div else 0; score+=div_pts
    criteria.append({"name":"Price–Vol Divergence","hit":div,"points":20,"earned":div_pts,
        "explanation":f"מחיר: {pc*100:+.1f}% | ווליום: {vc*100:+.1f}% — "+("לחץ מכירה קורס." if div else "אין דיברגנציה.")})

    l15=df.iloc[-15:]; tr=(l15["High"].max()-l15["Low"].min())/l15["Low"].min()
    inr=tr<0.12; tr_pts=15 if inr else 0; score+=tr_pts
    criteria.append({"name":"Trading Range","hit":inr,"points":15,"earned":tr_pts,
        "explanation":f"טווח 15 ימים: {tr*100:.1f}% "+("— צבירה שקטה." if inr else "— תנודתי מדי.")})

    vd="סבירות גבוהה לאיסוף מוסדי" if score>=75 else "סימנים חלקיים" if score>=45 else "אין ראיות לאיסוף"
    vc2="#26a69a" if score>=75 else "#ffa726" if score>=45 else "#ef5350"
    return score,criteria,vd,vc2,prereq,dd

def render_wyckoff_chart(df):
    dc=df.iloc[-65:].copy()
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.7,0.3],vertical_spacing=0.04)
    fig.add_trace(go.Candlestick(x=dc.index,open=dc["Open"],high=dc["High"],low=dc["Low"],close=dc["Close"],
        increasing_line_color="#26a69a",decreasing_line_color="#ef5350",name="Price"),row=1,col=1)
    vc=["#26a69a" if c>=o else "#ef5350" for c,o in zip(dc["Close"],dc["Open"])]
    fig.add_trace(go.Bar(x=dc.index,y=dc["Volume"],marker_color=vc,name="Volume",opacity=0.8),row=2,col=1)
    fig.add_trace(go.Scatter(x=dc.index,y=dc["VOL_MEAN"],line=dict(color="#4fc3f7",width=1.5,dash="dot"),name="VolMA20"),row=2,col=1)
    fig.update_layout(height=420,paper_bgcolor="#0a1520",plot_bgcolor="#0d1b2a",font_color="#e0eaf4",
                      xaxis_rangeslider_visible=False,legend=dict(orientation="h",y=1.02,x=0),margin=dict(t=10,b=10))
    fig.update_xaxes(gridcolor="#1e3040"); fig.update_yaxes(gridcolor="#1e3040")
    return fig

def screen_wyckoff():
    st.markdown("""
    <div class="header-box wyckoff">
      <h2>⬛ WYCKOFF ACCUMULATION SCOUT</h2>
      <p>מחפש חתימות של <strong>איסוף מוסדי מוקדם</strong> לפי מתודולוגיית ריצ'רד וייקוף.
      מוסדיים בונים פוזיציות בשקט כשהציבור פוחד — הבוט מזהה את חמש החתימות הקלאסיות.</p>
      <p>
        <span class="tag tag-w">SC – Selling Climax · 25 נק'</span>
        <span class="tag tag-w">AR – Automatic Rally · 20 נק'</span>
        <span class="tag tag-w">No Supply · 20 נק'</span>
        <span class="tag tag-w">Price–Vol Divergence · 20 נק'</span>
        <span class="tag tag-w">Trading Range · 15 נק'</span>
      </p>
      <p>
        <strong>SC:</strong> ווליום ≥2× ממוצע + זנב תחתון — פאניקה שנבלמת ע"י קונים גדולים.<br>
        <strong>AR:</strong> עלייה ≥4% תוך 10 ימים — אישור ראשון להסרת היצע.<br>
        <strong>No Supply:</strong> ווליום &lt;70% מהממוצע — המוכרים נעלמו.<br>
        <strong>Divergence:</strong> מחיר יורד אבל ווליום קורס — לחץ מתייבש.<br>
        <strong>Trading Range:</strong> טווח &lt;12% ב-15 ימים — צבירה שקטה.
      </p>
      <p style="color:#607d8b;font-size:0.82rem;">⚠️ תנאי מקדים: ירידה ≥12% מהשיא.</p>
    </div>""",unsafe_allow_html=True)

    c1,c2=st.columns(2)
    with c1:
        if st.button("🎯 מניות ספציפיות",use_container_width=True,
                     type="primary" if st.session_state.w_sub=="specific" else "secondary",key="w_sub_spec"):
            st.session_state.w_sub="specific"; st.rerun()
    with c2:
        if st.button("🌐 סריקת שוק",use_container_width=True,
                     type="primary" if st.session_state.w_sub=="scan" else "secondary",key="w_sub_scan"):
            st.session_state.w_sub="scan"; st.rerun()
    st.markdown("")

    if st.session_state.w_sub=="specific":
        raw,run=_ticker_input("w")
        if run: _run_specific(raw,"wyckoff",analyze_wyckoff,_wrap_w,"w_sub")
    else:
        st.markdown("""<div style="background:#0f1a10;border:1px solid #2a4a2a;border-radius:8px;
            padding:14px 20px;direction:rtl;color:#b0d8b0;font-size:0.88rem;margin-bottom:16px;">
            הסריקה עוברת על ~200 מניות מ-S&P 500, Nasdaq ומגזרים שונים.</div>""",unsafe_allow_html=True)
        min_s,max_r=_scan_controls("wyckoff","w_sc")
        if st.button("🚀 התחל סריקת שוק — Wyckoff",use_container_width=True,key="w_scan_go"):
            hits,errors=run_market_scan(analyze_wyckoff,"wyckoff",min_s,max_r)
            render_scan_results(hits,errors,"wyckoff",min_s)

    st.markdown("""<div class="disclaimer">⚠️ אנליזה טכנית בלבד, אינה המלצת השקעה.</div>""",unsafe_allow_html=True)

# ============================================================
# VOLUME PROFILE METHOD
# ============================================================
def build_volume_profile(df, bins=40):
    mn=df["Low"].min(); mx=df["High"].max()
    edges=np.linspace(mn,mx,bins+1); vap=np.zeros(bins)
    for _,row in df.iterrows():
        lo,hi,vol=row["Low"],row["High"],row["Volume"]
        if hi==lo: continue
        for i in range(bins):
            ol=max(edges[i],lo); oh=min(edges[i+1],hi)
            if oh>ol: vap[i]+=vol*(oh-ol)/(hi-lo)
    return (edges[:-1]+edges[1:])/2,vap,edges

def analyze_vp(df):
    score=0; criteria=[]; cur=df["Close"].iloc[-1]
    mids,vap,edges=build_volume_profile(df); total=vap.sum()
    poc_idx=np.argmax(vap); poc=mids[poc_idx]; poc_pct=vap[poc_idx]/total*100
    si=np.argsort(vap)[::-1]; va_vol=0; va_idx=[]
    for i in si:
        if va_vol>=total*0.70: break
        va_vol+=vap[i]; va_idx.append(i)
    vah=mids[max(va_idx)]; val=mids[min(va_idx)]
    below=(mids<cur); lvn_thr=np.percentile(vap,20)
    has_lvn=np.sum(vap[below]<lvn_thr)>=2 if below.any() else False
    above=(mids>cur); hvn_thr=np.percentile(vap,75)
    hvn_cnt=np.sum(vap[above]>hvn_thr) if above.any() else 0; has_hvn=hvn_cnt>=1
    poc_dist=abs(cur-poc)/poc; near_poc=poc_dist<=0.03
    below_val=cur<val; bval_pct=(val-cur)/val*100 if below_val else 0
    r30=df.iloc[-30:]; rm,rv,_=build_volume_profile(r30,bins=20)
    ci=np.argmin(np.abs(rm-cur)); surge=rv[ci]>rv.mean()*1.8

    bp=25 if below_val else 0; score+=bp
    criteria.append({"name":"מחיר מתחת ל-VAL","hit":below_val,"points":25,"earned":bp,
        "explanation":(f"מחיר ({cur:.2f}) נמצא {bval_pct:.1f}% מתחת ל-VAL ({val:.2f}) — דיסקאונט לאזור הערך." if below_val else f"מחיר ({cur:.2f}) בתוך Value Area או מעליו.")})
    lp=20 if has_lvn else 0; score+=lp
    criteria.append({"name":"LVN מתחת למחיר","hit":has_lvn,"points":20,"earned":lp,
        "explanation":("זוהו אזורי ריק מתחת למחיר — מגנטים לתנועה מהירה כלפי מעלה." if has_lvn else "לא זוהו LVN משמעותיים.")})
    hp=20 if has_hvn else 0; score+=hp
    criteria.append({"name":"HVN מעל המחיר","hit":has_hvn,"points":20,"earned":hp,
        "explanation":(f"זוהו {hvn_cnt} HVN מעל — אזורים בהם מוסדיים בנו פוזיציות." if has_hvn else "לא זוהו HVN מעל המחיר.")})
    pp=20 if near_poc else 0; score+=pp
    criteria.append({"name":"קרוב ל-POC","hit":near_poc,"points":20,"earned":pp,
        "explanation":(f"מחיר במרחק {poc_dist*100:.1f}% מה-POC ({poc:.2f})." if near_poc else f"רחוק מה-POC ({poc:.2f}) ב-{poc_dist*100:.1f}% (סף 3%).")})
    sp=15 if surge else 0; score+=sp
    criteria.append({"name":"Volume Surge ברמה","hit":surge,"points":15,"earned":sp,
        "explanation":("נפח חריג ברמת המחיר ב-30 ימים — חתימת Smart Money טרייה." if surge else "לא זוהה volume surge ברמה הנוכחית.")})

    vd="סבירות גבוהה לנוכחות מוסדית" if score>=75 else "סימנים חלקיים" if score>=45 else "אין ריכוז מוסדי"
    vc="#ab47bc" if score>=75 else "#ffa726" if score>=45 else "#ef5350"
    vpd={"poc":poc,"vah":vah,"val":val,"midpoints":mids,"vol_at_price":vap,"poc_vol_pct":poc_pct}
    return score,criteria,vd,vc,vpd

def render_vp_chart(df,vpd,ticker):
    cur=df["Close"].iloc[-1]; dc=df.iloc[-65:].copy()
    mids=vpd["midpoints"]; vap=vpd["vol_at_price"]
    poc=vpd["poc"]; vah=vpd["vah"]; val=vpd["val"]; step=mids[1]-mids[0]
    bc=["#ce93d8" if abs(m-poc)<step else "#5c35a0" if val<=m<=vah else "#2a3a5a" for m in mids]
    fig=make_subplots(rows=1,cols=2,column_widths=[0.72,0.28],shared_yaxes=True,horizontal_spacing=0.01)
    fig.add_trace(go.Candlestick(x=dc.index,open=dc["Open"],high=dc["High"],low=dc["Low"],close=dc["Close"],
        increasing_line_color="#26a69a",decreasing_line_color="#ef5350",name="Price"),row=1,col=1)
    xr=[dc.index[0],dc.index[-1]]
    for lv,cl,lb in [(poc,"#ce93d8","POC"),(vah,"#4fc3f7","VAH"),(val,"#4fc3f7","VAL")]:
        fig.add_trace(go.Scatter(x=xr,y=[lv,lv],mode="lines+text",line=dict(color=cl,width=1.5,dash="dash"),
            text=["",f" {lb}:{lv:.2f}"],textposition="top right",textfont=dict(color=cl,size=10),name=lb),row=1,col=1)
    fig.add_trace(go.Scatter(xr,y=[cur,cur],mode="lines",line=dict(color="#fff",width=1,dash="dot"),name=f"Now:{cur:.2f}"),row=1,col=1)
    fig.add_trace(go.Bar(x=vap/vap.max()*100,y=mids,orientation='h',marker_color=bc,name="VP",opacity=0.9,width=step*0.85),row=1,col=2)
    fig.update_layout(height=500,paper_bgcolor="#0a1520",plot_bgcolor="#0d1b2a",font_color="#e0eaf4",
                      xaxis_rangeslider_visible=False,legend=dict(orientation="h",y=1.04,x=0,font=dict(size=10)),
                      margin=dict(t=20,b=20,l=10,r=10))
    fig.update_xaxes(gridcolor="#1e3040"); fig.update_yaxes(gridcolor="#1e3040")
    fig.update_xaxes(title_text="Vol %",row=1,col=2)
    return fig

def screen_vp():
    st.markdown("""
    <div class="header-box vp">
      <h2>🔮 VOLUME PROFILE SCOUT</h2>
      <p>מנתח <strong>פרופיל הנפח</strong> — פיזור הנפח לפי רמות מחיר לאורך שנה.
      VP מסתכל על <em>מחיר</em> ולא על זמן: איפה "הכסף החכם" באמת ישב.</p>
      <p>
        <span class="tag tag-v">POC – Point of Control</span>
        <span class="tag tag-v">Value Area (VAH / VAL)</span>
        <span class="tag tag-v">HVN – High Volume Node</span>
        <span class="tag tag-v">LVN – Low Volume Node</span>
      </p>
      <p>
        <strong>POC:</strong> רמת המחיר עם הנפח הגבוה ביותר — אזור האיזון המוסדי.<br>
        <strong>Value Area:</strong> 70% מסך הנפח — טווח הפעילות של "הכסף החכם".<br>
        <strong>VAH/VAL:</strong> מחיר מתחת ל-VAL = דיסקאונט מובהק ביחס לאזור הערך.<br>
        <strong>HVN:</strong> ריכוז פעילות גבוה — תמיכה/התנגדות חזקה.<br>
        <strong>LVN:</strong> אזור ריק — המחיר נוטה לנוע דרכו מהר.
      </p>
      <p>
        <span class="tag tag-v">מחיר מתחת ל-VAL · 25 נק'</span>
        <span class="tag tag-v">LVN מתחת למחיר · 20 נק'</span>
        <span class="tag tag-v">HVN מעל המחיר · 20 נק'</span>
        <span class="tag tag-v">קרוב ל-POC · 20 נק'</span>
        <span class="tag tag-v">Volume Surge · 15 נק'</span>
      </p>
      <p style="color:#607d8b;font-size:0.82rem;">✦ הגרף: פרופיל אופקי — סגול כהה=Value Area, סגול בהיר=POC.</p>
    </div>""",unsafe_allow_html=True)

    c1,c2=st.columns(2)
    with c1:
        if st.button("🎯 מניות ספציפיות",use_container_width=True,
                     type="primary" if st.session_state.vp_sub=="specific" else "secondary",key="vp_sub_spec"):
            st.session_state.vp_sub="specific"; st.rerun()
    with c2:
        if st.button("🌐 סריקת שוק",use_container_width=True,
                     type="primary" if st.session_state.vp_sub=="scan" else "secondary",key="vp_sub_scan"):
            st.session_state.vp_sub="scan"; st.rerun()
    st.markdown("")

    if st.session_state.vp_sub=="specific":
        raw,run=_ticker_input("vp")
        if run: _run_specific(raw,"vp",analyze_vp,_wrap_vp,"vp_sub")
    else:
        st.markdown("""<div style="background:#160d20;border:1px solid #3a1a4a;border-radius:8px;
            padding:14px 20px;direction:rtl;color:#c8b0d8;font-size:0.88rem;margin-bottom:16px;">
            בניית פרופיל נפח לוקחת יותר זמן — הסריקה עשויה לארוך 5–10 דקות.</div>""",unsafe_allow_html=True)
        min_s,max_r=_scan_controls("vp","vp_sc")
        if st.button("🚀 התחל סריקת שוק — Volume Profile",use_container_width=True,key="vp_scan_go"):
            hits,errors=run_market_scan(analyze_vp,"vp",min_s,max_r)
            render_scan_results(hits,errors,"vp",min_s)

    st.markdown("""<div class="disclaimer">⚠️ אנליזה טכנית בלבד, אינה המלצת השקעה.</div>""",unsafe_allow_html=True)

# ============================================================
# VWAP DEVIATION METHOD
# ============================================================
def compute_vwap(df):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_tpv = (tp * df["Volume"]).cumsum()
    cum_vol = df["Volume"].cumsum()
    vwap = cum_tpv / cum_vol
    dev = df["Close"] - vwap
    rolling_std = dev.rolling(20).std()
    return vwap, dev, rolling_std

def analyze_vwap(df):
    score=0; criteria=[]
    vwap, dev, rolling_std = compute_vwap(df)
    cur = df["Close"].iloc[-1]; cur_vwap = vwap.iloc[-1]; cur_dev = dev.iloc[-1]; cur_std = rolling_std.iloc[-1]

    below_1std = cur_dev <= -1.0 * cur_std
    below_2std = cur_dev <= -2.0 * cur_std
    pts_below = 25 if below_2std else 15 if below_1std else 0
    score += pts_below; std_ratio = abs(cur_dev) / cur_std if cur_std > 0 else 0
    criteria.append({"name":"סטיית מחיר מה-VWAP","hit":pts_below>0,"points":25,"earned":pts_below,
        "explanation":(f"מחיר ({cur:.2f}) נמצא {std_ratio:.1f}σ מתחת ל-VWAP ({cur_vwap:.2f}). "+("דיסקאונט של 2σ+ — סטייה קיצונית, הסתברות גבוהה לחזרה." if below_2std else "דיסקאונט של 1σ — מחיר נמוך ביחס לממוצע המשוקלל." if below_1std else "מחיר קרוב ל-VWAP או מעליו — אין דיסקאונט."))})

    vwap_20 = vwap.iloc[-20:]
    slope_early = (vwap_20.iloc[10] - vwap_20.iloc[0]) / vwap_20.iloc[0]; slope_late = (vwap_20.iloc[-1] - vwap_20.iloc[-10]) / vwap_20.iloc[-10]
    flattening = (slope_early < -0.005) and (slope_late > slope_early * 0.5)
    flat_pts = 20 if flattening else 0; score += flat_pts
    criteria.append({"name":"VWAP Slope מתיישב","hit":flattening,"points":20,"earned":flat_pts,"explanation":(f"שיפוע מוקדם: {slope_early*100:+.2f}% | שיפוע מאוחר: {slope_late*100:+.2f}%. "+("ה-VWAP מתיישב — המומנטום השלילי נחלש, מוסדיים מתחילים לספוג." if flattening else "ה-VWAP עדיין במגמה שלילית ברורה — המומנטום לא השתנה."))})

    last_10 = df.iloc[-10:]; price_moving_up = last_10["Close"].iloc[-1] > last_10["Close"].iloc[0]; vol_expanding = last_10["Volume"].iloc[-5:].mean() > last_10["Volume"].iloc[:5].mean() * 1.2
    approach_pts = 20 if (price_moving_up and vol_expanding) else 0; score += approach_pts; v_ratio = last_10["Volume"].iloc[-5:].mean() / last_10["Volume"].iloc[:5].mean()
    criteria.append({"name":"נפח עולה עם התקרבות ל-VWAP","hit":approach_pts>0,"points":20,"earned":approach_pts,"explanation":(f"מחיר {'עולה' if price_moving_up else 'יורד'} | ווליום: {v_ratio:.2f}x הממוצע הקודם. "+("מחיר מתקרב ל-VWAP עם נפח גובר — מוסדיים נכנסים." if approach_pts>0 else "לא נראית ספיגה פעילה עם נפח ביחס ל-VWAP."))})

    low_idx = df["Low"].iloc[-65:].idxmin(); post_low = df.loc[low_idx:]
    if len(post_low) >= 5:
        tp_post = (post_low["High"]+post_low["Low"]+post_low["Close"])/3; avwap = (tp_post * post_low["Volume"]).cumsum() / post_low["Volume"].cumsum()
        above_avwap = cur > avwap.iloc[-1]; avwap_val = avwap.iloc[-1]
    else:
        above_avwap = False; avwap_val = cur
    avwap_pts = 20 if above_avwap else 0; score += avwap_pts; low_date = low_idx.strftime('%d/%m/%Y') if hasattr(low_idx,'strftime') else str(low_idx)[:10]
    criteria.append({"name":"מעל Anchored VWAP מהשפל","hit":above_avwap,"points":20,"earned":avwap_pts,"explanation":(f"Anchored VWAP מהשפל ({low_date}): {avwap_val:.2f}. "+(f"מחיר ({cur:.2f}) נסחר מעל — מבנה עולה מהשפל, סימן בריאות מבנית." if above_avwap else f"מחיר ({cur:.2f}) עדיין מתחת ל-AVWAP — לא בוצעה החלמה מהשפל."))})

    std_early = rolling_std.iloc[-20:-10].mean(); std_late = rolling_std.iloc[-10:].mean(); contracting = (std_late < std_early * 0.85) if (std_early > 0) else False
    cont_pts = 15 if contracting else 0; score += cont_pts
    criteria.append({"name":"סטיית תקן מתכווצת","hit":contracting,"points":15,"earned":cont_pts,"explanation":(f"σ מוקדם: {std_early:.3f} | σ מאוחר: {std_late:.3f}. "+("הסטיה מתכווצת — תנודתיות יורדת, המחיר מתייצב סביב VWAP." if contracting else "הסטיה לא מתכווצת — המחיר עדיין תנודתי."))})

    vd = "מיקום VWAP אופטימלי לכניסה" if score>=75 else "סטייה חלקית" if score>=45 else "אין הזדמנות VWAP"
    vc = "#4caf7d" if score>=75 else "#ffa726" if score>=45 else "#ef5350"
    vwap_data = {"vwap":vwap,"dev":dev,"rolling_std":rolling_std,"cur_vwap":cur_vwap,"cur_std":cur_std}
    return score,criteria,vd,vc,vwap_data

def render_vwap_chart(df, vwap_data, ticker):
    dc = df.iloc[-65:].copy(); vwap = vwap_data["vwap"].reindex(dc.index); std = vwap_data["rolling_std"].reindex(dc.index)
    fig = make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.65,0.35],vertical_spacing=0.04)
    fig.add_trace(go.Candlestick(x=dc.index,open=dc["Open"],high=dc["High"],low=dc["Low"],close=dc["Close"],increasing_line_color="#26a69a",decreasing_line_color="#ef5350",name="Price"),row=1,col=1)
    fig.add_trace(go.Scatter(x=dc.index,y=vwap,line=dict(color="#ffa726",width=2),name="VWAP"),row=1,col=1)
    fig.add_trace(go.Scatter(x=dc.index,y=vwap+std,line=dict(color="#4caf7d",width=1,dash="dot"),name="+1σ"),row=1,col=1)
    fig.add_trace(go.Scatter(x=dc.index,y=vwap-std,line=dict(color="#4caf7d",width=1,dash="dot"),fill='tonexty',fillcolor='rgba(76,175,61,0.05)',name="-1σ"),row=1,col=1)
    fig.add_trace(go.Scatter(x=dc.index,y=vwap+2*std,line=dict(color="#ef5350",width=1,dash="dash"),name="+2σ"),row=1,col=1)
    fig.add_trace(go.Scatter(x=dc.index,y=vwap-2*std,line=dict(color="#ef5350",width=1,dash="dash"),name="-2σ"),row=1,col=1)
    dev = (dc["Close"] - vwap) / std.replace(0, np.nan); dev_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in dev.fillna(0)]
    fig.add_trace(go.Bar(x=dc.index,y=dev,marker_color=dev_colors,name="Dev (σ)",opacity=0.8),row=2,col=1)
    fig.add_hline(y=0,line_color="#ffa726",line_width=1,row=2,col=1); fig.add_hline(y=-1,line_color="#4caf7d",line_dash="dot",line_width=1,row=2,col=1); fig.add_hline(y=-2,line_color="#ef5350",line_dash="dot",line_width=1,row=2,col=1)
    fig.update_layout(height=480,paper_bgcolor="#0a1520",plot_bgcolor="#0d1b2a",font_color="#e0eaf4",xaxis_rangeslider_visible=False,legend=dict(orientation="h",y=1.02,x=0,font=dict(size=9)),margin=dict(t=10,b=10))
    fig.update_xaxes(gridcolor="#1e3040"); fig.update_yaxes(gridcolor="#1e3040")
    return fig

def screen_vwap():
    st.markdown("""<div class="header-box vwap"><h2>📐 VWAP DEVIATION SCOUT</h2><p>מנתח סטיית תקן מהשנתי.</p></div>""",unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        if st.button("🎯 מניות ספציפיות",key="vw_spec"): st.session_state.vw_sub="specific"; st.rerun()
    with c2:
        if st.button("🌐 סריקת שוק",key="vw_scan"): st.session_state.vw_sub="scan"; st.rerun()
    if st.session_state.vw_sub=="specific":
        raw,run=_ticker_input("vw")
        if run: _run_specific(raw,"vwap",analyze_vwap,_wrap_vw,"vw_sub")
    else:
        min_s,max_r=_scan_controls("vwap","vw_sc")
        if st.button("🚀 התחל סריקת שוק — VWAP Deviation"):
            hits,errors=run_market_scan(analyze_vwap,"vwap",min_s,max_r); render_scan_results(hits,errors,"vwap",min_s)

# ============================================================
# COMPOSITE SCORE
# ============================================================
WEIGHTS = {"wyckoff": 0.35, "vp": 0.35, "vwap": 0.30}

def analyze_composite(df):
    w_score, w_crit, w_vd, w_vc, w_prereq, w_dd = analyze_wyckoff(df)
    v_score, v_crit, v_vd, v_vc, vpd = analyze_vp(df)
    vw_score,vw_crit,vw_vd,vw_vc,vwap_data = analyze_vwap(df)
    composite = int(round(w_score * WEIGHTS["wyckoff"] + v_score * WEIGHTS["vp"] + vw_score * WEIGHTS["vwap"]))
    methods_above_60 = sum(1 for s in [w_score,v_score,vw_score] if s >= 60); methods_above_75 = sum(1 for s in [w_score,v_score,vw_score] if s >= 75)
    if composite >= 75 and methods_above_75 >= 2:
        verdict = "Strong Signal"; vcolor = "#26a69a"; signal_class = "signal-strong"; action = "⚡ ALERT — שווה בדיקה"
    elif composite >= 60 and methods_above_60 >= 2:
        verdict = "Watch"; vcolor = "#ffa726"; signal_class = "signal-medium"; action = "👁 WATCH — עקוב"
    else:
        verdict = "אין קונצנזוס"; vcolor = "#ef5350"; signal_class = "signal-weak"; action = "⏳ WAIT"
    breakdown = [{"method":"Wyckoff Accumulation","score":w_score,"weight":35,"color":w_vc,"verdict":w_vd,"key":"wyckoff"},{"method":"Volume Profile","score":v_score,"weight":35,"color":v_vc,"verdict":v_vd,"key":"vp"},{"method":"VWAP Deviation","score":vw_score,"weight":30,"color":vw_vc,"verdict":vw_vd,"key":"vwap"}]
    return composite,vcolor,verdict,signal_class,action,breakdown,{"w":(w_score,w_crit,w_vd,w_vc,w_prereq,w_dd),"v":(v_score,v_crit,v_vd,v_vc,vpd),"vw":(vw_score,vw_crit,vw_vd,vw_vc,vwap_data)}

def screen_composite():
    st.markdown("""<div class="header-box composite"><h2>🏆 COMPOSITE SCORE</h2></div>""",unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        if st.button("🎯 מניות ספציפיות",key="comp_spec"): st.session_state.comp_sub="specific"; st.rerun()
    with c2:
        if st.button("🌐 סריקת שוק",key="comp_scan"): st.session_state.comp_sub="scan"; st.rerun()
    if st.session_state.comp_sub=="specific":
        raw,run=_ticker_input("comp")
        if run: _run_specific(raw,"composite",analyze_composite,_wrap_comp,"comp_sub")
    else:
        min_s,max_r=_scan_controls("composite","comp_sc")
        if st.button("🚀 התחל סריקת שוק — Composite"):
            hits,errors=run_market_scan(analyze_composite,"composite",min_s,max_r); render_scan_results(hits,errors,"composite",min_s)

# ============================================================
# MARKET SCAN ENGINE
# ============================================================
def run_market_scan(analyze_fn, mode, score_threshold=80, max_results=10):
    hits=[]; errors=[]; total=len(SCAN_UNIVERSE); accent = {"wyckoff":"#4fc3f7","vp":"#ce93d8","vwap":"#4caf7d","composite":"#ffa726"}.get(mode,"#fff")
    st.markdown(f"""<div style="background:#0a1520;border:1px solid #1e3040;border-radius:8px;padding:12px 18px;direction:rtl;color:#b0c8e0;font-size:0.85rem;margin-bottom:16px;">🔍 סורק <b style="color:#fff">{total}</b> מניות | ציון ≥ <b style="color:#ffa726">{score_threshold}</b></div>""", unsafe_allow_html=True)
    prog=st.progress(0); status=st.empty(); hits_ph=st.empty()
    for i,ticker in enumerate(SCAN_UNIVERSE):
        if len(hits)>=max_results: break
        prog.progress((i+1)/total); df=get_data(ticker)
        if df is None: errors.append(ticker); continue
        try:
            res=analyze_fn(df); score=res[0]
            if score>=score_threshold: hits.append({"ticker":ticker,"score":score,"verdict":res[2],"verdict_color":res[3] if mode!="composite" else res[1]})
        except Exception: errors.append(ticker); continue
    prog.empty(); return sorted(hits,key=lambda x:x["score"],reverse=True),errors

def render_scan_results(hits,errors,mode,score_threshold=80):
    accent={"wyckoff":"#4fc3f7","vp":"#ce93d8","vwap":"#4caf7d","composite":"#ffa726"}.get(mode,"#fff"); card_class={"wyckoff":"","vp":"vp-card","vwap":"vw-card","composite":"comp-card"}.get(mode,"")
    if not hits: st.warning("לא נמצאו תוצאות"); return
    top8=hits[:8]; cols=st.columns(min(len(top8),4))
    for idx,h in enumerate(top8):
        with cols[idx%4]: st.markdown(f"""<div class="overview-card {card_class}"><div class="ticker-label" style="color:{accent}">{h['ticker']}</div><div class="score-big" style="color:{h['verdict_color']}">{h['score']}</div><div class="verdict-label">{h['verdict']}</div></div>""",unsafe_allow_html=True)
    pick=st.selectbox("בחר מניה לניתוח",[h["ticker"] for h in hits])
    if st.button("▶ פתח ניתוח"):
        df=get_data(pick)
        if df: _dispatch_detail(pick,df,mode)

def _dispatch_detail(t,df,mode):
    if mode=="wyckoff": res=analyze_wyckoff(df); _render_w_detail(t,df,*res)
    elif mode=="vp": res=analyze_vp(df); _render_vp_detail(t,df,res[0],res[1],res[2],res[3],res[4])
    elif mode=="vwap": res=analyze_vwap(df); _render_vw_detail(t,df,res[0],res[1],res[2],res[3],res[4])
    else: res=analyze_composite(df); _render_composite_detail(t,df,res[0],res[1],res[2],res[3],res[4],res[5],res[6])

def _render_criteria(criteria,box_pos,box_neg):
    for c in criteria:
        box=box_pos if c["hit"] else box_neg; lbl="✅ הצליח" if c["hit"] else "❌ נכשל"; cls="hit" if c["hit"] else "miss"
        st.markdown(f"""<div class="score-reason-box {box}"><div class="criteria-row"><strong>{c['name']}</strong><span><span class="{cls}">{lbl}</span> | <strong>{c['earned']}/{c['points']}</strong></span></div></div>""",unsafe_allow_html=True)

def _render_w_detail(t,df,score,criteria,verdict,vcolor,prereq,dd):
    st.plotly_chart(render_gauge(score,verdict,vcolor,"wyckoff"),use_container_width=True); _render_criteria(criteria,"positive","negative"); st.plotly_chart(render_wyckoff_chart(df),use_container_width=True)
def _render_vp_detail(t,df,score,criteria,verdict,vcolor,vpd):
    st.plotly_chart(render_gauge(score,verdict,vcolor,"vp"),use_container_width=True); _render_criteria(criteria,"vp-positive","vp-negative"); st.plotly_chart(render_vp_chart(df,vpd,t),use_container_width=True)
def _render_vw_detail(t,df,score,criteria,verdict,vcolor,vwap_data):
    st.plotly_chart(render_gauge(score,verdict,vcolor,"vwap"),use_container_width=True); _render_criteria(criteria,"vw-positive","vw-negative"); st.plotly_chart(render_vwap_chart(df,vwap_data,t),use_container_width=True)
def _render_composite_detail(t,df,composite,vcolor,verdict,signal_class,action,breakdown,detail):
    st.plotly_chart(render_gauge(composite,verdict,vcolor,"composite"),use_container_width=True)

def _scan_controls(mode, scan_key_prefix):
    fc1,fc2=st.columns(2)
    with fc1: min_s=st.slider("ציון מינימום",60,95,80,5,key=f"{scan_key_prefix}_min")
    with fc2: max_r=st.slider("מקסימום תוצאות",3,30,10,1,key=f"{scan_key_prefix}_max")
    return min_s,max_r

def _ticker_input(key_prefix):
    ci,cb=st.columns([4,1])
    with ci: raw=st.text_input("טיקרים","NVDA, MSFT",key=f"{key_prefix}_input")
    with cb: st.markdown("<div style='margin-top:28px'></div>",unsafe_allow_html=True); run=st.button("▶ הרץ",key=f"{key_prefix}_run")
    return raw,run

def _run_specific(raw, mode, analyze_fn, render_fn, sub_state_key):
    tickers=[t.strip().upper() for t in raw.replace(","," ").split() if t.strip()]
    for t in tickers:
        df=get_data(t)
        if df is not None: render_fn(t,df,analyze_fn(df))

def _wrap_w(t,df,res):    _render_w_detail(t,df,*res)
def _wrap_vp(t,df,res):   _render_vp_detail(t,df,res[0],res[1],res[2],res[3],res[4])
def _wrap_vw(t,df,res):   _render_vw_detail(t,df,res[0],res[1],res[2],res[3],res[4])
def _wrap_comp(t,df,res): _render_composite_detail(t,df,res[0],res[1],res[2],res[3],res[4],res[5],res[6])

# ============================================================
# NEW BACKTEST SCREEN
# ============================================================
def screen_backtest():
    st.markdown("""
    <div class="header-box composite" style="background:linear-gradient(135deg,#121a24,#1a2636);border:1px solid #2a4a6a;">
      <h2>📈 BACKTEST ENGINE</h2>
      <p>הרצת סימולציות ואסטרטגיות על נתוני עבר.</p>
    </div>""",unsafe_allow_html=True)
    
    st.markdown("### הגדרות הבק-טסטינג")
    try:
        # כאן המערכת מריצה ומציגה את הקוד של backtest_engine.py
        # אם יש לך פונקציה מסוימת שמפעילה את ה-UI שם (כמו run_ui או main), תסיר את ה-# מהשורה הבאה:
        # backtest_engine.run_ui()
        st.info("💡 מודול backtest_engine זמין ומחובר! ברגע שתגדיר בתוכו פונקציה לתצוגה, תוכל להריץ אותה ישירות מכאן.")
    except Exception as e:
        st.error(f"⚠️ שגיאה בטעינת הבק-טסט: {e}")

# ============================================================
# ROUTER DISPATCH
# ============================================================
routes = {
    "wyckoff": screen_wyckoff,
    "vp": screen_vp,
    "vwap": screen_vwap,
    "composite": screen_composite,
    "backtest": screen_backtest
}
routes[st.session_state.mode]()
