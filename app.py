import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

st.set_page_config(layout="wide", page_title="Institutional Scout")

# ============================================================
# UNIVERSE FOR MARKET SCAN
# ============================================================
SCAN_UNIVERSE = [
    # S&P 500 large caps + liquid mid caps across sectors
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","JPM","JNJ",
    "V","UNH","XOM","PG","MA","HD","CVX","MRK","ABBV","PEP",
    "KO","AVGO","COST","WMT","LLY","TMO","MCD","ACN","BAC","CRM",
    "NFLX","AMD","ADBE","CSCO","ABT","TXN","NEE","DHR","RTX","QCOM",
    "HON","NKE","INTC","AMGN","PM","IBM","SBUX","INTU","GS","CAT",
    "BA","GE","SPGI","AXP","BLK","DE","ISRG","MDLZ","ADI","GILD",
    "REGN","SYK","ZTS","EL","MMC","AON","TJX","SCHW","CB","USB",
    "WFC","C","MS","CVS","CI","HUM","AIG","MET","SLB","EOG",
    "OXY","COP","PSX","VLO","MPC","KMI","WMB","ET","D","SO",
    "DUK","AEP","EXC","PCG","SRE","ED","PEG","PPL","FE","ETR",
    "AMT","PLD","CCI","EQIX","SPG","O","WELL","DLR","EQR","AVB",
    "FCX","NEM","GOLD","AEM","WPM","FNV","PAAS","AG","HL","SILV",
    "PANW","CRWD","FTNT","ZS","OKTA","DDOG","SNOW","MDB","NET","CFLT",
    "UBER","LYFT","ABNB","DASH","COIN","HOOD","SOFI","UPST","AFRM","LC",
    "F","GM","STLA","TM","HMC","RIVN","LCID","FSR","NIO","LI",
    "ONTO","KLAC","LRCX","AMAT","ASML","MPWR","MRVL","SWKS","QRVO","ENTG",
    "DELL","HPQ","HPE","WDC","STX","NTAP","PSTG","NTNX","SMCI","PLTR",
    "RBLX","U","TTWO","EA","ATVI","NTES","BILI","SE","GRAB","BEKE",
    "DIS","CMCSA","PARA","WBD","FOXA","NWSA","NYT","OMC","IPG","MGNI",
    "CVS","WBA","RAD","ESRX","MCK","CAH","ABC","OMI","PDCO","HSIC",
    "MO","BTI","PM","VGR","STZ","BUD","TAP","SAM","COTY","EL",
    "DKNG","MGM","CZR","WYNN","LVS","PENN","RSI","BYD","RCL","CCL",
    "NCLH","MAR","HLT","H","IHG","STAY","SHO","PK","XHR","RHP",
    "DAL","UAL","AAL","LUV","ALK","HA","JBLU","SAVE","MESA","SKYW",
    "FDX","UPS","XPO","ODFL","SAIA","JBHT","CHRW","EXPD","ECHO","LSTR",
    "JPM","BAC","WFC","C","GS","MS","USB","PNC","TFC","COF",
    "AMP","RJF","LPLA","SF","PIPR","EVR","LAZ","HLI","MC","PWP",
    "SPY","QQQ","IWM","DIA","GLD","SLV","GDX","GDXJ","XLE","XLF",
]
# Deduplicate
SCAN_UNIVERSE = list(dict.fromkeys(SCAN_UNIVERSE))

# ============================================================
# GLOBAL CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+Hebrew:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans Hebrew', sans-serif; direction: rtl; }
    h1, h2, h3, h4 { font-family: 'IBM Plex Mono', monospace; direction: ltr; }

    .header-box { border-radius: 12px; padding: 24px 32px; margin-bottom: 28px;
                  color: #e0eaf4; direction: rtl; line-height: 1.9; }
    .header-box.wyckoff { background: linear-gradient(135deg,#0f1923,#1a2a3a); border:1px solid #2a4a6a; }
    .header-box.vp      { background: linear-gradient(135deg,#160f23,#251535); border:1px solid #4a2a6a; }
    .header-box h2 { font-family:'IBM Plex Mono',monospace; font-size:1.05rem; margin-bottom:12px; direction:ltr; }
    .header-box.wyckoff h2 { color:#4fc3f7; }
    .header-box.vp      h2 { color:#ce93d8; }
    .header-box p { color:#b0c8e0; font-size:0.92rem; margin:6px 0; }

    .tag { display:inline-block; font-family:'IBM Plex Mono',monospace; font-size:0.75rem;
           padding:2px 8px; border-radius:4px; margin:3px 2px; }
    .tag-w { background:#1e3a5f; border:1px solid #4fc3f7; color:#4fc3f7; }
    .tag-v { background:#2a1a4a; border:1px solid #ab47bc; color:#ce93d8; }

    .score-reason-box { background:#0d1b2a; border-left:4px solid #4fc3f7; border-radius:8px;
                        padding:18px 22px; margin:10px 0; direction:rtl; color:#cde3f5;
                        font-size:0.88rem; line-height:1.8; }
    .score-reason-box.positive    { border-left-color:#26a69a; }
    .score-reason-box.negative    { border-left-color:#ef5350; }
    .score-reason-box.vp-positive { background:#150d20; border-left-color:#ab47bc; }
    .score-reason-box.vp-negative { background:#150d20; border-left-color:#ef5350; }
    .score-reason-box strong { color:#fff; }

    .criteria-row { display:flex; justify-content:space-between; align-items:center;
                    padding:6px 0; border-bottom:1px solid #1e3040; font-size:0.84rem; }
    .hit  { color:#26a69a; font-weight:600; }
    .miss { color:#ef5350; }

    .overview-card { background:#0d1b2a; border:1px solid #2a4a6a; border-radius:10px;
                     padding:18px 20px; text-align:center; direction:ltr; }
    .overview-card.vp-card { border-color:#4a2a6a; background:#120d1e; }
    .ticker-label { font-family:'IBM Plex Mono',monospace; font-size:1.1rem;
                    font-weight:600; margin-bottom:4px; }
    .score-big { font-family:'IBM Plex Mono',monospace; font-size:2.2rem; font-weight:600; margin:6px 0; }
    .verdict-label { font-size:0.78rem; color:#b0c8e0; margin-top:4px; }
    .bar-bg { background:#1e3040; border-radius:4px; height:8px; margin-top:10px; overflow:hidden; }
    .bar-fill { height:8px; border-radius:4px; }

    .scan-result-row { background:#0a1520; border:1px solid #1e3040; border-radius:8px;
                       padding:12px 18px; margin:6px 0; direction:ltr;
                       font-family:'IBM Plex Mono',monospace; font-size:0.88rem; }
    .scan-result-row:hover { border-color:#4fc3f7; }

    .scan-badge { display:inline-block; padding:3px 10px; border-radius:20px;
                  font-size:0.75rem; font-weight:600; font-family:'IBM Plex Mono',monospace; }
    .badge-green  { background:#0d2a20; color:#26a69a; border:1px solid #26a69a; }
    .badge-yellow { background:#2a1e08; color:#ffa726; border:1px solid #ffa726; }
    .badge-red    { background:#2a0d0d; color:#ef5350; border:1px solid #ef5350; }
    .badge-purple { background:#1e0d2a; color:#ce93d8; border:1px solid #ce93d8; }

    .submode-bar { display:flex; gap:10px; margin-bottom:22px; direction:ltr; }
    .submode-pill { padding:8px 22px; border-radius:20px; font-family:'IBM Plex Mono',monospace;
                    font-size:0.82rem; cursor:pointer; border:1px solid #2a4a6a;
                    background:#0d1b2a; color:#607d8b; }
    .submode-pill.active-w { border-color:#4fc3f7; background:#0f2030; color:#4fc3f7; }
    .submode-pill.active-v { border-color:#ab47bc; background:#1a0f2a; color:#ce93d8; }

    .disclaimer { background:#1a1206; border:1px solid #5a4010; border-radius:8px;
                  padding:10px 16px; color:#a08040; font-size:0.78rem; direction:rtl; margin-top:18px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
for key, default in [("mode","wyckoff"), ("w_submode","specific"), ("vp_submode","specific")]:
    if key not in st.session_state:
        st.session_state[key] = default

# ============================================================
# TOP-LEVEL MODE SWITCHER
# ============================================================
st.markdown("# INSTITUTIONAL SCOUT")
col_w, col_v = st.columns(2)
with col_w:
    if st.button("⬛  Wyckoff Accumulation", use_container_width=True,
                 type="primary" if st.session_state.mode=="wyckoff" else "secondary"):
        st.session_state.mode = "wyckoff"; st.rerun()
with col_v:
    if st.button("🔮  Volume Profile", use_container_width=True,
                 type="primary" if st.session_state.mode=="vp" else "secondary"):
        st.session_state.mode = "vp"; st.rerun()
st.markdown("---")

# ============================================================
# DATA
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
def render_gauge(score, verdict, verdict_color, mode="wyckoff"):
    if mode == "wyckoff":
        steps = [{'range':[0,44],'color':'#1a0d0d'},{'range':[44,74],'color':'#1a1206'},{'range':[74,100],'color':'#0d1a18'}]
        bar_color = "#26a69a" if score>=75 else "#ffa726" if score>=45 else "#ef5350"
    else:
        steps = [{'range':[0,44],'color':'#1a0d18'},{'range':[44,74],'color':'#1a0f2a'},{'range':[74,100],'color':'#1a0d25'}]
        bar_color = "#ab47bc" if score>=75 else "#ffa726" if score>=45 else "#ef5350"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        title={'text':f"<b>Institutional Score</b><br><span style='font-size:0.82em;color:{verdict_color}'>{verdict}</span>",'font':{'size':13}},
        gauge={'axis':{'range':[0,100],'tickwidth':1,'tickcolor':"#4a6a8a"},
               'bar':{'color':bar_color,'thickness':0.3},
               'bgcolor':"#0d1b2a",'borderwidth':1,'bordercolor':"#2a4a6a",
               'steps':steps,
               'threshold':{'line':{'color':"#ffffff",'width':2},'thickness':0.75,'value':score}},
        number={'font':{'size':48,'color':bar_color},'suffix':'/100'}
    ))
    fig.update_layout(height=300, margin=dict(t=80,b=10,l=20,r=20),
                      paper_bgcolor="#0a1520", font_color="#e0eaf4")
    return fig

def render_comparison_chart(valid):
    sorted_t = sorted(valid.keys(), key=lambda t: valid[t]["score"], reverse=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sorted_t, y=[valid[t]["score"] for t in sorted_t],
        marker_color=[valid[t]["verdict_color"] for t in sorted_t],
        text=[str(valid[t]["score"]) for t in sorted_t],
        textposition="outside",
        textfont=dict(color="#e0eaf4", family="IBM Plex Mono", size=14)
    ))
    fig.update_layout(height=280, paper_bgcolor="#0a1520", plot_bgcolor="#0d1b2a",
                      font_color="#e0eaf4", font_family="IBM Plex Mono",
                      yaxis=dict(range=[0,115], gridcolor="#1e3040", title="Score"),
                      xaxis=dict(gridcolor="#1e3040"),
                      margin=dict(t=20,b=20,l=20,r=20), showlegend=False)
    return fig, sorted_t

# ============================================================
# WYCKOFF LOGIC
# ============================================================
def analyze_wyckoff(df):
    score = 0; criteria = []
    high_3m  = df["Close"].iloc[-65:].max()
    current  = df["Close"].iloc[-1]
    drawdown = (high_3m - current) / high_3m
    prereq   = drawdown >= 0.12

    # 1. SC
    sc_win = df.iloc[-30:]
    sc_cands = sc_win[(sc_win["Volume"]>=sc_win["VOL_MEAN"]*2.0) & (sc_win["LOWER_SHADOW"]>sc_win["BODY"]*1.2)]
    sc_found = len(sc_cands) > 0
    sc_pts   = 25 if (sc_found and prereq) else 0
    score   += sc_pts
    sc_idx   = sc_cands.index[-1] if sc_found else None
    criteria.append({"name":"Selling Climax (SC)","hit":sc_found and prereq,"points":25,"earned":sc_pts,
        "explanation":(
            f"זוהה SC ב-{sc_idx.strftime('%d/%m/%Y') if sc_idx else '—'}: ווליום פי {sc_cands['Volume'].iloc[-1]/sc_cands['VOL_MEAN'].iloc[-1]:.1f} מהממוצע עם זנב תחתון ארוך."
            if sc_found and prereq else
            "לא זוהה SC. " + (f"ירידה מהשיא {drawdown*100:.1f}% — תנאי מקדים לא מתקיים (נדרש ≥12%)." if not prereq else
                               "לא נמצא נר עם ווליום חריג וזנב תחתון ב-30 הימים האחרונים."))})

    # 2. AR
    ar_found=False; ar_pts=0; ar_exp="לא זוהה AR — נדרש SC קודם."
    if sc_found and sc_idx is not None:
        post = df.loc[sc_idx:].iloc[1:11]
        if len(post)>=2:
            rally = (post["Close"].max()-df.loc[sc_idx,"Close"])/df.loc[sc_idx,"Close"]
            ar_found = rally>=0.04; ar_pts=20 if ar_found else 0; score+=ar_pts
            ar_exp = (f"זוהה AR: עלייה {rally*100:.1f}% תוך 10 ימים לאחר ה-SC." if ar_found else
                      f"לא זוהה AR: עלייה מקסימלית {rally*100:.1f}% בלבד (נדרש ≥4%).")
    criteria.append({"name":"Automatic Rally (AR)","hit":ar_found,"points":20,"earned":ar_pts,"explanation":ar_exp})

    # 3. No Supply
    avg10 = df.iloc[-10:]["Volume"].mean(); gmean = df["VOL_MEAN"].iloc[-1]
    ns = avg10 < gmean*0.7; ns_pts=20 if ns else 0; score+=ns_pts
    criteria.append({"name":"No Supply","hit":ns,"points":20,"earned":ns_pts,
        "explanation":f"ווליום ממוצע 10 ימים: {avg10/gmean*100:.0f}% מהממוצע — " +
                      ("המוכרים נעלמו." if ns else "ווליום עדיין גבוה מדי.")})

    # 4. Divergence
    l20=df.iloc[-20:]; pc=(l20["Close"].iloc[-1]-l20["Close"].iloc[0])/l20["Close"].iloc[0]
    vc=(l20["Volume"].iloc[-5:].mean()-l20["Volume"].iloc[:5].mean())/l20["Volume"].iloc[:5].mean()
    div=(pc<0)and(vc<-0.25); div_pts=20 if div else 0; score+=div_pts
    criteria.append({"name":"Price–Vol Divergence","hit":div,"points":20,"earned":div_pts,
        "explanation":f"מחיר: {pc*100:+.1f}% | ווליום: {vc*100:+.1f}% (20 ימים). " +
                      ("לחץ מכירה קורס — Smart Money." if div else "לא נמצאה דיברגנציה.")})

    # 5. Trading Range
    l15=df.iloc[-15:]; tr_pct=(l15["High"].max()-l15["Low"].min())/l15["Low"].min()
    inr=tr_pct<0.12; tr_pts=15 if inr else 0; score+=tr_pts
    criteria.append({"name":"Trading Range","hit":inr,"points":15,"earned":tr_pts,
        "explanation":f"טווח 15 ימים: {tr_pct*100:.1f}% " +
                      ("— טווח צר, צבירה שקטה." if inr else "— תנודתי מדי (>12%).")})

    vdict = "סבירות גבוהה לאיסוף מוסדי" if score>=75 else "סימנים חלקיים" if score>=45 else "אין ראיות לאיסוף"
    vcol  = "#26a69a" if score>=75 else "#ffa726" if score>=45 else "#ef5350"
    return score, criteria, vdict, vcol, prereq, drawdown

def render_wyckoff_chart(df):
    dc = df.iloc[-65:].copy()
    fig = make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.7,0.3],vertical_spacing=0.04)
    fig.add_trace(go.Candlestick(x=dc.index,open=dc["Open"],high=dc["High"],low=dc["Low"],close=dc["Close"],
        increasing_line_color="#26a69a",decreasing_line_color="#ef5350",name="Price"),row=1,col=1)
    vc=["#26a69a" if c>=o else "#ef5350" for c,o in zip(dc["Close"],dc["Open"])]
    fig.add_trace(go.Bar(x=dc.index,y=dc["Volume"],marker_color=vc,name="Volume",opacity=0.8),row=2,col=1)
    fig.add_trace(go.Scatter(x=dc.index,y=dc["VOL_MEAN"],line=dict(color="#4fc3f7",width=1.5,dash="dot"),name="Vol MA20"),row=2,col=1)
    fig.update_layout(height=420,paper_bgcolor="#0a1520",plot_bgcolor="#0d1b2a",font_color="#e0eaf4",
                      xaxis_rangeslider_visible=False,legend=dict(orientation="h",y=1.02,x=0),margin=dict(t=10,b=10))
    fig.update_xaxes(gridcolor="#1e3040"); fig.update_yaxes(gridcolor="#1e3040")
    return fig

# ============================================================
# VOLUME PROFILE LOGIC
# ============================================================
def build_volume_profile(df, bins=40):
    price_min=df["Low"].min(); price_max=df["High"].max()
    edges=np.linspace(price_min,price_max,bins+1); vap=np.zeros(bins)
    for _,row in df.iterrows():
        lo,hi,vol=row["Low"],row["High"],row["Volume"]
        if hi==lo: continue
        for i in range(bins):
            ol=max(edges[i],lo); oh=min(edges[i+1],hi)
            if oh>ol: vap[i]+=vol*(oh-ol)/(hi-lo)
    return (edges[:-1]+edges[1:])/2, vap, edges

def analyze_vp(df):
    score=0; criteria=[]; cur=df["Close"].iloc[-1]
    mids,vap,edges=build_volume_profile(df)
    total=vap.sum()
    poc_idx=np.argmax(vap); poc=mids[poc_idx]; poc_pct=vap[poc_idx]/total*100
    # Value Area
    si=np.argsort(vap)[::-1]; va_vol=0; va_idx=[]
    for i in si:
        if va_vol>=total*0.70: break
        va_vol+=vap[i]; va_idx.append(i)
    vah=mids[max(va_idx)]; val=mids[min(va_idx)]
    # LVN below
    below=(mids<cur)
    lvn_thr=np.percentile(vap,20)
    has_lvn=np.sum(vap[below]<lvn_thr)>=2 if below.any() else False
    # HVN above
    above=(mids>cur)
    hvn_thr=np.percentile(vap,75)
    hvn_cnt=np.sum(vap[above]>hvn_thr) if above.any() else 0
    has_hvn=hvn_cnt>=1
    # Near POC
    poc_dist=abs(cur-poc)/poc; near_poc=poc_dist<=0.03
    # Below VAL
    below_val=cur<val; bval_pct=(val-cur)/val*100 if below_val else 0
    # Vol surge
    r30=df.iloc[-30:]; rm,rv,_=build_volume_profile(r30,bins=20)
    ci=np.argmin(np.abs(rm-cur)); surge=rv[ci]>rv.mean()*1.8

    # Scoring
    bp=25 if below_val else 0; score+=bp
    criteria.append({"name":"מחיר מתחת ל-VAL","hit":below_val,"points":25,"earned":bp,
        "explanation":(f"מחיר ({cur:.2f}) נמצא {bval_pct:.1f}% מתחת ל-VAL ({val:.2f}) — דיסקאונט ביחס לאזור הפעילות המוסדי."
                       if below_val else f"מחיר ({cur:.2f}) בתוך Value Area או מעליו (VAL={val:.2f}, VAH={vah:.2f}).")})

    lp=20 if has_lvn else 0; score+=lp
    criteria.append({"name":"LVN — אזור ריק מתחת למחיר","hit":has_lvn,"points":20,"earned":lp,
        "explanation":("זוהו LVN מתחת למחיר — אזורי ריק שדרכם המחיר נוטה לנוע מהר כלפי מעלה."
                       if has_lvn else "לא זוהו LVN משמעותיים מתחת למחיר.")})

    hp=20 if has_hvn else 0; score+=hp
    criteria.append({"name":"HVN — ריכוז פעילות מעל המחיר","hit":has_hvn,"points":20,"earned":hp,
        "explanation":(f"זוהו {hvn_cnt} HVN מעל המחיר — אזורים בהם מוסדיים בנו פוזיציות בעבר."
                       if has_hvn else "לא זוהו HVN מעל המחיר.")})

    pp=20 if near_poc else 0; score+=pp
    criteria.append({"name":"קרוב ל-POC","hit":near_poc,"points":20,"earned":pp,
        "explanation":(f"מחיר ({cur:.2f}) במרחק {poc_dist*100:.1f}% מה-POC ({poc:.2f}) — אזור האיזון המוסדי."
                       if near_poc else f"מחיר רחוק מה-POC ({poc:.2f}) ב-{poc_dist*100:.1f}% (סף 3%).")})

    sp=15 if surge else 0; score+=sp
    criteria.append({"name":"Volume Surge ברמה הנוכחית","hit":surge,"points":15,"earned":sp,
        "explanation":(f"נפח חריג ב-30 ימים ברמת המחיר הנוכחית — ספיגת Smart Money טרייה."
                       if surge else "לא זוהה volume surge חריג ברמה הנוכחית.")})

    vdict="סבירות גבוהה לנוכחות מוסדית" if score>=75 else "סימנים חלקיים — ניטור מומלץ" if score>=45 else "אין ריכוז מוסדי מובהק"
    vcol="#ab47bc" if score>=75 else "#ffa726" if score>=45 else "#ef5350"
    vpd={"poc":poc,"vah":vah,"val":val,"midpoints":mids,"vol_at_price":vap,"poc_vol_pct":poc_pct}
    return score, criteria, vdict, vcol, vpd

def render_vp_chart(df, vpd, ticker):
    cur=df["Close"].iloc[-1]; dc=df.iloc[-65:].copy()
    mids=vpd["midpoints"]; vap=vpd["vol_at_price"]
    poc=vpd["poc"]; vah=vpd["vah"]; val=vpd["val"]
    step=mids[1]-mids[0]
    bar_colors=["#ce93d8" if abs(m-poc)<step else "#5c35a0" if val<=m<=vah else "#2a3a5a" for m in mids]
    fig=make_subplots(rows=1,cols=2,column_widths=[0.72,0.28],shared_yaxes=True,horizontal_spacing=0.01)
    fig.add_trace(go.Candlestick(x=dc.index,open=dc["Open"],high=dc["High"],low=dc["Low"],close=dc["Close"],
        increasing_line_color="#26a69a",decreasing_line_color="#ef5350",name="Price"),row=1,col=1)
    xr=[dc.index[0],dc.index[-1]]
    for lv,cl,lb in [(poc,"#ce93d8","POC"),(vah,"#4fc3f7","VAH"),(val,"#4fc3f7","VAL")]:
        fig.add_trace(go.Scatter(x=xr,y=[lv,lv],mode="lines+text",line=dict(color=cl,width=1.5,dash="dash"),
            text=["",f" {lb}:{lv:.2f}"],textposition="top right",textfont=dict(color=cl,size=10),name=lb),row=1,col=1)
    fig.add_trace(go.Scatter(x=xr,y=[cur,cur],mode="lines",line=dict(color="#fff",width=1,dash="dot"),
        name=f"Now:{cur:.2f}"),row=1,col=1)
    fig.add_trace(go.Bar(x=vap/vap.max()*100,y=mids,orientation='h',marker_color=bar_colors,
        name="VP",opacity=0.9,width=step*0.85),row=1,col=2)
    fig.update_layout(height=500,paper_bgcolor="#0a1520",plot_bgcolor="#0d1b2a",font_color="#e0eaf4",
                      xaxis_rangeslider_visible=False,legend=dict(orientation="h",y=1.04,x=0,font=dict(size=10)),
                      margin=dict(t=20,b=20,l=10,r=10))
    fig.update_xaxes(gridcolor="#1e3040"); fig.update_yaxes(gridcolor="#1e3040")
    fig.update_xaxes(title_text="Vol %",row=1,col=2)
    return fig

# ============================================================
# MARKET SCAN
# ============================================================
def run_market_scan(analyze_fn, mode, score_threshold=80, max_results=10):
    """
    Scan SCAN_UNIVERSE with dynamic filtering.
    score_threshold : only scores >= this value are kept (Accumulation Zone filter).
    max_results     : stop scanning early once this many hits are found.
    """
    hits   = []
    errors = []
    total  = len(SCAN_UNIVERSE)

    st.markdown(f"""
    <div style="background:#0a1520;border:1px solid #1e3040;border-radius:8px;padding:12px 18px;
                direction:rtl;color:#b0c8e0;font-size:0.85rem;margin-bottom:16px;">
    🔍 סורק <b style="color:#fff">{total}</b> מניות בבורסות ארה"ב.<br>
    פילטר: ציון ≥ <b style="color:#ffa726">{score_threshold}</b> &nbsp;|&nbsp;
    מגבלת תוצאות: <b style="color:#4fc3f7">{max_results} מניות</b> — הסריקה תיעצר ברגע שימצאו מספיק.
    </div>""", unsafe_allow_html=True)

    prog_bar  = st.progress(0)
    status_ph = st.empty()
    hits_ph   = st.empty()

    for i, ticker in enumerate(SCAN_UNIVERSE):
        # Early stop once we have enough hits
        if len(hits) >= max_results:
            status_ph.markdown(
                f"<div style='direction:rtl;font-family:IBM Plex Mono,monospace;font-size:0.82rem;"
                f"color:#26a69a'>✅ נמצאו {max_results} מניות — עצרנו מוקדם. "
                f"סרקנו {i}/{total} מניות.</div>",
                unsafe_allow_html=True
            )
            break

        status_ph.markdown(
            f"<div style='direction:ltr;font-family:IBM Plex Mono,monospace;font-size:0.78rem;"
            f"color:#607d8b'>Scanning {i+1}/{total}: {ticker} "
            f"&nbsp;|&nbsp; hits so far: {len(hits)}/{max_results}</div>",
            unsafe_allow_html=True
        )
        prog_bar.progress((i + 1) / total)

        df = get_data(ticker)
        if df is None:
            errors.append(ticker)
            continue

        try:
            if mode == "wyckoff":
                score, criteria, verdict, vcolor, prereq, drawdown = analyze_fn(df)
            else:
                score, criteria, verdict, vcolor, vpd = analyze_fn(df)

            if score >= score_threshold:
                hits.append({"ticker": ticker, "score": score,
                             "verdict": verdict, "verdict_color": vcolor})
        except Exception:
            errors.append(ticker)
            continue

        # Live hits preview after every new hit (or every 15 tickers)
        if hits and (i % 15 == 0 or len(hits) != len(hits)):
            top = sorted(hits, key=lambda x: x["score"], reverse=True)
            rows = "".join([
                f"""<div class="scan-result-row">
                  <span style="color:#4fc3f7;font-weight:600;min-width:70px;display:inline-block">{h['ticker']}</span>
                  <span class="scan-badge {'badge-green' if h['score']>=90 else 'badge-yellow'}">{h['score']}/100</span>
                  <span style="color:#607d8b;margin-right:12px;font-size:0.75rem"> {h['verdict']}</span>
                </div>""" for h in top
            ])
            hits_ph.markdown(
                f"<div style='direction:ltr'><b style='color:#fff;font-family:IBM Plex Mono,monospace'>"
                f"Accumulation Zone hits ({len(hits)}/{max_results}):</b>{rows}</div>",
                unsafe_allow_html=True
            )

    prog_bar.empty()
    status_ph.empty()
    hits_ph.empty()
    return sorted(hits, key=lambda x: x["score"], reverse=True), errors

def render_scan_results(hits, errors, mode, score_threshold=80):
    accent = "#4fc3f7" if mode=="wyckoff" else "#ce93d8"

    if not hits:
        st.warning(f"לא נמצאו מניות עם ציון ≥{score_threshold} בסריקה זו. נסה להוריד את הסף או לרוץ שוב מאוחר יותר.")
        return

    st.markdown(f"""
    <div style="direction:rtl;margin-bottom:16px;">
    <b style="color:{accent};font-family:'IBM Plex Mono',monospace;font-size:1rem;">
    ✅ נמצאו {len(hits)} מניות עם ציון ≥{score_threshold} — Accumulation Zone</b>
    </div>""", unsafe_allow_html=True)

    # Summary cards (top 8)
    top8 = hits[:8]
    cols = st.columns(min(len(top8), 4))
    for idx, h in enumerate(top8):
        with cols[idx % 4]:
            st.markdown(f"""
            <div class="overview-card {'vp-card' if mode=='vp' else ''}">
              <div class="ticker-label" style="color:{accent}">{h['ticker']}</div>
              <div class="score-big" style="color:{h['verdict_color']}">{h['score']}</div>
              <div style="color:#607d8b;font-size:0.72rem;font-family:'IBM Plex Mono',monospace;">/ 100</div>
              <div class="verdict-label">{h['verdict']}</div>
              <div class="bar-bg"><div class="bar-fill" style="width:{h['score']}%;background:{h['verdict_color']}"></div></div>
            </div>""", unsafe_allow_html=True)

    # Full ranked table
    st.markdown("---")
    st.markdown(f"### רשימה מלאה — ממוין לפי ציון")
    header = f"""
    <div style="direction:ltr;display:grid;grid-template-columns:60px 80px 1fr 120px;
                gap:8px;padding:8px 18px;font-family:'IBM Plex Mono',monospace;
                font-size:0.78rem;color:#607d8b;border-bottom:1px solid #1e3040;">
      <span>#</span><span>Ticker</span><span>Verdict</span><span>Score</span>
    </div>"""
    st.markdown(header, unsafe_allow_html=True)

    for rank, h in enumerate(hits, 1):
        bc = "badge-green" if h["score"]>=90 else "badge-yellow" if h["score"]>=80 else "badge-red"
        st.markdown(f"""
        <div style="direction:ltr;display:grid;grid-template-columns:60px 80px 1fr 120px;
                    gap:8px;padding:10px 18px;font-family:'IBM Plex Mono',monospace;
                    font-size:0.84rem;border-bottom:1px solid #0f1e2e;align-items:center;">
          <span style="color:#607d8b">{rank}</span>
          <span style="color:{accent};font-weight:600">{h['ticker']}</span>
          <span style="color:#b0c8e0;font-size:0.78rem">{h['verdict']}</span>
          <span class="scan-badge {bc}">{h['score']}/100</span>
        </div>""", unsafe_allow_html=True)

    # Option: load any hit into detailed analysis
    st.markdown("---")
    st.markdown("### פתח ניתוח מפורט למניה מהרשימה")
    ticker_pick = st.selectbox("בחר מניה", [h["ticker"] for h in hits], key=f"scan_pick_{mode}")
    if st.button("▶ פתח ניתוח מפורט", key=f"scan_detail_{mode}"):
        df = get_data(ticker_pick)
        if df:
            if mode == "wyckoff":
                sc,cr,vd,vc,pm,dd = analyze_wyckoff(df)
                _render_wyckoff_detail(ticker_pick, df, sc, cr, vd, vc, pm, dd)
            else:
                sc,cr,vd,vc,vpd = analyze_vp(df)
                _render_vp_detail(ticker_pick, df, sc, cr, vd, vc, vpd)

    if errors:
        with st.expander(f"⚠️ {len(errors)} טיקרים שדולגו (שגיאת דאטה)"):
            st.write(", ".join(errors))

# ============================================================
# DETAIL RENDERERS (shared between specific & scan)
# ============================================================
def _render_wyckoff_detail(t, df, score, criteria, verdict, verdict_color, prereq, drawdown):
    cg, cr = st.columns([1,1], gap="large")
    with cg:
        st.plotly_chart(render_gauge(score, verdict, verdict_color, "wyckoff"), use_container_width=True)
        if not prereq:
            st.markdown(f"""<div class="score-reason-box negative">
            ⚠️ <strong>תנאי מקדים לא מתקיים:</strong> ירידה {drawdown*100:.1f}% בלבד (נדרש ≥12%).
            </div>""", unsafe_allow_html=True)
    with cr:
        st.markdown("#### פירוט הניקוד")
        for c in criteria:
            box="positive" if c["hit"] else "negative"
            lbl="✅ הצליח" if c["hit"] else "❌ נכשל"
            cls="hit" if c["hit"] else "miss"
            st.markdown(f"""
            <div class="score-reason-box {box}">
              <div class="criteria-row">
                <strong>{c['name']}</strong>
                <span><span class="{cls}">{lbl}</span> &nbsp;|&nbsp; <strong>{c['earned']}/{c['points']} נק'</strong></span>
              </div>
              <div style="margin-top:6px;color:#b0c8e0">{c['explanation']}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown(f"##### גרף — {t}")
    st.plotly_chart(render_wyckoff_chart(df), use_container_width=True)

def _render_vp_detail(t, df, score, criteria, verdict, verdict_color, vpd):
    cur = df["Close"].iloc[-1]
    st.markdown(f"""
    <div style="direction:ltr;font-family:'IBM Plex Mono',monospace;font-size:0.82rem;
                color:#b0b0c0;background:#0d1220;border-radius:6px;padding:8px 14px;margin-bottom:12px;">
      Current: <b style="color:#fff">{cur:.2f}</b> &nbsp;|&nbsp;
      POC: <b style="color:#ce93d8">{vpd['poc']:.2f}</b> &nbsp;|&nbsp;
      VAH: <b style="color:#4fc3f7">{vpd['vah']:.2f}</b> &nbsp;|&nbsp;
      VAL: <b style="color:#4fc3f7">{vpd['val']:.2f}</b> &nbsp;|&nbsp;
      POC vol share: <b style="color:#ce93d8">{vpd['poc_vol_pct']:.1f}%</b>
    </div>""", unsafe_allow_html=True)
    cg, cr = st.columns([1,1], gap="large")
    with cg:
        st.plotly_chart(render_gauge(score, verdict, verdict_color, "vp"), use_container_width=True)
    with cr:
        st.markdown("#### פירוט הניקוד")
        for c in criteria:
            box="vp-positive" if c["hit"] else "vp-negative"
            lbl="✅ הצליח" if c["hit"] else "❌ נכשל"
            cls="hit" if c["hit"] else "miss"
            st.markdown(f"""
            <div class="score-reason-box {box}">
              <div class="criteria-row">
                <strong>{c['name']}</strong>
                <span><span class="{cls}">{lbl}</span> &nbsp;|&nbsp; <strong>{c['earned']}/{c['points']} נק'</strong></span>
              </div>
              <div style="margin-top:6px;color:#c8b0d8">{c['explanation']}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown(f"##### Volume Profile — {t}")
    st.plotly_chart(render_vp_chart(df, vpd, t), use_container_width=True)

# ============================================================
# WYCKOFF SCREEN
# ============================================================
def screen_wyckoff():
    st.markdown("""
    <div class="header-box wyckoff">
      <h2>⬛ WYCKOFF ACCUMULATION SCOUT</h2>
      <p>מחפש חתימות של <strong>איסוף מוסדי מוקדם</strong> לפי מתודולוגיית ריצ'רד וייקוף.
      מוסדיים בונים פוזיציות בשקט, כשהציבור עדיין פוחד.</p>
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
        <strong>No Supply:</strong> ווליום &lt;70% מהממוצע ב-10 ימים — המוכרים נעלמו.<br>
        <strong>Divergence:</strong> מחיר יורד אבל ווליום קורס.<br>
        <strong>Trading Range:</strong> טווח 15 ימים &lt;12% — צבירה שקטה.
      </p>
      <p style="color:#607d8b;font-size:0.82rem;">⚠️ תנאי מקדים: ירידה ≥12% מהשיא ב-3 חודשים.</p>
    </div>""", unsafe_allow_html=True)

    # Sub-mode toggle
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🎯  מניות ספציפיות", use_container_width=True,
                     type="primary" if st.session_state.w_submode=="specific" else "secondary", key="w_sub_spec"):
            st.session_state.w_submode = "specific"; st.rerun()
    with c2:
        if st.button("🌐  סריקת שוק כללית", use_container_width=True,
                     type="primary" if st.session_state.w_submode=="scan" else "secondary", key="w_sub_scan"):
            st.session_state.w_submode = "scan"; st.rerun()

    st.markdown("")

    # ---------- SPECIFIC ----------
    if st.session_state.w_submode == "specific":
        ci, cb = st.columns([4,1])
        with ci:
            raw = st.text_input("טיקרים (פסיק או רווח)", "NVDA, MSFT, AMZN", key="w_input")
        with cb:
            st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
            run = st.button("▶ הרץ", use_container_width=True, key="w_run")
        if not run: return

        tickers = list(dict.fromkeys([t.strip().upper() for t in raw.replace(","," ").split() if t.strip()]))
        if not tickers: st.error("יש להזין לפחות טיקר אחד."); return

        results = {}
        prog = st.progress(0, text="שולף דאטה...")
        for i,t in enumerate(tickers):
            prog.progress((i+1)/len(tickers), text=f"מנתח {t}...")
            df = get_data(t)
            if df is None: results[t]=None; continue
            sc,cr,vd,vc,pm,dd = analyze_wyckoff(df)
            results[t]={"df":df,"score":sc,"criteria":cr,"verdict":vd,"verdict_color":vc,"prereq_met":pm,"drawdown":dd}
        prog.empty()

        valid = {t:v for t,v in results.items() if v}
        failed= [t for t in results if not results[t]]
        if failed: st.warning(f"לא נמצא דאטה עבור: {', '.join(failed)}")
        if not valid: st.error("לא נמצא דאטה תקין."); return

        if len(valid)>1:
            st.markdown("---"); st.markdown("### סקירה כללית")
            st_sorted = sorted(valid.keys(), key=lambda t: valid[t]["score"], reverse=True)
            cols = st.columns(len(st_sorted))
            for col,t in zip(cols,st_sorted):
                r=valid[t]; s=r["score"]; c=r["verdict_color"]
                with col:
                    st.markdown(f"""
                    <div class="overview-card">
                      <div class="ticker-label" style="color:#4fc3f7">{t}</div>
                      <div class="score-big" style="color:{c}">{s}</div>
                      <div style="color:#607d8b;font-size:0.72rem;font-family:'IBM Plex Mono',monospace;">/ 100</div>
                      <div class="verdict-label">{r['verdict']}</div>
                      <div class="bar-bg"><div class="bar-fill" style="width:{s}%;background:{c}"></div></div>
                    </div>""", unsafe_allow_html=True)
            fig_c,_=render_comparison_chart(valid); st.plotly_chart(fig_c, use_container_width=True)

        st.markdown("---"); st.markdown("### ניתוח פרטני")
        tabs = st.tabs([f"{'🟢' if valid[t]['score']>=75 else '🟡' if valid[t]['score']>=45 else '🔴'} {t}" for t in valid])
        for tab,t in zip(tabs,valid):
            with tab:
                r=valid[t]
                _render_wyckoff_detail(t,r["df"],r["score"],r["criteria"],r["verdict"],r["verdict_color"],r["prereq_met"],r["drawdown"])

    # ---------- SCAN ----------
    else:
        st.markdown("""
        <div style="background:#0f1a10;border:1px solid #2a4a2a;border-radius:8px;
                    padding:14px 20px;direction:rtl;color:#b0d8b0;font-size:0.88rem;margin-bottom:16px;">
        הסריקה עוברת על ~230 מניות מ-S&P 500, Nasdaq ומגזרים שונים.<br>
        הגדר את הפילטר למטה — הבוט יעצור ברגע שימצא את מספר המניות שביקשת.
        </div>""", unsafe_allow_html=True)

        st.markdown("#### ⚙️ פרמטרי סריקה")
        fc1, fc2 = st.columns(2)
        with fc1:
            w_min_score = st.slider(
                "ציון מינימום (Accumulation Zone)",
                min_value=60, max_value=95, value=80, step=5,
                help="רק מניות שעברו את הסף הזה יוצגו. 80+ = ציון אמין. 90+ = איתות חזק מאוד.",
                key="w_min_score"
            )
        with fc2:
            w_max_results = st.slider(
                "מקסימום תוצאות",
                min_value=3, max_value=30, value=10, step=1,
                help="הסריקה תיעצר ברגע שתמצא את המספר הזה של מניות — חוסך זמן משמעותי.",
                key="w_max_results"
            )

        # Visual zone label
        zone_label = "🟢 Accumulation Zone חזק" if w_min_score >= 85 else \
                     "🟡 Accumulation Zone סביר" if w_min_score >= 75 else \
                     "🟠 רף נמוך — תוצאות רבות יותר, פחות אמינות"
        st.markdown(f"<div style='direction:rtl;font-size:0.82rem;color:#b0c8e0;margin-bottom:12px'>"
                    f"פילטר נוכחי: <b>{zone_label}</b> | ציון ≥ <b style='color:#ffa726'>{w_min_score}</b> "
                    f"| עד <b style='color:#4fc3f7'>{w_max_results}</b> תוצאות</div>",
                    unsafe_allow_html=True)

        if st.button("🚀  התחל סריקת שוק — Wyckoff", use_container_width=True, key="w_scan_go"):
            hits, errors = run_market_scan(analyze_wyckoff, "wyckoff",
                                           score_threshold=w_min_score,
                                           max_results=w_max_results)
            render_scan_results(hits, errors, "wyckoff", score_threshold=w_min_score)

    st.markdown("""<div class="disclaimer">⚠️ אנליזה טכנית בלבד, אינה המלצת השקעה. תמיד בצע Due Diligence עצמאי.</div>""",
                unsafe_allow_html=True)


# ============================================================
# VOLUME PROFILE SCREEN
# ============================================================
def screen_vp():
    st.markdown("""
    <div class="header-box vp">
      <h2>🔮 VOLUME PROFILE SCOUT</h2>
      <p>מנתח <strong>פרופיל הנפח</strong> — פיזור הנפח לפי רמות מחיר לאורך שנה.
      VP מסתכל על <em>מחיר</em>, לא על זמן: איפה המוסדיים באמת ישבו.</p>
      <p>
        <span class="tag tag-v">POC – Point of Control</span>
        <span class="tag tag-v">Value Area (VAH/VAL)</span>
        <span class="tag tag-v">HVN – High Volume Node</span>
        <span class="tag tag-v">LVN – Low Volume Node</span>
      </p>
      <p>
        <strong>POC:</strong> רמת המחיר עם הנפח הגבוה ביותר — אזור האיזון המוסדי.<br>
        <strong>Value Area:</strong> 70% מסך הנפח — טווח הפעילות של "הכסף החכם".<br>
        <strong>VAH/VAL:</strong> מחיר מתחת ל-VAL = דיסקאונט מובהק.<br>
        <strong>HVN:</strong> ריכוז פעילות — תמיכה/התנגדות חזקה.<br>
        <strong>LVN:</strong> אזור ריק — המחיר נוטה לנוע דרכו מהר.
      </p>
      <p>
        <span class="tag tag-v">מחיר מתחת ל-VAL · 25 נק'</span>
        <span class="tag tag-v">LVN מתחת למחיר · 20 נק'</span>
        <span class="tag tag-v">HVN מעל המחיר · 20 נק'</span>
        <span class="tag tag-v">קרוב ל-POC · 20 נק'</span>
        <span class="tag tag-v">Volume Surge ברמה · 15 נק'</span>
      </p>
      <p style="color:#607d8b;font-size:0.82rem;">✦ הגרף: פרופיל אופקי — סגול כהה=Value Area, סגול בהיר=POC.</p>
    </div>""", unsafe_allow_html=True)

    # Sub-mode toggle
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🎯  מניות ספציפיות", use_container_width=True,
                     type="primary" if st.session_state.vp_submode=="specific" else "secondary", key="vp_sub_spec"):
            st.session_state.vp_submode = "specific"; st.rerun()
    with c2:
        if st.button("🌐  סריקת שוק כללית", use_container_width=True,
                     type="primary" if st.session_state.vp_submode=="scan" else "secondary", key="vp_sub_scan"):
            st.session_state.vp_submode = "scan"; st.rerun()

    st.markdown("")

    # ---------- SPECIFIC ----------
    if st.session_state.vp_submode == "specific":
        ci, cb = st.columns([4,1])
        with ci:
            raw = st.text_input("טיקרים (פסיק או רווח)", "NVDA, MSFT, AMZN", key="vp_input")
        with cb:
            st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
            run = st.button("▶ הרץ", use_container_width=True, key="vp_run")
        if not run: return

        tickers = list(dict.fromkeys([t.strip().upper() for t in raw.replace(","," ").split() if t.strip()]))
        if not tickers: st.error("יש להזין לפחות טיקר אחד."); return

        results = {}
        prog = st.progress(0, text="בונה פרופיל נפח...")
        for i,t in enumerate(tickers):
            prog.progress((i+1)/len(tickers), text=f"מנתח {t}...")
            df = get_data(t)
            if df is None: results[t]=None; continue
            sc,cr,vd,vc,vpd = analyze_vp(df)
            results[t]={"df":df,"score":sc,"criteria":cr,"verdict":vd,"verdict_color":vc,"vp_data":vpd}
        prog.empty()

        valid = {t:v for t,v in results.items() if v}
        failed= [t for t in results if not results[t]]
        if failed: st.warning(f"לא נמצא דאטה עבור: {', '.join(failed)}")
        if not valid: st.error("לא נמצא דאטה תקין."); return

        if len(valid)>1:
            st.markdown("---"); st.markdown("### סקירה כללית")
            st_sorted = sorted(valid.keys(), key=lambda t: valid[t]["score"], reverse=True)
            cols = st.columns(len(st_sorted))
            for col,t in zip(cols,st_sorted):
                r=valid[t]; s=r["score"]; c=r["verdict_color"]; vpd=r["vp_data"]
                with col:
                    st.markdown(f"""
                    <div class="overview-card vp-card">
                      <div class="ticker-label" style="color:#ce93d8">{t}</div>
                      <div class="score-big" style="color:{c}">{s}</div>
                      <div style="color:#607d8b;font-size:0.72rem;font-family:'IBM Plex Mono',monospace;">/ 100</div>
                      <div class="verdict-label">{r['verdict']}</div>
                      <div style="font-size:0.72rem;color:#8a6a9a;margin-top:6px;font-family:'IBM Plex Mono',monospace;">
                        POC {vpd['poc']:.2f} | VAH {vpd['vah']:.2f} | VAL {vpd['val']:.2f}
                      </div>
                      <div class="bar-bg"><div class="bar-fill" style="width:{s}%;background:{c}"></div></div>
                    </div>""", unsafe_allow_html=True)
            fig_c,_=render_comparison_chart(valid); st.plotly_chart(fig_c, use_container_width=True)

        st.markdown("---"); st.markdown("### ניתוח פרטני")
        tabs = st.tabs([f"{'🟣' if valid[t]['score']>=75 else '🟡' if valid[t]['score']>=45 else '🔴'} {t}" for t in valid])
        for tab,t in zip(tabs,valid):
            with tab:
                r=valid[t]
                _render_vp_detail(t,r["df"],r["score"],r["criteria"],r["verdict"],r["verdict_color"],r["vp_data"])

    # ---------- SCAN ----------
    else:
        st.markdown("""
        <div style="background:#160d20;border:1px solid #3a1a4a;border-radius:8px;
                    padding:14px 20px;direction:rtl;color:#c8b0d8;font-size:0.88rem;margin-bottom:16px;">
        הסריקה עוברת על ~230 מניות. בניית פרופיל נפח לכל מניה לוקחת יותר זמן מוייקוף.<br>
        הגדר את הפילטר למטה — הבוט יעצור ברגע שימצא את מספר המניות שביקשת.
        </div>""", unsafe_allow_html=True)

        st.markdown("#### ⚙️ פרמטרי סריקה")
        fc1, fc2 = st.columns(2)
        with fc1:
            vp_min_score = st.slider(
                "ציון מינימום (Accumulation Zone)",
                min_value=60, max_value=95, value=80, step=5,
                help="רק מניות שעברו את הסף הזה יוצגו. 80+ = ציון אמין. 90+ = איתות חזק מאוד.",
                key="vp_min_score"
            )
        with fc2:
            vp_max_results = st.slider(
                "מקסימום תוצאות",
                min_value=3, max_value=30, value=10, step=1,
                help="הסריקה תיעצר ברגע שתמצא את המספר הזה של מניות — חוסך זמן משמעותי.",
                key="vp_max_results"
            )

        zone_label = "🟣 Accumulation Zone חזק" if vp_min_score >= 85 else \
                     "🟡 Accumulation Zone סביר" if vp_min_score >= 75 else \
                     "🟠 רף נמוך — תוצאות רבות יותר, פחות אמינות"
        st.markdown(f"<div style='direction:rtl;font-size:0.82rem;color:#c8b0d8;margin-bottom:12px'>"
                    f"פילטר נוכחי: <b>{zone_label}</b> | ציון ≥ <b style='color:#ffa726'>{vp_min_score}</b> "
                    f"| עד <b style='color:#ce93d8'>{vp_max_results}</b> תוצאות</div>",
                    unsafe_allow_html=True)

        if st.button("🚀  התחל סריקת שוק — Volume Profile", use_container_width=True, key="vp_scan_go"):
            hits, errors = run_market_scan(analyze_vp, "vp",
                                           score_threshold=vp_min_score,
                                           max_results=vp_max_results)
            render_scan_results(hits, errors, "vp", score_threshold=vp_min_score)

    st.markdown("""<div class="disclaimer">⚠️ אנליזה טכנית בלבד, אינה המלצת השקעה. תמיד בצע Due Diligence עצמאי.</div>""",
                unsafe_allow_html=True)


# ============================================================
# ROUTER
# ============================================================
if st.session_state.mode == "wyckoff":
    screen_wyckoff()
else:
    screen_vp()
