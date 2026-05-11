import streamlit as st
import requests
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="AI Heist", page_icon="🔴", layout="wide")

# ── THEME CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Bebas+Neue&family=Rajdhani:wght@400;600&display=swap');

html, body, [class*="css"] {
    background-color: #0a0a0a !important;
    color: #e0e0e0 !important;
    font-family: 'Rajdhani', sans-serif !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0f0f0f !important;
    border-right: 1px solid #ff2020 !important;
}

/* Main area */
[data-testid="stAppViewContainer"] {
    background-color: #0a0a0a !important;
}

/* Tabs */
[data-testid="stTabs"] button {
    font-family: 'Share Tech Mono', monospace !important;
    color: #888 !important;
    font-size: 14px !important;
    letter-spacing: 2px !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #ff2020 !important;
    border-bottom: 2px solid #ff2020 !important;
}

/* Sliders */
[data-testid="stSlider"] > div > div > div {
    background: #ff2020 !important;
}

/* Buttons */
[data-testid="stButton"] > button {
    background: transparent !important;
    border: 1px solid #ff2020 !important;
    color: #ff2020 !important;
    font-family: 'Share Tech Mono', monospace !important;
    letter-spacing: 3px !important;
    font-size: 13px !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.2s !important;
}
[data-testid="stButton"] > button:hover {
    background: #ff2020 !important;
    color: #000 !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #111 !important;
    border: 1px solid #1f1f1f !important;
    border-left: 3px solid #ff2020 !important;
    padding: 1rem !important;
    border-radius: 4px !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Share Tech Mono', monospace !important;
    color: #666 !important;
    font-size: 11px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Bebas Neue', cursive !important;
    color: #ff2020 !important;
    font-size: 2rem !important;
}

/* Progress bar */
[data-testid="stProgressBar"] > div > div {
    background: #ff2020 !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #1f1f1f !important;
}

/* Number input */
[data-testid="stNumberInput"] input {
    background: #111 !important;
    border: 1px solid #333 !important;
    color: #e0e0e0 !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* Success / Error boxes */
[data-testid="stAlert"] {
    font-family: 'Share Tech Mono', monospace !important;
    border-radius: 2px !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0a0a; }
::-webkit-scrollbar-thumb { background: #ff2020; }

/* Glitch keyframes */
@keyframes glitch {
    0%   { text-shadow: 2px 0 #ff2020, -2px 0 #00ffff; }
    25%  { text-shadow: -2px 0 #ff2020, 2px 0 #00ffff; }
    50%  { text-shadow: 2px 2px #ff2020, -2px -2px #00ffff; }
    75%  { text-shadow: -2px 2px #ff2020, 2px -2px #00ffff; }
    100% { text-shadow: 2px 0 #ff2020, -2px 0 #00ffff; }
}
@keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0; }
}
@keyframes scanline {
    0%   { top: -10%; }
    100% { top: 110%; }
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

API_URL   = "http://localhost:5000/predict"
STATS_URL = "http://localhost:5000/stats"
COPYCAT_PATH = os.path.join(os.path.dirname(__file__), "../model/copycat_model.pkl")
STOLEN_CSV   = os.path.join(os.path.dirname(__file__), "../stolen_data.csv")

FEATURE_RANGES = {
    "variance":  (-7.0,   7.0),
    "skewness":  (-14.0, 14.0),
    "curtosis":  (-5.0,  18.0),
    "entropy":   (-8.5,   2.5),
}

def query_api(features):
    try:
        r = requests.post(API_URL, json={"features": features}, timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def api_online():
    try:
        requests.get("http://localhost:5000", timeout=2)
        return True
    except:
        return False

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem;'>
        <div style='font-family: Bebas Neue, cursive; font-size: 2.2rem;
                    letter-spacing: 6px; color: #ff2020;
                    animation: glitch 3s infinite;'>AI HEIST</div>
        <div style='font-family: Share Tech Mono, monospace; font-size: 10px;
                    color: #444; letter-spacing: 3px; margin-top: 2px;'>
            MODEL STEALING FRAMEWORK
        </div>
    </div>
    <hr style='border-color: #1a1a1a; margin: 0.5rem 0 1rem;'>
    """, unsafe_allow_html=True)

    online = api_online()
    status_color = "#ff2020" if online else "#444"
    status_text  = "TARGET ONLINE" if online else "TARGET OFFLINE"
    st.markdown(f"""
    <div style='font-family: Share Tech Mono, monospace; font-size: 11px;
                color: {status_color}; letter-spacing: 2px; margin-bottom: 1rem;'>
        ● {status_text}
    </div>
    """, unsafe_allow_html=True)

    try:
        s = requests.get(STATS_URL, timeout=2).json()
        st.metric("QUERIES INTERCEPTED", s["total_queries"])
        st.metric("UPTIME (s)", s["uptime_seconds"])
    except:
        pass



# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding: 2rem 0 1rem; animation: fadeIn 0.8s ease;'>
    <div style='font-family: Bebas Neue, cursive; font-size: 3.5rem;
                letter-spacing: 10px; color: #ff2020;
                animation: glitch 4s infinite;'>
        AI HEIST
    </div>
    <div style='font-family: Share Tech Mono, monospace; font-size: 12px;
                color: #444; letter-spacing: 4px; margin-top: -8px;'>
        BANKNOTE AUTHENTICATION
    </div>
    <div style='margin-top: 0.8rem; padding: 0.6rem 1rem;
                border: 1px solid #1a1a1a; border-left: 3px solid #ff2020;
                font-family: Share Tech Mono, monospace; font-size: 11px;
                color: #555; display: inline-block;'>
        ⚠ FOR EDUCATIONAL PURPOSES ONLY — UNAUTHORIZED MODEL EXTRACTION DEMO
    </div>
</div>
<hr style='border-color: #1a1a1a; margin-bottom: 1.5rem;'>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["PREDICT", "ATTACK", "RESULTS"])

# ── TAB 1: PREDICT ────────────────────────────────────────────────────────────
with tab1:


    col1, col2 = st.columns(2)
    with col1:
        variance = st.slider("VARIANCE",  -7.0,  7.0,  0.0)
        skewness = st.slider("SKEWNESS", -14.0, 14.0,  0.0)
        curtosis = st.slider("CURTOSIS",  -5.0, 18.0,  0.0)
        entropy  = st.slider("ENTROPY",   -8.5,  2.5,  0.0)
        btn = st.button("► EXECUTE PREDICTION", use_container_width=True)

    with col2:
        if btn:
            result = query_api([variance, skewness, curtosis, entropy])
            if "error" in result:
                st.error(f"CONNECTION FAILED: {result['error']}")
            else:
                label = result["label"]
                if label == "Real":
                    st.markdown("""
                    <div style='padding: 2rem; border: 1px solid #00ff41;
                                border-left: 4px solid #00ff41; margin-top: 1rem;
                                animation: fadeIn 0.5s ease;'>
                        <div style='font-family: Bebas Neue, cursive; font-size: 2rem;
                                    color: #00ff41; letter-spacing: 4px;'>
                            ✓ REAL
                        </div>
                        <div style='font-family: Share Tech Mono, monospace;
                                    font-size: 11px; color: #444; margin-top: 4px;'>
                            BANKNOTE VERIFIED — REAL
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style='padding: 2rem; border: 1px solid #ff2020;
                                border-left: 4px solid #ff2020; margin-top: 1rem;
                                animation: fadeIn 0.5s ease;'>
                        <div style='font-family: Bebas Neue, cursive; font-size: 2rem;
                                    color: #ff2020; letter-spacing: 4px;'>
                            ✗ FAKE
                        </div>
                        <div style='font-family: Share Tech Mono, monospace;
                                    font-size: 11px; color: #444; margin-top: 4px;'>
                            FORGERY DETECTED — FAKE BANKNOTE
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='padding: 2rem; border: 1px solid #1a1a1a;
                        margin-top: 1rem; font-family: Share Tech Mono, monospace;
                        font-size: 11px; color: #333; letter-spacing: 2px;'>
                AWAITING INPUT VECTOR...<br>
                <span style='animation: blink 1s infinite;
                             display: inline-block; color: #ff2020;'>█</span>
            </div>
            """, unsafe_allow_html=True)

# ── TAB 2: ATTACK ─────────────────────────────────────────────────────────────
with tab2:


    col1, col2 = st.columns(2)
    with col1:
        n_queries = st.number_input("NUMBER OF PROBE QUERIES", 100, 3000, 1000, 100)
        run = st.button("► LAUNCH HEIST", use_container_width=True)

    with col2:
        if run:
            st.markdown("""
            <div style='font-family: Share Tech Mono, monospace; font-size: 11px;
                        color: #ff2020; letter-spacing: 2px; margin-bottom: 1rem;'>
                ⚠ OPERATION INITIATED — PROBING TARGET...
            </div>
            """, unsafe_allow_html=True)

            rng  = np.random.default_rng(42)
            rows = []
            bar  = st.progress(0, text="EXTRACTING...")
            log  = st.empty()

            for i in range(int(n_queries)):
                features = [float(rng.uniform(lo, hi)) for lo, hi in FEATURE_RANGES.values()]
                result   = query_api(features)
                if "error" not in result:
                    rows.append(features + [result["prediction"]])
                if (i + 1) % max(1, int(n_queries) // 20) == 0:
                    bar.progress((i + 1) / int(n_queries))
                    log.markdown(f"""
                    <div style='font-family: Share Tech Mono, monospace;
                                font-size: 11px; color: #444;'>
                        SAMPLES COLLECTED: {len(rows)} / {int(n_queries)}
                    </div>
                    """, unsafe_allow_html=True)

            df = pd.DataFrame(rows, columns=["variance","skewness","curtosis","entropy","label"])
            df.to_csv(STOLEN_CSV, index=False)

            X = df[["variance","skewness","curtosis","entropy"]].values
            y = df["label"].values
            copycat = RandomForestClassifier(n_estimators=100, random_state=42)
            copycat.fit(X, y)
            joblib.dump(copycat, COPYCAT_PATH)

            bar.progress(1.0)
            st.markdown(f"""
            <div style='padding: 1rem; border: 1px solid #ff2020;
                        border-left: 4px solid #ff2020; margin-top: 1rem;
                        font-family: Share Tech Mono, monospace; font-size: 11px;
                        color: #ff2020; letter-spacing: 2px;
                        animation: fadeIn 0.5s ease;'>
                ✓ HEIST COMPLETE<br>
                <span style='color: #444;'>
                SAMPLES STOLEN: {len(rows)}<br>
                COPYCAT MODEL: TRAINED AND SAVED<br>
                PROCEED TO RESULTS TAB
                </span>
            </div>
            """, unsafe_allow_html=True)

# ── TAB 3: RESULTS ────────────────────────────────────────────────────────────
with tab3:


    if not os.path.exists(COPYCAT_PATH):
        st.markdown("""
        <div style='font-family: Share Tech Mono, monospace; font-size: 12px;
                    color: #444; letter-spacing: 2px;'>
            ⚠ NO COPYCAT MODEL FOUND — EXECUTE HEIST FIRST
        </div>
        """, unsafe_allow_html=True)
    else:
        copycat  = joblib.load(COPYCAT_PATH)
        n_eval   = st.slider("TEST SAMPLES", 100, 500, 300, 50)
        compare  = st.button("► RUN COMPARISON", use_container_width=False)

        if compare:
            rng    = np.random.default_rng(99)
            X_test = np.column_stack([
                rng.uniform(lo, hi, int(n_eval))
                for lo, hi in FEATURE_RANGES.values()
            ])

            prog   = st.progress(0, text="QUERYING TARGET...")
            vpreds = []
            for idx, row in enumerate(X_test):
                r = query_api(row.tolist())
                vpreds.append(r.get("prediction", -1))
                if (idx + 1) % max(1, int(n_eval) // 10) == 0:
                    prog.progress((idx + 1) / int(n_eval))
            prog.empty()

            vpreds    = np.array(vpreds)
            valid     = vpreds != -1
            cpreds    = copycat.predict(X_test[valid])
            agreement = accuracy_score(vpreds[valid], cpreds)
            matching  = int((vpreds[valid] == cpreds).sum())
            mismatch  = int(valid.sum()) - matching

            c1, c2, c3 = st.columns(3)
            c1.metric("AGREEMENT RATE",  f"{agreement*100:.1f}%")
            c2.metric("MATCHING",        matching)
            c3.metric("MISMATCHING",     mismatch)

            if agreement >= 0.90:
                st.markdown("""
                <div style='padding: 1rem; border: 1px solid #ff2020;
                            border-left: 4px solid #ff2020; margin: 1rem 0;
                            font-family: Share Tech Mono, monospace; font-size: 12px;
                            color: #ff2020; letter-spacing: 2px;
                            animation: fadeIn 0.5s ease;'>
                    ● THREAT LEVEL: CRITICAL<br>
                    <span style='color: #444; font-size: 11px;'>
                    HEIST SUCCESSFUL — COPYCAT MIRRORS ORIGINAL WITH HIGH FIDELITY
                    </span>
                </div>
                """, unsafe_allow_html=True)
            elif agreement >= 0.75:
                st.markdown("""
                <div style='padding: 1rem; border: 1px solid #ff8c00;
                            border-left: 4px solid #ff8c00; margin: 1rem 0;
                            font-family: Share Tech Mono, monospace; font-size: 12px;
                            color: #ff8c00; letter-spacing: 2px;'>
                    ● THREAT LEVEL: ELEVATED<br>
                    <span style='color: #444; font-size: 11px;'>
                    PARTIAL SUCCESS — ADDITIONAL PROBING RECOMMENDED
                    </span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='padding: 1rem; border: 1px solid #333;
                            border-left: 4px solid #444; margin: 1rem 0;
                            font-family: Share Tech Mono, monospace; font-size: 12px;
                            color: #444; letter-spacing: 2px;'>
                    ● THREAT LEVEL: LOW<br>
                    <span style='font-size: 11px;'>
                    INSUFFICIENT DATA — INCREASE PROBE COUNT
                    </span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""
            <div style='font-family: Share Tech Mono, monospace; font-size: 11px;
                        color: #333; letter-spacing: 3px; margin: 1.5rem 0 0.5rem;'>
                // SAMPLE PREDICTION LOG
            </div>
            """, unsafe_allow_html=True)

            sample = pd.DataFrame({
                "ORIGINAL": ["REAL" if p==0 else "FAKE" for p in vpreds[valid][:20]],
                "COPYCAT":  ["REAL" if p==0 else "FAKE" for p in cpreds[:20]],
                "STATUS":   ["✓ MATCH" if v==c else "✗ MISMATCH"
                             for v,c in zip(vpreds[valid][:20], cpreds[:20])]
            })
            st.dataframe(sample, use_container_width=True)