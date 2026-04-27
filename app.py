import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import FinanceDataReader as fdr
from pykrx import stock
from datetime import datetime, timedelta

# ==========================================
# 1. 페이지 설정 및 제목 (모바일 최적화)
# ==========================================
st.set_page_config(page_title="StockMap", layout="wide")

st.markdown("""
    <style>
    .reportview-container .main .block-container { padding-top: 1rem; }
    [data-testid="stMetric"] { 
        background-color: rgba(128, 128, 128, 0.1); 
        padding: 10px; border-radius: 10px; 
        border: 1px solid rgba(128, 128, 128, 0.2); 
    }
    .style-box {
        padding: 12px;
        border-radius: 8px;
        margin-top: 10px;
        font-size: 0.85rem;
        line-height: 1.5;
        background-color: rgba(255, 165, 0, 0.05);
        border-left: 4px solid #FF8C00;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 StockMap")
st.markdown("---")

if 'recent_searches' not in st.session_state:
    st.session_state.recent_searches = []

# ==========================================
# 2. 데이터 처리 및 지표 계산 함수
# ==========================================

# 거시 지표 수집 (분석 의견 반영용)
@st.cache_data(ttl=3600)
def get_macro_context():
    try:
        vix_df = fdr.DataReader('^VIX').tail(2)
        vix_val = vix_df['Close'].iloc[-1]
        usd_df = fdr.DataReader('USD/KRW').tail(2)
        usd_val = usd_df['Close'].iloc[-1]
        usd_diff = usd_df['Close'].iloc[-1] - usd_df['Close'].iloc[-2]
        return vix_val, usd_val, usd_diff
    except:
        return 20.0, 1350.0, 0.0

# 실전 수급 주체 데이터 수집 (pykrx 연동)
@st.cache_data(ttl=3600)
def get_investor_data(code, currency):
    if currency != "원":
        return None, None, None
    try:
        today = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
        df = stock.get_market_trading_volume_by_investor(start, today, code)
        if df.empty: return "관망", "관망", "관망"
        
        f_net = df.loc["외국인", "순매수"] if "외국인" in df.index else 0
        i_net = df.loc["기관합계", "순매수"] if "기관합계" in df.index else 0
        r_net = df.loc["개인", "순매수"] if "개인" in df.index else 0
        
        def check(v): return "매수" if v > 0 else ("매도" if v < 0 else "관망")
        return check(f_net), check(i_net), check(r_net)
    except:
        return "관망", "관망", "관망"

@st.cache_data(ttl=86400)
def get_krx_data():
    return fdr.StockListing('KRX')

def parse_query(query):
    query = query.strip().upper()
    krx_df = get_krx_data()
    if query.isdigit() and len(query) == 6:
        matched = krx_df[krx_df['Code'] == query]
        if not matched.empty:
            return f"{matched.iloc[0]['Name']} ({query})", query, query, "원", 0
    matched = krx_df[krx_df['Name'] == query]
    if not matched.empty:
        code = matched.iloc[0]['Code']
        return f"{query} ({code})", code, query, "원", 0
    return f"{query} (해외)", query, query, "$", 2

def calculate_indicators(df):
    if df.empty or len(df) < 2: return df
    close = df['Close'].squeeze()
    df['MA20'] = close.rolling(window=20).mean()
    df['MA60'] = close.rolling(window=60).mean()
    
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-10))))
    
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    tr = pd.concat([df['High'] - df['Low'], (df['High'] - close.shift()).abs(), (df['Low'] - close.shift()).abs()], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    direction = (delta > 0).astype(int) - (delta < 0).astype(int)
    df['OBV'] = (df['Volume'] * direction).cumsum()
    df['Vol_MA5'] = df['Volume'].rolling(window=5).mean()
    df['Vol_Ratio'] = (df['Volume'] / df['Vol_MA5']) * 100
    return df

def calculate_quant_score(df, is_short_term):
    if len(df) < 5: return 0
    latest, prev = df.iloc[-1], df.iloc[-2]
    score = 0
    if is_short_term:
        if latest['RSI'] < 30: score += 25
        elif latest['RSI'] < 50: score += 15
        if latest['MACD'] > latest['Signal']: score += 25
        if latest['OBV'] > df['OBV'].iloc[-5]: score += 30
        if latest['Vol_Ratio'] >= 150 and latest['Close'] > prev['Close']: score += 20
    else:
        if latest['Close'] > latest['MA60']: score += 30
        if latest['Close'] >= df['Close'].tail(60).max() * 0.95: score += 20
        if latest['MACD'] > latest['Signal']: score += 20
        if 40 <= latest['RSI'] <= 70: score += 10
        if latest['OBV'] > df['OBV'].iloc[-5]: score += 20
    return min(score, 100)

def detect_patterns_and_levels(df):
    if len(df) < 60: return [], 0, 0
    latest = df.iloc[-1]
    support, resistance = df['Close'].tail(60).min(), df['Close'].tail(60).max()
    pts = []
    body = abs(latest['Open'] - latest['Close'])
    if (min(latest['Open'], latest['Close']) - latest['Low']) > body * 2: pts.append("🔨 망치형 (반등 신호)")
    if latest['Close'] > df['High'].iloc[-2]: pts.append("🚀 전고점 돌파")
    return pts, support, resistance

@st.cache_data(ttl=60)
def get_stock_data(code):
    start = (datetime.now() - timedelta(days=1825)).strftime('%Y-%m-%d')
    try:
        df = fdr.DataReader(code, start=start)
        if df.empty: return pd.DataFrame()
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        return df
    except: return pd.DataFrame()

def generate_detailed_opinions(df, sup, res, currency, decimals, is_short_term, time_unit, macro, investor):
    latest = df.iloc[-1]
    close, rsi, macd, signal = float(latest['Close']), float(latest['RSI']), float(latest['MACD']), float(latest['Signal'])
    vix, usd, usd_d = macro
    f_net, i_net, r_net = investor
    
    # 다이버전스 판별 (스윙 로우 기준)
    swing_lb = min(60, len(df)-1)
    past = df.iloc[-swing_lb:-1]
    m_idx = past['Close'].idxmin()
    p_close, p_rsi, p_obv = float(df.loc[m_idx, 'Close']), float(df.loc[m_idx, 'RSI']), float(df.loc[m_idx, 'OBV'])
    b_div = (close < p_close) and (latest['OBV'] > p_obv or rsi > p_rsi)

    # 매크로/수급 텍스트 합성
    m_txt = f"VIX({vix:.1f}) {'패닉 구간 주의' if vix >= 25 else '안정적'}. "
    if currency == "원": m_txt += f"환율({usd:,.1f}원) {'외인 이탈 우려' if usd_d > 5 else '우호적'}. "
    
    i_txt = ""
    if f_net == "매도" and i_net == "매도": i_txt = "현재 개인만 매수 중이며 외인/기관 쌍끌이 매도 중입니다. 반등 시 탈출을 고려하세요. "
    elif f_net == "매수" and i_net == "매수": i_txt = "외인/기관이 개인 물량을 쌍끌이 매집 중인 매우 긍정적 수급입니다. "

    pos, strat = "⚖️ 관망", "추세 확인이 필요합니다."
    if is_short_term:
        if (rsi < 40 and macd > signal) or b_div: pos, strat = "🔴 적극 매수", "바닥권 수급 개선 및 다이버전스 포착."
        elif rsi > 70 and macd < signal: pos, strat = "🔷 적극 매도", "고점 과열 및 모멘텀 둔화."
    else:
        if close > latest['MA60'] and macd > signal: pos, strat = "🔴 비중 확대", "대세 상승장 진입."
        elif close < latest['MA60'] and rsi < 35: pos, strat = "🟠 저점 매수", "역사적 저평가 구간."

    # 최종 의견 조립
    ai_op = f"🤖 **StockMap AI 진단**\n\n🌍 **[환경]** {m_txt}\n👥 **[수급]** {i_txt if i_txt else '수급 눈치보기 국면.'}\n📊 **[분석]** {time_unit}봉 기준 {'우상향' if macd > signal else '조정'} 중. "
    if b_div: ai_op += "💡 상승 다이버전스 발생! "
    
    comments = {'AI': f"{ai_op}\n\n🎯 **전략:** {strat} (포지션: **{pos}**)"}
    # 기존 지표별 코멘트 생략 (AI 의견에 통합)
    return pos, strat, comments

# ==========================================
# 3. 사이드바 및 실행 UI
# ==========================================
with st.sidebar:
    st.header("⚙️ 설정")
    analyze_mode = st.radio("투자 성향", ["단기 (6개월/일봉)", "장기 (2년/주봉)"])
    new_search = st.text_input("종목명/코드", placeholder="삼성전자, NVDA 등")
    run_btn = st.button("🚀 분석 실행", type="primary", use_container_width=True)
    
    st.markdown('<div class="style-box"><b>6개월(단기)</b> 또는 <b>2년(장기)</b> 구간을 고정 분석합니다.</div>', unsafe_allow_html=True)
    
    target_query = new_search if run_btn else None
    st.divider()
    for idx, item in enumerate(st.session_state.recent_searches):
        if st.button(f"▪️ {item['display_name']}", key=f"rs_{idx}", use_container_width=True): target_query = item['query']

# ==========================================
# 4. 분석 및 차트 출력
# ==========================================
if target_query:
    display_name, ticker, raw_q, curr, dec = parse_query(target_query)
    if {'query': raw_q, 'display_name': display_name} not in st.session_state.recent_searches:
        st.session_state.recent_searches.insert(0, {'query': raw_q, 'display_name': display_name})
        st.session_state.recent_searches = st.session_state.recent_searches[:5]

    raw_df = get_stock_data(ticker)
    if raw_df.empty:
        st.error("데이터가 없습니다.")
    else:
        is_st = "단기" in analyze_mode
        unit = "일" if is_st else "주"
        if not is_st:
            df = raw_df.resample('W').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
            df = calculate_indicators(df)
            days = 730
        else:
            df = calculate_indicators(raw_df.copy())
            days = 180

        st.subheader(f"📑 {display_name} 리포트")
        st.metric("현재가", f"{df['Close'].iloc[-1]:,.{dec}f} {curr}", f"{df['Close'].iloc[-1]-df['Close'].iloc[-2]:+,.{dec}f}")
        st.progress(calculate_quant_score(df, is_st)/100, text=f"퀀트 스코어")

        pts, sup, res = detect_patterns_and_levels(df)
        macro, investor = get_macro_context(), get_investor_data(ticker, curr)
        pos, strat, comms = generate_detailed_opinions(df, sup, res, curr, dec, is_st, unit, macro, investor)

        col1, col2 = st.columns(2)
        col1.warning(f"### 🎯 전략: {pos}\n{strat}")
        col2.info(f"### 🔍 수치\n🛡️ 지지: {sup:,.{dec}f} | 🚧 저항: {res:,.{dec}f}\n📍 패턴: {', '.join(pts) if pts else '없음'}")
        
        st.info(comms['AI'])

        # 🌟 차트: 데이터 슬라이싱 및 Y축 능동 반응
        f_start = datetime.now() - timedelta(days=days)
        plot_df = df[df.index >= f_start]
        
        if not plot_df.empty:
            c_min = plot_df[['Low','MA20','MA60']].min().min()
            c_max = plot_df[['High','MA20','MA60']].max().max()
            pad = (c_max - c_min) * 0.05
            y_range = [c_min - pad, c_max + pad]
        else: y_range = None

        tab1, tab2 = st.tabs(["📉 차트", "📊 수급(OBV)"])
        with tab1:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
            fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name='주가'), row=1, col=1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA20'], name='20선', line=dict(color='orange', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA60'], name='60선', line=dict(color='green', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['RSI'], name='RSI', line=dict(color='#00BFFF', width=1.5)), row=2, col=1)
            fig.update_layout(height=550, margin=dict(t=10,b=10,l=0,r=0), dragmode=False, hovermode='x unified', showlegend=False)
            fig.update_xaxes(range=[f_start, datetime.now()], fixedrange=True, rangeslider=dict(visible=False))
            fig.update_yaxes(range=y_range, fixedrange=True, row=1, col=1)
            fig.update_yaxes(range=[0, 100], fixedrange=True, row=2, col=1)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        with tab2:
            obv_fig = go.Figure(data=[go.Scatter(x=plot_df.index, y=plot_df['OBV'], fill='tozeroy', line=dict(color='purple'))])
            obv_fig.update_layout(height=350, margin=dict(t=10,b=10,l=0,r=0), dragmode=False)
            obv_fig.update_xaxes(range=[f_start, datetime.now()], fixedrange=True)
            st.plotly_chart(obv_fig, use_container_width=True, config={'displayModeBar': False})
else:
    st.info("👈 사이드바에서 종목을 검색하세요.")
