import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots # 🌟 RSI 하단 배치를 위한 서브플롯 임포트
import FinanceDataReader as fdr
from datetime import datetime, timedelta
import random

# ==========================================
# 1. 페이지 설정 및 제목 (상태 관리 초기화)
# ==========================================
st.set_page_config(page_title="StockMap", layout="wide")

if 'recent_searches' not in st.session_state:
    st.session_state.recent_searches = []
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = [
        {'name': '삼성전자', 'code': '005930'},
        {'name': 'APPLE', 'code': 'AAPL'}
    ]

# 커스텀 스타일 (매크로 패널 및 사이드바 최적화)
st.markdown("""
    <style>
    .reportview-container .main .block-container { padding-top: 1rem; }
    [data-testid="stMetric"] { 
        background-color: rgba(128, 128, 128, 0.1); 
        padding: 10px; border-radius: 10px; 
        border: 1px solid rgba(128, 128, 128, 0.2); 
    }
    .macro-panel {
        padding: 15px;
        border-radius: 10px;
        background-color: rgba(0, 150, 255, 0.05);
        border: 1px solid rgba(0, 150, 255, 0.1);
        margin-bottom: 20px;
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

# ==========================================
# 2. 글로벌 매크로 패널
# ==========================================
@st.cache_data(ttl=3600)
def get_macro_data():
    try:
        kospi = fdr.DataReader('KS11').tail(2)
        sp500 = fdr.DataReader('US500').tail(2)
        usdkrw = fdr.DataReader('USD/KRW').tail(2)
        vix = fdr.DataReader('^VIX').tail(2)
        return kospi, sp500, usdkrw, vix
    except:
        return None, None, None, None

st.title("📈 StockMap Dashboard")

m_kospi, m_sp500, m_usd, m_vix = get_macro_data()

if m_kospi is not None:
    with st.container():
        st.markdown('<div class="macro-panel">', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        
        def get_delta(df):
            val = df['Close'].iloc[-1]
            diff = val - df['Close'].iloc[-2]
            return val, diff

        v, d = get_delta(m_kospi)
        c1.metric("KOSPI", f"{v:,.2f}", f"{d:+.2f}")
        v, d = get_delta(m_sp500)
        c2.metric("S&P 500", f"{v:,.2f}", f"{d:+.2f}")
        v, d = get_delta(m_usd)
        c3.metric("USD/KRW", f"{v:,.1f}원", f"{d:+.1f}")
        v, d = get_delta(m_vix)
        c4.metric("VIX (공포지수)", f"{v:,.2f}", f"{d:+.2f}", delta_color="inverse")
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# 3. 데이터 처리 로직
# ==========================================
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

# 🌟 수정: RSI 기간 매개변수를 동적으로 받을 수 있도록 추가
def calculate_indicators(df, rsi_period=14):
    if df.empty or len(df) < 2: return df
    close = df['Close'].squeeze()
    df['MA20'] = close.rolling(window=20).mean()
    df['MA60'] = close.rolling(window=60).mean()
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
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

def calculate_quant_score(df):
    if len(df) < 5: return 0
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0
    if not pd.isna(latest['RSI']):
        if latest['RSI'] < 30: score += 25
        elif latest['RSI'] < 50: score += 15
        elif latest['RSI'] < 70: score += 5
    if not pd.isna(latest['MACD']) and not pd.isna(latest['Signal']):
        if latest['MACD'] > latest['Signal']: score += 25
    if not pd.isna(latest['OBV']) and latest['OBV'] > df['OBV'].iloc[-5]: score += 30
    if not pd.isna(latest['Vol_Ratio']):
        if latest['Vol_Ratio'] >= 150 and latest['Close'] > prev['Close']: score += 20
    return score

def detect_patterns_and_levels(df):
    if len(df) < 60: return [], 0, 0
    latest = df.iloc[-1]
    support = df['Low'].tail(60).min()
    resistance = df['High'].tail(60).max()
    pts = []
    if (latest['High']-max(latest['Open'],latest['Close'])) > abs(latest['Open']-latest['Close'])*2: pts.append("🔨 유성침형 (고점 경고)")
    if latest['Close'] > df['High'].iloc[-2]: pts.append("🚀 전고점 돌파")
    return pts, support, resistance

@st.cache_data(ttl=60)
def get_stock_data(code):
    try:
        df = fdr.DataReader(code, start=(datetime.now() - timedelta(days=1825)).strftime('%Y-%m-%d'))
        if df.empty: return pd.DataFrame()
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        return df
    except: return pd.DataFrame()

def generate_detailed_opinions(df, sup, res, currency, decimals, is_short_term, time_unit):
    latest = df.iloc[-1]
    close = float(latest['Close'])
    rsi = float(latest['RSI']) if not pd.isna(latest['RSI']) else 50
    macd = float(latest['MACD']) if not pd.isna(latest['MACD']) else 0
    signal = float(latest['Signal']) if not pd.isna(latest['Signal']) else 0
    vol_ratio = float(latest['Vol_Ratio']) if not pd.isna(latest['Vol_Ratio']) else 100
    atr = float(latest['ATR']) if not pd.isna(latest['ATR']) else 0
    obv = float(latest['OBV']) if not pd.isna(latest['OBV']) else 0
    
    prev_lookback = min(20, len(df) - 1) if len(df) > 1 else 1
    prev_close, prev_obv = float(df['Close'].iloc[-prev_lookback]), float(df['OBV'].iloc[-prev_lookback])
    prev_rsi = float(df['RSI'].iloc[-prev_lookback]) if not pd.isna(df['RSI'].iloc[-prev_lookback]) else 50

    comments = {}
    rsi_status = "과매수" if rsi >= 70 else ("과매도" if rsi <= 30 else "안정")
    comments['RSI'] = f"RSI **{rsi:.1f} ({rsi_status})**: " + ("단기 고점권 매물 주의" if rsi >= 70 else "바닥권 기술적 반등 기대" if rsi <= 30 else "안정적 추세 탐색 중")
    comments['MACD'] = f"MACD **{'상승' if macd > signal else '하락'}**: " + ("상승 탄력 유지 중" if macd > signal else "단기 하락 압력 우세")
    comments['VOL'] = f"거래량 **{vol_ratio:.0f}%**: " + ("에너지 분출 중" if vol_ratio > 150 else "평이한 흐름")
    comments['OBV'] = f"수급 **{'개선' if obv > prev_obv else '이탈'}**: " + ("세력 매집 징후 포착" if obv > prev_obv else "자금 이탈 주의")
    comments['ATR'] = f"변동폭 **{atr:,.{decimals}f}{currency}**: " + f"현재 주가의 {time_unit}당 평균 흔들림"

    dist_to_sup = (close - sup) / sup * 100 if sup > 0 else 100
    near_sup = dist_to_sup <= 5
    bullish_div = (close < prev_close) and (obv > prev_obv or rsi > prev_rsi)
    
    if is_short_term:
        position, strategy = "⚖️ 단기 관망", "추세 확인이 필요합니다."
        if (rsi < 40 and macd > signal) or (near_sup and bullish_div): position, strategy = "🔴 단기 적극 매수", "바닥권 수급 개선 신호 포착"
        elif (rsi > 70 and macd < signal): position, strategy = "🔷 단기 적극 매도", "고점 과열 해소 국면 진입"
    else:
        ma60 = latest['MA60'] if not pd.isna(latest['MA60']) else close
        position, strategy = "⚖️ 장기 관망", "대세 전환 에너지를 대기 중입니다."
        if close > ma60 and macd > signal: position, strategy = "🔴 비중 확대 (장기)", "주봉상 대세 상승장 진입"
        elif close < ma60 and rsi < 35: position, strategy = "🟠 저점 매수 (장기)", "역사적 저평가 구간 진입"

    mode_str = "단기 스윙" if is_short_term else "장기 가치투자"
    ai_opinion = f"🤖 **AI {mode_str} 진단:** 현재 **{'상승 추세' if macd > signal else '조정 국면'}**에 있으며, "
    if near_sup: ai_opinion += f"지지 가격(**{sup:,.{decimals}f}{currency}**) 부근의 방어력이 견고합니다. "
    comments['AI'] = f"{ai_opinion}\n\n🎯 **최종 전략:** {strategy} (포지션: **{position}**)"
    
    return position, strategy, comments

# ==========================================
# 4. 사이드바 (설정 및 Watchlist)
# ==========================================
with st.sidebar:
    st.header("⚙️ 분석 설정")
    analyze_mode = st.radio("투자 성향", ["단기 (6개월/일봉)", "장기 (2년/주봉)"])
    
    # 🌟 신규: RSI 기간 동적 조정
    st.subheader("💡 지표 설정")
    user_rsi_period = st.number_input("RSI 기간", min_value=5, max_value=50, value=14, step=1)
    
    st.divider()
    st.subheader("⭐ Watchlist 추가")
    add_col1, add_col2 = st.columns([0.7, 0.3])
    add_query = add_col1.text_input("추가", label_visibility="collapsed", placeholder="종목/코드")
    if add_col2.button("추가", use_container_width=True):
        if add_query:
            d_name, t_code, _, _, _ = parse_query(add_query)
            if not any(x['code'] == t_code for x in st.session_state.watchlist):
                st.session_state.watchlist.append({'name': d_name.split(' (')[0], 'code': t_code})
                st.rerun()

    st.subheader("📋 내 관심 종목")
    target_query = None
    for i, stock in enumerate(st.session_state.watchlist):
        c1, c2 = st.columns([0.8, 0.2])
        if c1.button(f"🔍 {stock['name']}", key=f"ws_{i}_{stock['code']}", use_container_width=True):
            target_query = stock['code']
        if c2.button("❌", key=f"del_{i}_{stock['code']}", use_container_width=True):
            st.session_state.watchlist.pop(i)
            st.rerun()

    st.divider()
    st.subheader("🕒 최근 검색")
    new_search = st.text_input("새로운 종목 검색", placeholder="삼성전자, NVDA 등")
    if st.button("🚀 검색 실행", type="primary", use_container_width=True):
        target_query = new_search
        
    for idx, item in enumerate(st.session_state.recent_searches):
        if st.button(f"▪️ {item['display_name']}", key=f"rs_{idx}_{item['query']}", use_container_width=True):
            target_query = item['query']

# ==========================================
# 5. 분석 실행 및 메인 화면
# ==========================================
if target_query:
    display_name, ticker_symbol, raw_query, currency, decimals = parse_query(target_query)
    if {'query': raw_query, 'display_name': display_name, 'code': ticker_symbol} not in st.session_state.recent_searches:
        st.session_state.recent_searches.insert(0, {'query': raw_query, 'display_name': display_name, 'code': ticker_symbol})
        st.session_state.recent_searches = st.session_state.recent_searches[:5]

    with st.spinner("📡 데이터 로드 및 분석 중..."):
        raw_df = get_stock_data(ticker_symbol)
        
    if raw_df.empty:
        st.error("데이터를 찾을 수 없습니다.")
    else:
        is_short_term = "단기" in analyze_mode
        time_unit = "일" if is_short_term else "주"
        
        if not is_short_term:
            chart_df = raw_df.resample('W').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
            chart_df = calculate_indicators(chart_df, rsi_period=user_rsi_period)
            default_days = 730
        else:
            chart_df = calculate_indicators(raw_df.copy(), rsi_period=user_rsi_period)
            default_days = 180

        cur_price = raw_df['Close'].iloc[-1]
        diff = cur_price - raw_df['Close'].iloc[-2] if len(raw_df) > 1 else 0
        st.subheader(f"📑 {display_name} 리포트")
        st.metric("현재 주가", f"{cur_price:,.{decimals}f} {currency}", f"{diff:,.{decimals}f} {currency}")

        q_score = calculate_quant_score(chart_df)
        st.progress(q_score / 100, text=f"퀀트 스코어: {q_score}점")

        pts, sup, res = detect_patterns_and_levels(chart_df)
        pos, strat, comments = generate_detailed_opinions(chart_df, sup, res, currency, decimals, is_short_term, time_unit)
        
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown(f"### 🎯 **종합 전략: {pos}**")
                st.info(comments.get('AI'))
        with col2:
            with st.container(border=True):
                st.markdown("### 🔍 **핵심 수치**")
                st.write(f"🛡️ **지지:** {sup:,.{decimals}f} | 🚧 **저항:** {res:,.{decimals}f}")
                st.write(f"📍 **패턴:** {', '.join(pts) if pts else '특이사항 없음'}")

        with st.expander("🔬 지표별 상세 분석", expanded=True):
            for label, key in [("상대 거래량", "VOL"), ("OBV 누적", "OBV"), ("RSI 강도", "RSI"), ("MACD 흐름", "MACD"), ("ATR 변동성", "ATR")]:
                st.write(f"▪️ **{label}:** {comments.get(key)}")

        # ==========================================
        # 🌟 능동 반응형 차트 (RSI 통합 & 변동성 줌 슬라이더)
        # ==========================================
        tab1, tab2 = st.tabs(["📉 주가 & RSI 차트", "📊 수급 에너지(OBV)"])
        
        # 신규 상장주 공백 제거를 위한 동적 시작점
        data_start_date = chart_df.index[0]
        calculated_start_date = datetime.now() - timedelta(days=default_days)
        final_start_date = max(data_start_date, calculated_start_date)
        
        with tab1:
            # 🌟 신규: Y축 변동성 확대 컨트롤 (기본 1.0, 숫자가 커질수록 줌인/확대됨)
            st.markdown("<p style='font-size:0.85em; color:gray; margin-bottom:5px;'>💡 슬라이더를 우측으로 당겨 미세한 캔들 변동성을 크게 확대하세요.</p>", unsafe_allow_html=True)
            y_zoom = st.slider("세로 변동성 확대 (Y축)", min_value=1.0, max_value=5.0, value=1.0, step=0.5, label_visibility="collapsed")
            
            # 주가와 RSI를 위아래로 붙이기 위한 서브플롯 생성
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
            
            # Row 1: 주가 및 이동평균선
            fig.add_trace(go.Candlestick(x=chart_df.index, open=chart_df['Open'], high=chart_df['High'], low=chart_df['Low'], close=chart_df['Close'], name='주가'), row=1, col=1)
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['MA20'], name=f'20{time_unit}선', line=dict(color='orange', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['MA60'], name=f'60{time_unit}선', line=dict(color='green', width=1)), row=1, col=1)
            
            # Row 2: RSI 라인 및 과매수/과매도 밴드
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['RSI'], name='RSI', line=dict(color='#00BFFF', width=1.5)), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=1, row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=1, row=2, col=1)
            fig.add_hrect(y0=30, y1=70, fillcolor="gray", opacity=0.1, line_width=0, row=2, col=1)
            
            # 🌟 Y축 능동적 범위 계산 로직
            visible_df = chart_df[chart_df.index >= final_start_date]
            if not visible_df.empty:
                c_min, c_max = visible_df['Low'].min(), visible_df['High'].max()
                c_mid, c_span = (c_max + c_min) / 2, (c_max - c_min) / 2
                y_range = [c_mid - (c_span / y_zoom) * 1.1, c_mid + (c_span / y_zoom) * 1.1]
            else:
                y_range = None

            fig.update_layout(
                height=600, # RSI 공간 확보를 위해 전체 높이 상향
                margin=dict(t=10, b=10, l=0, r=0), 
                dragmode='pan', hovermode='x unified', showlegend=False
            )
            # 모바일 좌우 스와이프를 위해 X축은 풀어두고 Y축은 고정
            fig.update_xaxes(range=[final_start_date, datetime.now()], rangeslider=dict(visible=False), fixedrange=False, row=1, col=1)
            fig.update_xaxes(rangeslider=dict(visible=False), fixedrange=False, row=2, col=1)
            fig.update_yaxes(range=y_range, fixedrange=True, row=1, col=1) # 능동 계산된 Y축 적용
            fig.update_yaxes(range=[0, 100], fixedrange=True, row=2, col=1) # RSI는 0~100 고정
            
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': False, 'displayModeBar': False})
            
        with tab2:
            if 'OBV' in chart_df.columns:
                obv_fig = go.Figure(data=[go.Scatter(x=chart_df.index, y=chart_df['OBV'], name='OBV', fill='tozeroy', line=dict(color='purple'))])
                obv_fig.update_layout(
                    height=350, margin=dict(t=10, b=10, l=0, r=0), 
                    dragmode='pan', hovermode='x unified',
                    xaxis=dict(range=[final_start_date, datetime.now()], fixedrange=False),
                    yaxis=dict(fixedrange=True)
                )
                st.plotly_chart(obv_fig, use_container_width=True, config={'scrollZoom': False, 'displayModeBar': False})
else:
    st.info("👈 사이드바에서 종목을 검색하거나 내 관심 종목을 클릭하여 분석을 시작하세요.")
