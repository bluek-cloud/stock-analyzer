import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import FinanceDataReader as fdr
from datetime import datetime, timedelta
import random

# ==========================================
# 1. 페이지 설정 및 제목
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
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
        font-size: 0.85rem;
        line-height: 1.4;
        background-color: rgba(255, 165, 0, 0.05);
        border-left: 3px solid orange;
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
    df['MA200'] = close.rolling(window=200).mean()
    
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
    patterns = []
    body = abs(latest['Open'] - latest['Close'])
    lower_shadow = min(latest['Open'], latest['Close']) - latest['Low']
    upper_shadow = latest['High'] - max(latest['Open'], latest['Close'])
    
    if lower_shadow > body * 2 and upper_shadow < body: patterns.append("🔨 망치형 (바닥권 반등 신호)")
    if latest['Close'] > latest['Open'] and latest['Close'] > df['High'].iloc[-2]: patterns.append("🚀 상승 장악형 (추세 전환)")
    
    # 전달받은 df의 타임프레임(일봉/주봉)에 완벽히 동기화된 지지/저항
    support = df['Low'].tail(60).min()
    resistance = df['High'].tail(60).max()
    return patterns, support, resistance

@st.cache_data(ttl=60)
def get_stock_data(code):
    start_date = (datetime.now() - timedelta(days=1825)).strftime('%Y-%m-%d')
    try:
        df = fdr.DataReader(code, start=start_date)
        if df.empty: return pd.DataFrame()
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        return df # 지표 계산은 분기 처리 이후로 이동 (최적화)
    except: return pd.DataFrame()

# 🌟 고도화: 분석용 메인 함수 (df 자체가 이미 완벽한 일봉/주봉으로 세팅되어 들어옴)
def generate_signal_and_comments(df, sup, res, currency, decimals, is_short_term, time_unit):
    latest = df.iloc[-1]
    close = float(latest['Close'])
    rsi = float(latest['RSI']) if not pd.isna(latest['RSI']) else 50
    macd = float(latest['MACD']) if not pd.isna(latest['MACD']) else 0
    signal = float(latest['Signal']) if not pd.isna(latest['Signal']) else 0
    vol_ratio = float(latest['Vol_Ratio']) if not pd.isna(latest['Vol_Ratio']) else 100
    atr = float(latest['ATR']) if not pd.isna(latest['ATR']) else 0
    obv = float(latest['OBV']) if not pd.isna(latest['OBV']) else 0
    
    # Index Error 방지를 위한 동적 룩백 설정
    prev_lookback = min(20, len(df) - 1) if len(df) > 1 else 1
    prev_close = float(df['Close'].iloc[-prev_lookback])
    prev_obv = float(df['OBV'].iloc[-prev_lookback])
    prev_rsi = float(df['RSI'].iloc[-prev_lookback]) if not pd.isna(df['RSI'].iloc[-prev_lookback]) else 50
    
    recent_obv_lookback = min(5, len(df) - 1) if len(df) > 1 else 1
    prev_recent_obv = float(df['OBV'].iloc[-recent_obv_lookback])

    comments = {}
    
    # 지표 상세 코멘트 동적 생성
    comments['RSI'] = f"RSI: {rsi:.1f} (안정)" if 30 < rsi < 70 else (f"🔥 과매수 ({rsi:.1f})" if rsi >= 70 else f"❄️ 과매도 ({rsi:.1f})")
    comments['MACD'] = "🚀 상승 추세" if macd > signal else "⚠️ 하락 추세"
    comments['VOL'] = f"🌋 상대 거래량 {vol_ratio:.0f}%" if vol_ratio > 150 else f"➖ 보통 거래량 {vol_ratio:.0f}%"
    comments['ATR'] = f"{time_unit}평균 변동폭: {atr:,.{decimals}f}{currency}"
    comments['OBV'] = f"🕵️‍♂️ 매집 확인: 최근 5{time_unit}간 누적 OBV가 {'개선' if obv > prev_recent_obv else '악화'} 중입니다."

    # 패턴 및 논리 도출
    dist_to_sup = (close - sup) / sup * 100 if sup > 0 else 100
    near_sup = dist_to_sup <= 5
    dist_to_res = (res - close) / res * 100 if res > 0 else 100
    near_res = dist_to_res <= 5

    bullish_div = (close < prev_close) and (obv > prev_obv or rsi > prev_rsi)
    bearish_div = (close > prev_close) and (obv < prev_obv or rsi < prev_rsi)

    if is_short_term:
        position, reason = "⚖️ 단기 관망", "단기 지표의 방향성이 혼재되어 있어 명확한 추세 확인이 필요합니다."
        if (rsi < 40 and macd > signal) or (near_sup and bullish_div):
            position, reason = "🔴 단기 적극 매수", "단기 바닥권에서 수급 개선 시그널(다이버전스)이 명확히 포착되었습니다."
        elif (rsi > 70 and macd < signal) or bearish_div:
            position, reason = "🔷 단기 적극 매도", "단기 고점 징후가 뚜렷하며 수급이 이탈하는 현상이 감지되었습니다."
        elif macd > signal:
            position, reason = "🟠 단기 분할 매수", "안정적인 단기 우상향 흐름이 이어지고 있습니다."
    else:
        ma60 = latest['MA60'] if not pd.isna(latest['MA60']) else close
        position, reason = "⚖️ 장기 관망", "거시적인 큰 추세 전환을 확신하기에는 에너지가 아직 부족합니다."
        if close > ma60 and macd > signal:
            position, reason = "🔴 비중 확대 (장기)", "주봉상 대세 상승장에 진입하여 긴 호흡으로 수익을 극대화하기 최적의 구간입니다."
        elif close < ma60 and rsi < 35:
            position, reason = "🟠 저점 매수 (장기)", "주봉상 역사적 저평가 구간입니다. 단기 등락을 무시하고 분할 매집을 시작할 시점입니다."
        elif rsi > 75:
            position, reason = "🔷 비중 축소 (장기)", "주봉상 강력한 저항선 및 과열권에 도달했습니다. 일부 수익 실현을 권장합니다."

    ai_opinion = f"🤖 **StockMap AI {'단기 스윙' if is_short_term else '장기 가치투자'} 정밀 분석**\n\n"
    div_msg = "💡 **[핵심 패턴 포착]** 가격은 낮아지나 보조지표는 상승하는 '상승 다이버전스'가 발생했습니다. " if bullish_div else ""
    
    ai_body = f"현재 이 종목은 {time_unit}봉 기준 {'안정적인 우상향' if macd > signal else '조정 및 하락'} 흐름을 보이고 있습니다. "
    if near_sup: ai_body += f"특히 주요 지지 가격(**{sup:,.{decimals}f}{currency}**)에 인접해 있어 하방 경직성이 기대되는 위치입니다. "
    elif near_res: ai_body += f"강력한 저항선(**{res:,.{decimals}f}{currency}**)에 근접하여 매물 돌파 여부가 중요한 시점입니다. "
    
    comments['AI'] = f"{ai_opinion}{div_msg}{ai_body}\n\n🎯 **최종 전략:** {reason} 따라서 현재는 **{position}** 포지션을 추천합니다."
    
    return position, reason, comments

# ==========================================
# 3. 사이드바 및 실행 UI
# ==========================================
with st.sidebar:
    st.header("⚙️ 분석 설정")
    analyze_mode = st.radio("투자 성향 설정", ["단기 투자 (6개월 차트/일봉)", "장기 투자 (2년 차트/주봉)"])
    new_search = st.text_input("종목명/코드 입력", placeholder="삼성전자, AAPL, NVDA 등")
    run_btn = st.button("🚀 분석 실행", type="primary", use_container_width=True)
    
    st.markdown(f"""
    <div class="style-box">
    <b>🔍 분석 모드 차이 안내</b><br>
    • <b>단기</b>: 일봉의 미세한 파동을 포착하여 며칠 내의 '단기 반등 타점'을 잡습니다.<br>
    • <b>장기</b>: <b>주봉 단위</b>의 거대 추세를 분석하여 잔파동을 무시하고 대세 흐름을 판별합니다.
    </div>
    """, unsafe_allow_html=True)
    
    target_query = new_search if run_btn else None
    st.divider()
    for item in st.session_state.recent_searches:
        if st.button(f"▪️ {item['display_name']}", use_container_width=True): target_query = item['query']

if target_query:
    display_name, ticker_symbol, raw_query, currency, decimals = parse_query(target_query)
    if {'query': raw_query, 'display_name': display_name} not in st.session_state.recent_searches:
        st.session_state.recent_searches.insert(0, {'query': raw_query, 'display_name': display_name})
        st.session_state.recent_searches = st.session_state.recent_searches[:5]

    with st.spinner(f"📡 '{display_name}' 딥다이브 분석 중..."):
        raw_df = get_stock_data(ticker_symbol)
        
    if raw_df.empty:
        st.error("데이터를 찾을 수 없습니다. 종목명을 다시 확인해주세요.")
    else:
        # 🌟 핵심 구조 최적화: 모드에 따라 뼈대(Dataframe) 자체를 일원화
        is_short_term = "단기" in analyze_mode
        time_unit = "일" if is_short_term else "주"
        
        if not is_short_term:
            chart_df = raw_df.resample('W').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
            chart_df = calculate_indicators(chart_df) # 오직 주봉 기준으로만 지표 연산
            view_days = 730
        else:
            chart_df = calculate_indicators(raw_df.copy()) # 일봉 기준 연산
            view_days = 180

        # 현재가 및 변동 금액은 가장 신선한 '일봉(raw_df)' 데이터로 유지
        cur_price = raw_df['Close'].iloc[-1]
        diff = cur_price - raw_df['Close'].iloc[-2] if len(raw_df) > 1 else 0
        
        st.subheader(f"📑 {display_name} 리포트")
        st.metric("현재 주가", f"{cur_price:,.{decimals}f} {currency}", f"{diff:,.{decimals}f} {currency}")

        # 모든 점수와 분석이 chart_df(선택된 타임프레임)에 완벽히 연동됨
        q_score = calculate_quant_score(chart_df)
        st.write(f"### 💯 퀀트 스코어: **{q_score}점**")
        st.progress(q_score / 100)

        pts, sup, res = detect_patterns_and_levels(chart_df)
        pos, reason, comments = generate_signal_and_comments(chart_df, sup, res, currency, decimals, is_short_term, time_unit)
        
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown("### 🎯 **종합 전략**")
                st.warning(f"**포지션: {pos}**\n\n**의견:** {reason}")
        with col2:
            with st.container(border=True):
                st.markdown("### 🔍 **차트 패턴 및 지지/저항**")
                p_text = ", ".join(pts) if pts else "특이 패턴 없음"
                st.write(f"📍 **패턴:** {p_text}")
                st.write(f"🛡️ **지지:** {sup:,.{decimals}f} {currency} | 🚧 **저항:** {res:,.{decimals}f} {currency}")

        with st.expander("🔬 지표별 상세 수치 분석 (용어 클릭)", expanded=True):
            for label, key, desc in [
                ("상대 거래량", "VOL", f"최근 5{time_unit} 평균 대비 현재의 거래 에너지를 측정합니다."),
                ("OBV 누적", "OBV", "거래량은 주가에 선행한다는 원리의 세력 매집 지표입니다."),
                ("RSI 강도", "RSI", "주가의 과매수(70↑) 및 과매도(30↓) 상태를 나타냅니다."),
                ("MACD 흐름", "MACD", "단기/장기 추세의 교차를 통해 반전 타점을 잡습니다."),
                ("ATR 변동성", "ATR", f"일정 기간 주가의 평균 실질 변동폭을 보여줍니다.")
            ]:
                c1, c2 = st.columns([0.2, 0.8])
                with c1.popover(label, use_container_width=True): st.info(f"**{label}**\n\n{desc}")
                c2.markdown(comments.get(key, '데이터 없음'))
            st.divider()
            st.info(comments.get('AI'))

        # ==========================================
        # 🌟 타임프레임이 완벽히 동기화된 차트 시각화
        # ==========================================
        tab1, tab2 = st.tabs(["주가 차트 (핀치 줌)", "수급 에너지"])
        initial_start = datetime.now() - timedelta(days=view_days)
        
        with tab1:
            fig = go.Figure(data=[go.Candlestick(x=chart_df.index, open=chart_df['Open'], high=chart_df['High'], low=chart_df['Low'], close=chart_df['Close'], name='주가')])
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['MA20'], name=f'20{time_unit}선', line=dict(color='orange', width=1)))
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['MA60'], name=f'60{time_unit}선', line=dict(color='green', width=1)))
            fig.update_layout(height=450, margin=dict(t=10, b=10, l=0, r=0), dragmode='pan', hovermode='x unified', xaxis=dict(range=[initial_start, datetime.now()], rangeslider=dict(visible=False)))
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': False, 'doubleClick': 'reset+autosize'})
        with tab2:
            if 'OBV' in chart_df.columns:
                obv_fig = go.Figure(data=[go.Scatter(x=chart_df.index, y=chart_df['OBV'], name='OBV', fill='tozeroy', line=dict(color='purple'))])
                obv_fig.update_layout(height=350, margin=dict(t=10, b=10, l=0, r=0), dragmode='pan', xaxis=dict(range=[initial_start, datetime.now()]))
                st.plotly_chart(obv_fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': False})
