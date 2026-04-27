import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import FinanceDataReader as fdr
from datetime import datetime, timedelta
import random

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
            name = matched.iloc[0]['Name']
            return f"{name} ({query})", query, query, "원", 0
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
        return calculate_indicators(df)
    except: return pd.DataFrame()

# 🌟 수정: 단기/장기 모두 상세 수치를 포함한 정밀 분석 로직
def generate_signal_and_comments(df, sup, res, currency, decimals, mode):
    is_short_term = "단기" in mode
    # 장기 모드일 경우 주봉으로 변환하여 지표 재계산
    if not is_short_term:
        working_df = df.resample('W').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).copy()
        working_df = calculate_indicators(working_df)
        time_unit = "주"
    else:
        working_df = df.copy()
        time_unit = "일"

    latest = working_df.iloc[-1]
    close = float(latest['Close'])
    rsi = float(latest['RSI']) if not pd.isna(latest['RSI']) else 50
    macd = float(latest['MACD']) if not pd.isna(latest['MACD']) else 0
    signal = float(latest['Signal']) if not pd.isna(latest['Signal']) else 0
    vol_ratio = float(latest['Vol_Ratio']) if not pd.isna(latest['Vol_Ratio']) else 100
    atr = float(latest['ATR']) if not pd.isna(latest['ATR']) else 0
    obv = float(latest['OBV']) if not pd.isna(latest['OBV']) else 0
    
    prev_lookback = 20 if len(working_df) >= 20 else len(working_df)-1
    prev_close = float(working_df['Close'].iloc[-prev_lookback])
    prev_obv = float(working_df['OBV'].iloc[-prev_lookback])
    prev_rsi = float(working_df['RSI'].iloc[-prev_lookback])

    comments = {}
    
    # RSI 상세 분석
    if rsi >= 70: 
        comments['RSI'] = f"🔥 **과매수 (RSI: {rsi:.1f})**: 지수가 70을 초과한 강력한 과열 구간입니다. 추가 상승보다는 차익 매물 출회에 따른 단기 조정에 대비해야 합니다."
    elif rsi <= 30:
        comments['RSI'] = f"❄️ **과매도 (RSI: {rsi:.1f})**: 지수가 30 이하인 극도의 침체 구간입니다. 매도세가 소진되어 기술적 반등이 임박한 '공포에 사는 구간'입니다."
    else:
        comments['RSI'] = f"📈 **정상 범위 (RSI: {rsi:.1f})**: 중립적인 흐름입니다. 현재가 근처에서 매수/매도 공방이 치열하며 새로운 추세 형성을 대기 중입니다."

    # MACD 상세 분석
    macd_diff = macd - signal
    if macd > signal:
        comments['MACD'] = f"🚀 **상승 추세 (차이: {macd_diff:,.{decimals}f})**: MACD가 시그널선을 상향 돌파하여 유지 중입니다. 긍정적인 모멘텀이 강화되는 시점입니다."
    else:
        comments['MACD'] = f"⚠️ **하락 추세 (차이: {macd_diff:,.{decimals}f})**: MACD가 시그널선 아래에 위치합니다. 단기적인 하방 압력이 우세하여 보수적인 접근이 필요합니다."

    # 거래량 및 수급 상세 분석
    obv_change = obv - prev_obv
    comments['VOL'] = f"🌋 **상대 거래량 {vol_ratio:.0f}%**: 최근 5{time_unit} 평균 대비 유의미한 거래 에너지가 실리고 있습니다. 추세 변화의 강력한 신호입니다." if vol_ratio > 150 else f"➖ **보통 거래량 {vol_ratio:.0f}%**: 평이한 수준의 손바뀜이 일어나고 있으며, 주가는 횡보할 가능성이 큽니다."
    comments['OBV'] = f"🕵️‍♂️ **수급 흐름**: 최근 {prev_lookback}{time_unit}간 누적 OBV가 {'증가' if obv_change > 0 else '감소'} 중입니다. 이는 세력이 물량을 {'매집' if obv_change > 0 else '이탈'}하고 있다는 증거입니다."

    # ATR 상세 분석
    volatility_pct = (atr / close) * 100
    comments['ATR'] = f"현재 {time_unit}평균 변동성은 **{volatility_pct:.1f}% ({atr:,.{decimals}f}{currency})**입니다. 이 범위를 벗어나는 움직임은 추세의 시작으로 봅니다."

    # 최종 전략 논리
    dist_to_sup = (close - sup) / sup * 100 if sup > 0 else 100
    near_sup = dist_to_sup <= 5
    bullish_div = (close < prev_close) and (obv > prev_obv or rsi > prev_rsi)
    
    if is_short_term:
        position, reason = "⚖️ 단기 관망", "단기 지표의 방향성이 혼재되어 있어 명확한 추세 확인이 필요합니다."
        if (rsi < 40 and macd > signal) or (near_sup and bullish_div):
            position, reason = "🔴 단기 적극 매수", "단기 바닥권에서 수급 개선 시그널(다이버전스)이 명확히 포착되었습니다."
        elif (rsi > 70 and macd < signal):
            position, reason = "🔷 단기 적극 매도", "단기 과열 해소 과정에서 추세가 꺾이고 있어 즉각적인 리스크 관리가 필요합니다."
    else:
        ma60 = latest['MA60'] if not pd.isna(latest['MA60']) else close
        position, reason = "⚖️ 장기 관망", "거시적인 큰 추세 전환을 확신하기에는 수급 에너지가 아직 부족합니다."
        if close > ma60 and macd > signal:
            position, reason = "🔴 비중 확대 (장기)", "주봉상 대세 상승장에 진입하여 긴 호흡으로 수익을 극대화하기 최적의 구간입니다."
        elif close < ma60 and rsi < 35:
            position, reason = "🟠 저점 매수 (장기)", "주봉상 역사적 저평가 구간입니다. 단기 등락을 무시하고 분할 매집을 시작할 시점입니다."
        elif rsi > 75:
            position, reason = "🔷 비중 축소 (장기)", "주봉상 강력한 저항선 및 과열권에 도달했습니다. 일부 수익 실현을 통한 현금 확보를 권장합니다."

    ai_opinion = f"🤖 **StockMap AI {mode} 정밀 분석**\n\n"
    div_msg = "💡 **[핵심 패턴 포착]** 가격은 낮아지나 보조지표는 상승하는 '상승 다이버전스'가 발생했습니다. " if bullish_div else ""
    ai_body = f"현재 이 종목은 {time_unit}봉 기준 {'안정적인 우상향' if macd > signal else '조정 및 하향'} 흐름을 보이고 있습니다. "
    if near_sup: ai_body += f"특히 주요 지지 가격(**{sup:,.{decimals}f}{currency}**)에 인접해 있어 하방 경직성이 기대되는 위치입니다. "
    
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
    • <b>단기</b>: 일봉의 미세한 파동을 포착하여 며칠 내의 '반등 타점'을 잡습니다.<br>
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
        df = get_stock_data(ticker_symbol)
        
    if df.empty:
        st.error("데이터를 찾을 수 없습니다. 종목명을 다시 확인해주세요.")
    else:
        cur_price = df['Close'].iloc[-1]
        diff = cur_price - df['Close'].iloc[-2]
        st.subheader(f"📑 {display_name} 리포트")
        st.metric("현재 주가", f"{cur_price:,.{decimals}f} {currency}", f"{diff:,.{decimals}f} {currency}")

        q_score = calculate_quant_score(df)
        st.write(f"### 💯 퀀트 스코어: **{q_score}점**")
        st.progress(q_score / 100)

        pts, sup, res = detect_patterns_and_levels(df)
        pos, reason, comments = generate_signal_and_comments(df, sup, res, currency, decimals, analyze_mode)
        
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
                ("상대 거래량", "VOL", "평균 거래량 대비 오늘의 거래 에너지를 측정합니다."),
                ("OBV 누적", "OBV", "거래량은 주가에 선행한다는 원리의 세력 매집 지표입니다."),
                ("RSI 강도", "RSI", "주가의 과매수(70↑) 및 과매도(30↓) 상태를 나타냅니다."),
                ("MACD 흐름", "MACD", "단기/장기 추세의 교차를 통해 반전 타점을 잡습니다."),
                ("ATR 변동성", "ATR", "일정 기간 주가의 평균 실질 변동폭을 보여줍니다.")
            ]:
                c1, c2 = st.columns([0.2, 0.8])
                with c1.popover(label, use_container_width=True): st.info(f"**{label}**\n\n{desc}")
                c2.markdown(comments.get(key, '데이터 없음'))
            st.divider()
            st.info(comments.get('AI'))

        # ==========================================
        # 🌟 주봉 차트 변환 지원 핀치 줌 차트
        # ==========================================
        tab1, tab2 = st.tabs(["주가 차트 (핀치 줌)", "수급 에너지"])
        
        # 투자 성향에 따라 차트 데이터 구성 (일봉 vs 주봉)
        if "장기" in analyze_mode:
            chart_df = df.resample('W').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'})
            view_days = 730 # 초기엔 2년치 주봉
        else:
            chart_df = df
            view_days = 180 # 초기엔 6개월치 일봉
            
        initial_start = datetime.now() - timedelta(days=view_days)
        
        with tab1:
            fig = go.Figure(data=[go.Candlestick(x=chart_df.index, open=chart_df['Open'], high=chart_df['High'], low=chart_df['Low'], close=chart_df['Close'], name='주가')])
            # 이평선은 일봉 데이터 기준으로 그대로 유지하여 정밀도 확보
            fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='20일선', line=dict(color='orange', width=1), opacity=0.5))
            fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], name='60일선', line=dict(color='green', width=1), opacity=0.5))
            fig.update_layout(height=450, margin=dict(t=10, b=10, l=0, r=0), dragmode='pan', hovermode='x unified', xaxis=dict(range=[initial_start, datetime.now()], rangeslider=dict(visible=False)))
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': False, 'doubleClick': 'reset+autosize'})
        with tab2:
            obv_fig = go.Figure(data=[go.Scatter(x=chart_df.index, y=chart_df['OBV'] if 'OBV' in chart_df else df['OBV'], name='OBV', fill='tozeroy', line=dict(color='purple'))])
            obv_fig.update_layout(height=350, margin=dict(t=10, b=10, l=0, r=0), dragmode='pan', xaxis=dict(range=[initial_start, datetime.now()]))
            st.plotly_chart(obv_fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': False})
