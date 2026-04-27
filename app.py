import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import FinanceDataReader as fdr
from datetime import datetime, timedelta

# ==========================================
# 1. 페이지 설정 및 제목 (모바일 최적화)
# ==========================================
st.set_page_config(page_title="StockMap", layout="wide")

# 모바일 환경에서 상단 여백 조정 및 다크모드 대응 CSS 수정
st.markdown("""
    <style>
    .reportview-container .main .block-container { padding-top: 1rem; }
    /* 다크/라이트 테마 모두에서 가독성을 높이도록 반투명 배경과 테두리 적용 */
    [data-testid="stMetric"] { background-color: rgba(128, 128, 128, 0.1); padding: 10px; border-radius: 10px; border: 1px solid rgba(128, 128, 128, 0.2); }
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
            return f"{name} ({query})", query, query
    matched = krx_df[krx_df['Name'] == query]
    if not matched.empty:
        code = matched.iloc[0]['Code']
        return f"{query} ({code})", code, query
    return f"{query} (해외/기타)", query, query

def calculate_indicators(df):
    close = df['Close'].squeeze()
    high = df['High'].squeeze()
    low = df['Low'].squeeze()
    vol = df['Volume'].squeeze()

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

    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    
    direction = (delta > 0).astype(int) - (delta < 0).astype(int)
    df['OBV'] = (vol * direction).cumsum()
    
    df['Vol_MA5'] = vol.rolling(window=5).mean()
    df['Vol_Ratio'] = (vol / df['Vol_MA5']) * 100
    
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
        
    if not pd.isna(latest['OBV']) and latest['OBV'] > df['OBV'].iloc[-5]: 
        score += 30
        
    if not pd.isna(latest['Vol_Ratio']):
        if latest['Vol_Ratio'] >= 150 and latest['Close'] > prev['Close']:
            score += 20
            
    return score

def detect_patterns_and_levels(df):
    if len(df) < 60: return [], 0, 0
    latest = df.iloc[-1]
    patterns = []
    body = abs(latest['Open'] - latest['Close'])
    lower_shadow = min(latest['Open'], latest['Close']) - latest['Low']
    upper_shadow = latest['High'] - max(latest['Open'], latest['Close'])
    
    if lower_shadow > body * 2 and upper_shadow < body:
        patterns.append("🔨 망치형 (바닥권 반등 신호)")
    if latest['Close'] > latest['Open'] and latest['Close'] > df['High'].iloc[-2]:
        patterns.append("🚀 상승 장악형 (추세 전환)")
        
    support = df['Low'].tail(60).min()
    resistance = df['High'].tail(60).max()
    return patterns, support, resistance

@st.cache_data(ttl=60)
def get_stock_data(code, mode):
    days = 730 if "장기" in mode else 365
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    try:
        df = fdr.DataReader(code, start=start_date)
        if df.empty: return pd.DataFrame()
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        return calculate_indicators(df)
    except: return pd.DataFrame()

# 수치를 포함한 상세 분석 로직
def generate_signal_and_comments(df, mode):
    latest = df.iloc[-1]
    close = float(latest['Close'])
    rsi = float(latest['RSI'])
    macd = float(latest['MACD'])
    signal = float(latest['Signal'])
    vol_ratio = float(latest['Vol_Ratio'])
    atr = float(latest['ATR'])
    obv = float(latest['OBV'])
    prev_obv = float(df['OBV'].iloc[-5])

    comments = {}
    
    # RSI 상세 분석
    if pd.isna(rsi): comments['RSI'] = "데이터 부족"
    elif rsi >= 70: comments['RSI'] = f"🔥 **과매수 (RSI: {rsi:.1f})**: 지수가 70을 넘어 단기 과열권입니다. 추격 매수보다는 수익 실현을 고려할 시점입니다."
    elif rsi <= 30: comments['RSI'] = f"❄️ **과매도 (RSI: {rsi:.1f})**: 지수가 30 이하로 공포 구간입니다. 기술적 반등 가능성이 매우 높습니다."
    else: comments['RSI'] = f"📈 **정상 범위 (RSI: {rsi:.1f})**: 매수/매도세가 균형을 이루고 있습니다."

    # MACD 상세 분석
    macd_diff = macd - signal
    if macd > signal:
        comments['MACD'] = f"🚀 **상승 추세 (차이: {macd_diff:.2f})**: MACD가 시그널선을 상향 돌파하여 긍정적인 흐름을 유지하고 있습니다."
    else:
        comments['MACD'] = f"⚠️ **하락 추세 (차이: {macd_diff:.2f})**: MACD가 시그널선 아래에 위치하여 단기 조정 압력이 존재합니다."

    # 거래량 및 OBV 상세 분석
    obv_status = "상승" if obv > prev_obv else "하락"
    comments['VOL'] = f"🌋 **상대 거래량 ({vol_ratio:.0f}%)**: 평소 거래량 대비 유의미한 에너지가 포착되었습니다." if vol_ratio > 150 else f"➖ **보통 거래량 ({vol_ratio:.0f}%)**: 평이한 수준의 거래가 이뤄지고 있습니다."
    comments['OBV'] = f"🕵️‍♂️ **매집 확인**: 최근 5일간 누적 OBV가 {obv_status}하며 자금이 유입되는 흐름입니다."

    # 변동성 상세 분석
    volatility_pct = (atr / close) * 100
    comments['ATR'] = f"현재 주가는 일평균 **{volatility_pct:.1f}% ({int(atr)}원)** 정도의 변동폭을 보이며 움직이고 있습니다."

    # 매매 전략 결정
    position, reason = "⚖️ 관망", "주요 지표의 방향성이 엇갈리고 있습니다."
    t_buy, t_sell, s_loss = close * 0.95, close * 1.05, close * 0.90
    
    if rsi < 40 and macd > signal:
        position, reason = "🔴 적극 매수", f"RSI({rsi:.1f}) 저평가 구간에서 MACD 골든크로스가 발생했습니다."
    elif rsi > 70 and macd < signal:
        position, reason = "🔷 적극 매도", f"RSI({rsi:.1f}) 과열권에서 추세가 꺾이기 시작했습니다."

    return position, t_buy, t_sell, s_loss, reason, rsi, atr, comments

# ==========================================
# 3. 사이드바 및 실행 UI
# ==========================================
with st.sidebar:
    st.header("⚙️ 분석 설정")
    analyze_mode = st.radio("투자 성향", ["단기/스윙 (6개월)", "장기 투자 (2년)"])
    new_search = st.text_input("종목명/코드 입력", placeholder="삼성전자, 005930 등")
    target_query = new_search if st.button("🚀 분석 실행", type="primary", use_container_width=True) else None
    st.divider()
    for item in st.session_state.recent_searches:
        if st.button(f"▪️ {item['display_name']}", use_container_width=True): target_query = item['query']

if target_query:
    display_name, ticker_symbol, raw_query = parse_query(target_query)
    if {'query': raw_query, 'display_name': display_name} not in st.session_state.recent_searches:
        st.session_state.recent_searches.insert(0, {'query': raw_query, 'display_name': display_name})
        st.session_state.recent_searches = st.session_state.recent_searches[:5]

    with st.spinner(f"📡 '{display_name}' 퀀트 분석 중..."):
        df = get_stock_data(ticker_symbol, analyze_mode)
        
    if df.empty:
        st.error("데이터를 찾을 수 없습니다.")
    else:
        cur_price = df['Close'].iloc[-1]
        diff = cur_price - df['Close'].iloc[-2]
        currency = "원"
        
        st.subheader(f"📑 {display_name} 리포트")
        st.metric("현재 주가", f"{cur_price:,.0f} {currency}", f"{diff:,.0f} {currency}")

        q_score = calculate_quant_score(df)
        st.write(f"### 💯 퀀트 스코어: **{q_score}점**")
        st.progress(q_score / 100)

        pos, buy, sell, stop, reason, rsi, atr, comments = generate_signal_and_comments(df, analyze_mode)
        pts, sup, res = detect_patterns_and_levels(df)
        
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown("### 🎯 **종합 매매 타이밍**")
                st.warning(f"**포지션: {pos}**\n\n**상세의견:** {reason}")
                st.write(f"진입가: {buy:,.0f} | 목표가: {sell:,.0f}\n\n손절가: {stop:,.0f}")
                
        with col2:
            with st.container(border=True):
                st.markdown("### 🔍 **차트 패턴 및 지지/저항**")
                p_text = ", ".join(patterns) if (patterns := pts) else "특이 패턴 없음"
                st.write(f"📍 **패턴:** {p_text}")
                st.write(f"🛡️ **지지:** {sup:,.0f} | 🚧 **저항:** {res:,.0f}")

        with st.expander("🔬 지표별 상세 수치 분석", expanded=True):
            # 🌟 아이콘을 제거하고 지표 이름을 버튼화하여 클릭 시 설명 표시
            c1, c2 = st.columns([0.2, 0.8])
            with c1.popover("상대 거래량", use_container_width=True):
                st.info("**상대 거래량(Relative Volume)**\n\n최근 5일 평균 거래량 대비 오늘 거래량이 얼마나 터졌는지를 나타냅니다. 150~200% 이상이면 세력 유입이나 강한 추세 변화의 신호로 봅니다.")
            c2.markdown(comments.get('VOL', '데이터 없음'))
            
            c1, c2 = st.columns([0.2, 0.8])
            with c1.popover("OBV 누적", use_container_width=True):
                st.info("**OBV(On-Balance Volume)**\n\n거래량은 주가에 선행한다는 원리를 이용한 지표입니다. 주가가 하락해도 OBV가 상승하면 '숨은 매집'으로 판단하며, 반대의 경우 '이탈 징후'로 봅니다.")
            c2.markdown(comments.get('OBV', '데이터 없음'))
            
            c1, c2 = st.columns([0.2, 0.8])
            with c1.popover("RSI 강도", use_container_width=True):
                st.info("**RSI(상대강도지수)**\n\n주가의 상승 압력과 하락 압력 간의 상대적 강도를 나타냅니다. 70 이상은 '과매수(거품)', 30 이하는 '과매도(저평가)' 구간으로 해석합니다.")
            c2.markdown(comments.get('RSI', '데이터 없음'))
            
            c1, c2 = st.columns([0.2, 0.8])
            with c1.popover("MACD 흐름", use_container_width=True):
                st.info("**MACD(이동평균 수렴확산)**\n\n단기 추세선과 장기 추세선이 얼마나 가까워지고 멀어지는지를 측정합니다. 골든크로스가 발생하면 상승 추세의 시작으로 봅니다.")
            c2.markdown(comments.get('MACD', '데이터 없음'))
            
            c1, c2 = st.columns([0.2, 0.8])
            with c1.popover("ATR 변동성", use_container_width=True):
                st.info("**ATR(평균 실변동폭)**\n\n일정 기간 동안 주가가 얼마나 '출렁'거렸는지 변동성을 보여줍니다. ATR이 높을수록 주가가 급등락하기 쉬우므로 위험 관리가 필요합니다.")
            c2.markdown(comments.get('ATR', '데이터 없음'))

        # 스마트폰용 핀치 줌 지원 차트
        tab1, tab2 = st.tabs(["주가 차트", "수급(OBV) 차트"])
        chart_df = df.tail(100)
        
        with tab1:
            fig = go.Figure(data=[go.Candlestick(x=chart_df.index, open=chart_df['Open'], high=chart_df['High'], low=chart_df['Low'], close=chart_df['Close'], name='주가')])
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['MA20'], name='20일선', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['MA60'], name='60일선', line=dict(color='green')))
            
            fig.update_layout(
                height=450, 
                xaxis_rangeslider_visible=False, 
                margin=dict(t=10, b=10, l=0, r=0),
                dragmode='pan', # 모바일 드래그 이동 우선
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True, config={
                'scrollZoom': True, # 핀치 줌 활성화
                'displayModeBar': False,
                'doubleClick': 'reset+autosize'
            })
            
        with tab2:
            obv_fig = go.Figure(data=[go.Scatter(x=chart_df.index, y=chart_df['OBV'], name='OBV', fill='tozeroy', line=dict(color='purple'))])
            obv_fig.update_layout(height=350, margin=dict(t=10, b=10, l=0, r=0), dragmode='pan')
            st.plotly_chart(obv_fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': False})
