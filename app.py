import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import FinanceDataReader as fdr
from datetime import datetime, timedelta

# ==========================================
# 1. 페이지 설정 및 제목 (모바일 및 테마 최적화)
# ==========================================
st.set_page_config(page_title="StockMap", layout="wide")

# 가독성 및 디자인 개선을 위한 CSS
st.markdown("""
    <style>
    /* 메트릭 및 컨테이너 가독성 개선 */
    [data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.1);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    .stExpander { border: none !important; box-shadow: none !important; }
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
    # 데이터가 비어있는 경우 방어 로직
    if df.empty: return df
    
    close = df['Close'].squeeze()
    high = df['High'].squeeze()
    low = df['Low'].squeeze()
    vol = df['Volume'].squeeze()

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

    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    
    direction = (delta > 0).astype(int) - (delta < 0).astype(int)
    df['OBV'] = (vol * direction).cumsum()
    df['Vol_MA5'] = vol.rolling(window=5).mean()
    df['Vol_Ratio'] = (vol / df['Vol_MA5']) * 100
    
    return df

@st.cache_data(ttl=60)
def get_stock_data(code):
    # 🌟 개선: 탐색을 위해 기본적으로 5년치 데이터를 불러옵니다.
    start_date = (datetime.now() - timedelta(days=1825)).strftime('%Y-%m-%d')
    try:
        df = fdr.DataReader(code, start=start_date)
        if df.empty: return pd.DataFrame()
        return calculate_indicators(df)
    except: return pd.DataFrame()

def generate_signal_and_comments(df):
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
    
    # RSI 수치 포함 상세 분석
    if rsi >= 70: comments['RSI'] = f"🔥 **과매수 (RSI: {rsi:.1f})**: 지수가 70을 초과했습니다. 단기 고점 신호일 수 있으니 주의가 필요합니다."
    elif rsi <= 30: comments['RSI'] = f"❄️ **과매도 (RSI: {rsi:.1f})**: 지수가 30 미만입니다. 과도한 낙폭에 따른 기술적 반등 구간입니다."
    else: comments['RSI'] = f"📈 **안정적 (RSI: {rsi:.1f})**: 매수/매도 세력이 균형을 이루며 추세를 탐색 중입니다."

    # MACD 상세 분석
    macd_diff = macd - signal
    comments['MACD'] = f"🚀 **상승세** (차이: {macd_diff:.2f})" if macd > signal else f"⚠️ **하락세** (차이: {macd_diff:.2f})"

    # 거래량 및 변동성
    comments['VOL'] = f"🌋 **상대 거래량 {vol_ratio:.0f}%**" if vol_ratio > 150 else f"➖ **평이한 거래량 {vol_ratio:.0f}%**"
    comments['ATR'] = f"현재 일평균 **{int(atr)}원 ({(atr/close*100):.1f}%)** 정도의 변동성을 보입니다."

    # 매매 전략
    position, reason = "⚖️ 관망", "주요 지표가 중립 상태입니다."
    if rsi < 45 and macd > signal: position, reason = "🔴 적극 매수", f"RSI({rsi:.1f}) 바닥권에서 MACD 골든크로스가 포착되었습니다."
    elif rsi > 65 and macd < signal: position, reason = "🔷 적극 매도", f"RSI({rsi:.1f}) 과열권에서 추세 하락이 감지되었습니다."

    return position, reason, comments

# ==========================================
# 3. 사이드바 및 실행 UI
# ==========================================
with st.sidebar:
    st.header("⚙️ StockMap 설정")
    analyze_mode = st.radio("초기 차트 범위", ["단기 (6개월)", "장기 (2년)"])
    new_search = st.text_input("종목명/코드 입력", placeholder="삼성전자, 005930 등")
    target_query = new_search if st.button("🚀 분석 실행", type="primary", use_container_width=True) else None
    
    if st.session_state.recent_searches:
        st.divider()
        for item in st.session_state.recent_searches:
            if st.button(f"▪️ {item['display_name']}", use_container_width=True): target_query = item['query']

if target_query:
    display_name, ticker_symbol, raw_query = parse_query(target_query)
    if {'query': raw_query, 'display_name': display_name} not in st.session_state.recent_searches:
        st.session_state.recent_searches.insert(0, {'query': raw_query, 'display_name': display_name})
        st.session_state.recent_searches = st.session_state.recent_searches[:5]

    with st.spinner(f"📡 '{display_name}' 데이터를 지도로 그리는 중..."):
        df = get_stock_data(ticker_symbol)
        
    if df.empty:
        st.error("데이터를 불러오지 못했습니다. 종목명을 정확히 입력해주세요.")
    else:
        cur_price = df['Close'].iloc[-1]
        diff = cur_price - df['Close'].iloc[-2]
        
        st.subheader(f"📑 {display_name} 리포트")
        # 🌟 다크모드에서도 잘 보이도록 수정된 메트릭
        st.metric("현재 주가", f"{cur_price:,.0f} 원", f"{diff:,.0f} 원")

        pos, reason, comments = generate_signal_and_comments(df)
        
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown("### 🎯 **종합 전략**")
                st.warning(f"**포지션: {pos}**\n\n**의견:** {reason}")
                
        with col2:
            with st.container(border=True):
                st.markdown("### 🔍 **지표 분석 요약**")
                st.write(f"**RSI:** {comments['RSI']}")
                st.write(f"**MACD:** {comments['MACD']}")

        with st.expander("🔬 기술적 지표 상세 데이터 보기", expanded=True):
            st.write(f"**[상대 거래량]** {comments['VOL']}")
            st.write(f"**[변동성(ATR)]** {comments['ATR']}")

        # 🌟 차트 기간 확장 및 초기 뷰포트 설정
        tab1, tab2 = st.tabs(["주가 맵 (확대/축소 가능)", "수급 에너지"])
        
        # 초기 보여줄 날짜 계산
        view_days = 180 if "단기" in analyze_mode else 730
        initial_start_date = (datetime.now() - timedelta(days=view_days))
        
        with tab1:
            fig = go.Figure(data=[go.Candlestick(
                x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='주가'
            )])
            fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='20일선', line=dict(color='orange', width=1)))
            fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], name='60일선', line=dict(color='green', width=1)))
            
            fig.update_layout(
                height=500,
                xaxis=dict(
                    rangeslider=dict(visible=False),
                    range=[initial_start_date, datetime.now()] # 🌟 초기 범위는 6개월/2년이지만 전체 탐색 가능
                ),
                margin=dict(t=10, b=10, l=0, r=0),
                dragmode='pan',
                hovermode='x unified',
                template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True, config={
                'scrollZoom': True, # 🌟 핀치 줌 지원
                'displayModeBar': False
            })
            
        with tab2:
            obv_fig = go.Figure(data=[go.Scatter(x=df.index, y=df['OBV'], name='OBV', fill='tozeroy', line=dict(color='purple'))])
            obv_fig.update_layout(
                height=350, 
                xaxis=dict(range=[initial_start_date, datetime.now()]),
                margin=dict(t=10, b=10, l=0, r=0), 
                dragmode='pan',
                template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white"
            )
            st.plotly_chart(obv_fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': False})
