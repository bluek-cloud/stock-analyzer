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
    # 국내 주식 (코스피 지수 KS11과 비교)
    if query.isdigit() and len(query) == 6:
        matched = krx_df[krx_df['Code'] == query]
        if not matched.empty:
            name = matched.iloc[0]['Name']
            return f"{name} ({query})", query, query, "원", 0, "KS11"
    matched = krx_df[krx_df['Name'] == query]
    if not matched.empty:
        code = matched.iloc[0]['Code']
        return f"{query} ({code})", code, query, "원", 0, "KS11"
    # 해외 주식 (S&P 500 지수 US500과 비교)
    return f"{query} (해외)", query, query, "$", 2, "US500"

def calculate_indicators(df):
    if df.empty: return df
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

# 🌟 누락되었던 퀀트 점수 계산 함수 복구
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

# 🌟 누락되었던 패턴 감지 및 지지/저항 계산 함수 복구
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
def get_comprehensive_data(stock_code, index_code):
    start_date = (datetime.now() - timedelta(days=1825)).strftime('%Y-%m-%d')
    try:
        stock_df = fdr.DataReader(stock_code, start=start_date)
        index_df = fdr.DataReader(index_code, start=start_date)
        if not stock_df.empty and stock_df.index.tz is not None: stock_df.index = stock_df.index.tz_localize(None)
        if not index_df.empty and index_df.index.tz is not None: index_df.index = index_df.index.tz_localize(None)
        return calculate_indicators(stock_df), index_df
    except: return pd.DataFrame(), pd.DataFrame()

def generate_signal_and_comments(df, index_df, sup, res, currency, decimals):
    latest = df.iloc[-1]
    close = float(latest['Close'])
    rsi, macd, signal = float(latest['RSI']), float(latest['MACD']), float(latest['Signal'])
    vol_ratio, atr, obv = float(latest['Vol_Ratio']), float(latest['ATR']), float(latest['OBV'])
    
    # 주봉 추세 (빅 포레스트 분석)
    weekly_df = df['Close'].resample('W').last()
    weekly_ma = weekly_df.rolling(window=20).mean()
    is_weekly_bull = weekly_df.iloc[-1] > weekly_ma.iloc[-1] if not pd.isna(weekly_ma.iloc[-1]) else True

    # 시장 상대 강도
    stock_ret = (df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1) * 100 if len(df) >= 20 else 0
    index_ret = (index_df['Close'].iloc[-1] / index_df['Close'].iloc[-20] - 1) * 100 if len(index_df) >= 20 else 0
    relative_strength = stock_ret - index_ret

    comments = {}
    comments['RSI'] = f"RSI: {rsi:.1f} (안정)" if 30 < rsi < 70 else (f"🔥 과매수 ({rsi:.1f})" if rsi >= 70 else f"❄️ 과매도 ({rsi:.1f})")
    comments['MACD'] = "🚀 상승 추세" if macd > signal else "⚠️ 하락 추세"
    comments['VOL'] = f"🌋 상대 거래량 {vol_ratio:.0f}%" if vol_ratio > 150 else f"➖ 보통 거래량 {vol_ratio:.0f}%"
    comments['ATR'] = f"일평균 변동폭: {atr:,.{decimals}f}{currency}"
    
    prev_obv = float(df['OBV'].iloc[-5]) if len(df) >= 5 else obv
    obv_status = "상승" if obv > prev_obv else "하락"
    comments['OBV'] = f"🕵️‍♂️ 매집 확인: 최근 5일간 누적 OBV가 {obv_status}하며 자금 흐름을 보여줍니다."

    # 전략 결정 논리
    position, reason = "⚖️ 관망", "시장 지수와 개별 지표의 신호가 엇갈려 확인이 필요한 시점입니다."
    if is_weekly_bull:
        if macd > signal or relative_strength > 2:
            position, reason = "🔴 적극 매수", "대세 상승장(주봉) 속에서 시장 대비 강력한 수급이 확인되는 '주도주' 패턴입니다."
        else:
            position, reason = "🟠 분할 매수", "큰 흐름은 우상향이나 단기적으로 숨을 고르는 건강한 조정 구간입니다."
    else:
        if macd < signal or relative_strength < -2:
            position, reason = "🔷 적극 매도", "대세 하락장(주봉)에 진입했으며 시장보다 하락폭이 커 리스크 관리가 시급합니다."
        else:
            position, reason = "🔵 비중 축소", "단기 반등은 시도 중이나 큰 추세가 꺾여 있어 기술적 반등 시 매도 전략이 유효합니다."

    # AI 넥스트 레벨 리포트 구성
    ai_opinion = f"🤖 **StockMap AI 넥스트 레벨 리포트**\n\n"
    ai_opinion += f"🌲 **[빅 포레스트 분석]** 현재 이 종목의 거대한 흐름(주봉)은 **{'우상향' if is_weekly_bull else '우하향'}** 국면에 있습니다. "
    ai_opinion += f"잔파동에 흔들리기보다 큰 줄기를 따라가는 매매가 유리합니다.\n\n"
    
    ai_opinion += f"📊 **[시장 상대 강도]** 최근 1개월간 시장 지수 대비 **{relative_strength:+.1f}%**의 초과 수익률을 기록 중입니다. "
    ai_opinion += f"{'시장을 이기는 주도주' if relative_strength > 0 else '시장 대비 소외된 흐름'}을 보이고 있습니다.\n\n"
    
    if sup > 0 and abs(close - sup) / sup < 0.05: 
        ai_opinion += f"📍 **[핵심 타점]** 강력한 지지선(**{sup:,.{decimals}f}{currency}**) 부근에서 하방 경직성을 테스트 중입니다. "
    
    ai_opinion += f"\n🎯 **최종 전략:** {reason} 따라서 **{position}** 포지션을 유지하며 대응하시길 권장합니다."

    comments['AI'] = ai_opinion
    return position, reason, comments

# ==========================================
# 3. 사이드바 및 실행 UI
# ==========================================
with st.sidebar:
    st.header("⚙️ 분석 설정")
    analyze_mode = st.radio("초기 차트 뷰", ["단기 (6개월)", "장기 (2년)"])
    new_search = st.text_input("종목명/코드 입력", placeholder="삼성전자, AAPL, NVDA 등")
    target_query = new_search if st.button("🚀 분석 실행", type="primary", use_container_width=True) else None
    st.divider()
    for item in st.session_state.recent_searches:
        if st.button(f"▪️ {item['display_name']}", use_container_width=True): target_query = item['query']

if target_query:
    display_name, ticker_symbol, raw_query, currency, decimals, index_code = parse_query(target_query)
    if {'query': raw_query, 'display_name': display_name} not in st.session_state.recent_searches:
        st.session_state.recent_searches.insert(0, {'query': raw_query, 'display_name': display_name})
        st.session_state.recent_searches = st.session_state.recent_searches[:5]

    with st.spinner(f"📡 '{display_name}' 넥스트 레벨 분석 중..."):
        df, index_df = get_comprehensive_data(ticker_symbol, index_code)
        
    if df.empty:
        st.error("데이터를 찾을 수 없습니다. 종목명이나 코드를 다시 확인해주세요.")
    else:
        cur_price = df['Close'].iloc[-1]
        diff = cur_price - df['Close'].iloc[-2]
        
        st.subheader(f"📑 {display_name} 리포트")
        st.metric("현재 주가", f"{cur_price:,.{decimals}f} {currency}", f"{diff:,.{decimals}f} {currency}")

        # 🌟 누락되었던 퀀트 스코어 바 복구
        q_score = calculate_quant_score(df)
        st.write(f"### 💯 퀀트 스코어: **{q_score}점**")
        st.progress(q_score / 100)

        # 🌟 누락되었던 지지/저항 패턴 감지 함수 호출
        pts, sup, res = detect_patterns_and_levels(df)
        pos, reason, comments = generate_signal_and_comments(df, index_df, sup, res, currency, decimals)
        
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

        with st.expander("🔬 지표별 상세 분석 (용어 클릭)", expanded=True):
            for label, key, desc in [
                ("상대 거래량", "VOL", "최근 5일 평균 거래량 대비 당일의 거래 에너지를 나타냅니다."),
                ("OBV 누적", "OBV", "거래량은 주가에 선행한다는 원리를 이용한 세력 매집 지표입니다."),
                ("RSI 강도", "RSI", "주가의 과매수(70 이상) 및 과매도(30 이하) 상태를 측정합니다."),
                ("MACD 흐름", "MACD", "단기/장기 추세의 수렴과 확산을 통해 추세 반전을 포착합니다."),
                ("ATR 변동성", "ATR", "일정 기간 주가의 평균 실질 변동폭을 화폐 단위로 보여줍니다.")
            ]:
                c1, c2 = st.columns([0.2, 0.8])
                with c1.popover(label, use_container_width=True): st.info(f"**{label}**\n\n{desc}")
                c2.markdown(comments.get(key, '데이터 없음'))
            st.divider()
            st.info(comments.get('AI'))

        # 🌟 5년 탐색 핀치 줌 차트
        tab1, tab2 = st.tabs(["주가 차트 (핀치 줌)", "수급 에너지"])
        view_days = 180 if "단기" in analyze_mode else 730
        initial_start = datetime.now() - timedelta(days=view_days)
        
        with tab1:
            fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='주가')])
            fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='20일선', line=dict(color='orange', width=1)))
            fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], name='60일선', line=dict(color='green', width=1)))
            fig.update_layout(height=450, margin=dict(t=10, b=10, l=0, r=0), dragmode='pan', hovermode='x unified', xaxis=dict(range=[initial_start, datetime.now()], rangeslider=dict(visible=False)))
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': False, 'doubleClick': 'reset+autosize'})
        with tab2:
            obv_fig = go.Figure(data=[go.Scatter(x=df.index, y=df['OBV'], name='OBV', fill='tozeroy', line=dict(color='purple'))])
            obv_fig.update_layout(height=350, margin=dict(t=10, b=10, l=0, r=0), dragmode='pan', xaxis=dict(range=[initial_start, datetime.now()]))
            st.plotly_chart(obv_fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': False})
