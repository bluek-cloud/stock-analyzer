import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import FinanceDataReader as fdr

# ==========================================
# 1. 페이지 설정 및 제목
# ==========================================
st.set_page_config(page_title="매수매도분석기 v2.9", layout="wide")

st.title("📈 실시간 매수매도분석기 v2.9")
st.markdown("기술적 지표(**RSI, MACD**)와 수급 지표(**OBV, ATR**)에 대한 **상세 코멘트 분석**을 제공합니다.")
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
            name, market = matched.iloc[0]['Name'], matched.iloc[0]['Market']
            return f"{name} ({query})", f"{query}{'.KS' if market in ['KOSPI', 'KOSPI200'] else '.KQ'}", query
    matched = krx_df[krx_df['Name'] == query]
    if not matched.empty:
        code, market = matched.iloc[0]['Code'], matched.iloc[0]['Market']
        return f"{query} ({code})", f"{code}{'.KS' if market in ['KOSPI', 'KOSPI200'] else '.KQ'}", query
    return f"{query} (해외/기타)", query, query

def calculate_indicators(df):
    close = df['Close'].squeeze()
    high = df['High'].squeeze()
    low = df['Low'].squeeze()
    vol = df['Volume'].squeeze()

    # 공통 지표
    df['MA20'] = close.rolling(window=20).mean()
    df['MA60'] = close.rolling(window=60).mean()

    # 장기 모드용 지표 (200일선, 볼린저밴드)
    df['MA200'] = close.rolling(window=200).mean()
    df['BB_Mid'] = close.rolling(window=20).mean()
    df['BB_Std'] = close.rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)

    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-10))))

    # MACD
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # ATR
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()

    # OBV
    df['OBV'] = (vol * (close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)))).cumsum()
    
    return df

@st.cache_data(ttl=60)
def get_stock_data(ticker, mode):
    period = "2y" if "장기" in mode else "1y"
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df.empty: return pd.DataFrame()
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        return calculate_indicators(df)
    except: return pd.DataFrame()

def generate_signal_and_comments(df, mode):
    latest, prev = df.iloc[-1], df.iloc[-2]
    close, rsi = float(latest['Close']), float(latest['RSI'])
    macd_curr, sig_curr = float(latest['MACD']), float(latest['Signal'])
    macd_prev, sig_prev = float(prev['MACD']), float(prev['Signal'])
    atr, obv = float(latest['ATR']), float(latest['OBV'])
    
    ma20 = float(latest['MA20'])
    ma60 = float(latest['MA60'])
    ma200 = float(latest['MA200']) if not pd.isna(latest['MA200']) else close
    bb_lower = float(latest['BB_Lower'])
    bb_upper = float(latest['BB_Upper'])

    # 수급 다이버전스를 정확히 보기 위해 20일(약 1달) 기준 데이터 활용
    prev20_close = float(df['Close'].iloc[-20]) if len(df) >= 20 else float(df['Close'].iloc[0])
    prev20_obv = float(df['OBV'].iloc[-20]) if len(df) >= 20 else float(df['OBV'].iloc[0])

    # ==========================================
    # 🎯 지표별 실전 심층 분석 로직
    # ==========================================
    comments = {}
    
    # 1. RSI (맹목적 매수/매도 방지)
    if rsi >= 70: 
        comments['RSI'] = "🔥 **과매수 (Overbought)**: 주가가 단기적으로 과열되었습니다. 상승 여력은 있으나 차익 실현 매물이 쏟아질 위험 구간이므로 신규 진입은 자제하세요."
    elif 55 <= rsi < 70: 
        comments['RSI'] = "📈 **상승 모멘텀**: 매수세가 시장을 주도하며 안정적인 상승 탄력을 받고 있습니다."
    elif 45 <= rsi < 55: 
        comments['RSI'] = "➖ **방향성 탐색**: 매수와 매도 세력이 팽팽하게 맞서며 뚜렷한 방향성이 없는 중립 구간입니다."
    elif 30 < rsi < 45: 
        comments['RSI'] = "📉 **약세 국면**: 매도세가 압도하고 있습니다. 섣부른 물타기보다는 바닥이 확인될 때까지 기다리세요."
    else: 
        comments['RSI'] = "❄️ **과매도 (Oversold)**: 시장이 과도한 공포에 빠졌습니다. 기술적 반등 위치지만, '떨어지는 칼날'일 수 있으므로 추세 반전을 꼭 확인하고 진입하세요."

    # 2. MACD (0선 기준을 적용한 진짜 추세 분석)
    if macd_curr > sig_curr:
        if macd_curr > 0:
            comments['MACD'] = "🚀 **강세장 상승 추세**: MACD가 0선 위에서 시그널 선을 상회합니다. 대세 상승 국면으로, 주가 상승폭이 커질 수 있습니다."
        else:
            comments['MACD'] = "🌱 **약세장 단기 반등**: 하락 추세(0선 아래) 중 골든크로스가 발생했습니다. 바닥을 다지고 기술적 반등을 시도하는 긍정적 시그널입니다."
    else:
        if macd_curr > 0:
            comments['MACD'] = "⚠️ **강세장 단기 조정**: 0선 위에서 데드크로스가 발생했습니다. 상승 흐름 중 잠시 쉬어가는 '눌림목'이거나 단기 고점일 수 있습니다."
        else:
            comments['MACD'] = "🔻 **본격 하락 추세**: MACD가 0선 아래로 꺾였습니다. 하락 압력이 매우 강하므로 리스크 관리가 최우선입니다."

    # 3. OBV (20일 기준 세력 매집/이탈 다이버전스)
    price_change = (close - prev20_close) / prev20_close * 100
    obv_change = obv - prev20_obv
    
    if price_change <= 0 and obv_change > 0: 
        comments['OBV'] = "🕵️‍♂️ **숨은 매집 (다이버전스)**: 최근 20일간 주가는 횡보/하락했지만 누적 수급(OBV)은 증가했습니다! 스마트머니(세력)의 저가 매집이 강하게 의심되는 상승 전조입니다."
    elif price_change > 0 and obv_change < 0: 
        comments['OBV'] = "🚨 **이탈 징후 (다이버전스)**: 주가는 올랐지만 거래량은 빠져나가고 있습니다. 상승 동력이 소진된 '가짜 상승'일 확률이 높으니 추격 매수를 주의하세요."
    elif price_change > 0 and obv_change > 0: 
        comments['OBV'] = "💪 **건강한 상승**: 주가 상승과 함께 매수 거래량도 탄탄하게 유입되며 추세를 잘 뒷받침하고 있습니다."
    else: 
        comments['OBV'] = "🍂 **수급 악화**: 매도 거래량이 압도하며 자금이 지속적으로 이탈하고 있습니다."

    # 4. ATR (변동성 퍼센티지 안내)
    volatility_pct = (atr / close) * 100
    if volatility_pct > 5.0: 
        comments['ATR'] = f"🌪️ **초고변동성 (일평균 {volatility_pct:.1f}%)**: 주가가 위아래로 거칠게 움직입니다. 평소보다 비중을 줄이고 손절선을 넉넉히 잡아야 털리지 않습니다."
    elif volatility_pct < 1.5: 
        comments['ATR'] = f"🐢 **저변동성 (일평균 {volatility_pct:.1f}%)**: 주가 움직임이 둔화된 지루한 횡보장입니다. 돌파가 나오기 전까지는 수익을 내기 어렵습니다."
    else: 
        comments['ATR'] = f"🌊 **일반적 수준 (일평균 {volatility_pct:.1f}%)**: 안정적이고 정상적인 주가 등락폭을 유지하고 있습니다."

    # 장기 모드 전용 코멘트
    if "장기" in mode:
        trend_status = "우상향 정배열" if close > ma200 else "하락 역배열"
        comments['LONG'] = f"**[200일선 & 볼린저밴드]** 주가가 장기 생명선인 200일선 {'**위**' if close > ma200 else '**아래**'}에 있어 장기 **{trend_status}** 흐름입니다."
        if close <= bb_lower * 1.02: comments['LONG'] += " 현재 볼린저 밴드 하단을 강하게 터치하여 역사적 저평가 매수 기회로 볼 수 있습니다."
        elif close >= bb_upper * 0.98: comments['LONG'] += " 현재 볼린저 밴드 상단에 위치하여 가격 부담이 큰 과열 상태입니다."


    # ==========================================
    # 🎯 종합 매매 시그널 로직 (복합 조건식 적용)
    # ==========================================
    position, reason = "⚖️ 관망 (Neutral)", "뚜렷한 추세 전환이 확인되지 않았습니다. 방향성이 정해질 때까지 관망하세요."

    if "단기" in mode:
        # 단기 복합 로직
        if rsi < 40 and macd_curr > sig_curr and macd_prev <= sig_prev:
            position, reason = "🔴 적극 매수 (Strong Buy)", "과매도 부근에서 MACD 골든크로스가 발생했습니다. 확률 높은 단기 반등 타점입니다."
        elif close > ma20 and macd_curr > sig_curr and rsi < 70:
            position, reason = "🟠 분할 매수 (Buy)", "20일선을 돌파하며 단기 상승 추세를 탔습니다. 추세 추종 매수가 유효합니다."
        elif rsi > 70 and macd_curr < sig_curr:
            position, reason = "🔷 적극 매도 (Strong Sell)", "단기 과열 상태(RSI>70)에서 데드크로스가 발생했습니다. 즉각적인 차익 실현이 필요합니다."
        elif close < ma20 and macd_curr < sig_curr:
            position, reason = "🔵 비중 축소 (Reduce)", "주가가 20일선 아래로 밀리며 하락세가 뚜렷합니다. 리스크 관리를 권장합니다."
        
        t_buy = int(close - (atr * 0.5)) if close > 1000 else round(close - (atr * 0.5), 2)
        t_sell = int(close + (atr * 1.5)) if close > 1000 else round(close + (atr * 1.5), 2)
        s_loss = int(close - (atr * 2.0)) if close > 1000 else round(close - (atr * 2.0), 2)
        
    else: 
        # 장기 복합 로직
        if close < bb_lower and rsi < 35:
            position, reason = "🔴 적극 매수 (Strong Buy)", "볼린저 밴드 하단을 이탈한 역사적 과매도 구간입니다. 중장기적 관점에서 매우 매력적인 자리입니다."
        elif close > ma200 and macd_curr > sig_curr:
            position, reason = "🟠 비중 확대 (Buy)", "장기 생명선(200일선) 위에서 단기 모멘텀도 살아있습니다. 지속 보유 및 추가 매수가 유효합니다."
        elif close > bb_upper and rsi > 70:
            position, reason = "🔵 비중 축소 (Reduce)", "볼린저 밴드 상단을 돌파하며 장기적 관점에서도 가격 부담이 커졌습니다. 일부 분할 매도를 권장합니다."
        elif close < ma200 and macd_curr < sig_curr:
            position, reason = "🔷 보수적 관망 (Wait)", "장기 추세선(200일선)이 깨졌고 하락 압력이 거셉니다. 바닥이 완전히 확인될 때까지 관망하세요."
        
        t_buy = int(close - (atr * 1.0)) if close > 1000 else round(close - (atr * 1.0), 2)
        t_sell = int(close + (atr * 4.0)) if close > 1000 else round(close + (atr * 4.0), 2)
        s_loss = int(close - (atr * 3.0)) if close > 1000 else round(close - (atr * 3.0), 2)

    return position, t_buy, t_sell, s_loss, reason, rsi, atr, comments

# ==========================================
# 3. 사이드바 및 UI 구성
# ==========================================
with st.sidebar:
    st.header("⚙️ 분석 설정 및 검색")
    # 투자 모드 토글 추가
    analyze_mode = st.radio("투자 성향 선택", ["단기/스윙 (최근 6개월 흐름 분석)", "장기 투자 (최근 1~2년 흐름 분석)"])
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    new_search = st.text_input("종목명/코드 입력", placeholder="분석할 종목을 입력하세요")
    target_query = new_search if st.button("🚀 데이터 분석 실행", type="primary", use_container_width=True) else None
    
    st.divider()
    st.markdown("### 🕒 최근 검색 리스트")
    for item in st.session_state.recent_searches:
        if st.button(f"▪️ {item['display_name']}", use_container_width=True):
            target_query = item['query']

if target_query:
    display_name, ticker_symbol, raw_query = parse_query(target_query)
    
    if {'query': raw_query, 'display_name': display_name} not in st.session_state.recent_searches:
        st.session_state.recent_searches.insert(0, {'query': raw_query, 'display_name': display_name})
        st.session_state.recent_searches = st.session_state.recent_searches[:5]

    with st.spinner(f"📡 '{display_name}' 데이터를 심층 분석 중입니다..."):
        df = get_stock_data(ticker_symbol, analyze_mode)
        
    if df.empty:
        st.error("데이터를 불러오지 못했습니다. 종목명이나 티커를 확인해주세요.")
    else:
        # 상단 요약
        cur_price = df['Close'].iloc[-1]
        diff = cur_price - df['Close'].iloc[-2]
        currency = "원" if cur_price > 1000 else "USD"
        
        mode_badge = "단기 트레이딩" if "단기" in analyze_mode else "장기 투자"
        st.subheader(f"📑 {display_name} AI 심층 리포트 ({mode_badge} 모드)")
        st.metric("현재 주가", f"{cur_price:,.0f} {currency}" if cur_price > 1000 else f"{cur_price:,.2f} {currency}", f"{diff:,.0f} {currency} (전일대비)")

        # 종합 매매 시그널 박스
        pos, buy, sell, stop, reason, rsi, atr, comments = generate_signal_and_comments(df, analyze_mode)
        
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("### 🎯 **종합 매매 타이밍**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("추천 포지션", pos)
            c2.metric("스마트 진입가", f"{buy:,}")
            c3.metric("목표 매도가", f"{sell:,}")
            c4.metric("ATR 손절가", f"{stop:,}")
            st.info(f"**종합 의견:** {reason}")

        # 개별 지표 상세 코멘트 박스
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🔬 **지표별 상세 코멘트**")
        with st.container(border=True):
            st.markdown(f"**[RSI - 매수/매도 강도]** (현재: {rsi:.1f})<br> <span style='color:gray;'>{comments['RSI']}</span>", unsafe_allow_html=True)
            st.divider()
            st.markdown(f"**[MACD - 추세 전환 흐름]**<br> <span style='color:gray;'>{comments['MACD']}</span>", unsafe_allow_html=True)
            st.divider()
            st.markdown(f"**[ATR - 가격 변동성]**<br> <span style='color:gray;'>{comments['ATR']}</span>", unsafe_allow_html=True)
            st.divider()
            st.markdown(f"**[OBV - 세력 매집/수급 흐름]**<br> <span style='color:gray;'>{comments['OBV']}</span>", unsafe_allow_html=True)
            
            if "장기" in analyze_mode and 'LONG' in comments:
                st.divider()
                st.markdown(f"📈 **[장기 추세 판단 지표]**<br> <span style='color:gray;'>{comments['LONG']}</span>", unsafe_allow_html=True)

        # 차트 영역
        st.markdown("<br>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["📈 주가 & 이동평균선 차트", "📊 OBV 누적 수급 차트"])
        
        chart_len = 250 if "장기" in analyze_mode else 120
        chart_df = df.tail(chart_len)
        
        with tab1:
            fig = go.Figure(data=[go.Candlestick(
                x=chart_df.index, open=chart_df['Open'], high=chart_df['High'], 
                low=chart_df['Low'], close=chart_df['Close'], name='주가',
                increasing_line_color='red', decreasing_line_color='blue'
            )])
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['MA20'], name='20일선', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['MA60'], name='60일선', line=dict(color='green')))
            
            if "장기" in analyze_mode:
                fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['MA200'], name='200일선 (장기추세)', line=dict(color='purple', width=2)))
                fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['BB_Upper'], name='볼린저 상단', line=dict(color='gray', dash='dash', width=1)))
                fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['BB_Lower'], name='볼린저 하단', line=dict(color='gray', dash='dash', width=1)))

            fig.update_layout(height=450, xaxis_rangeslider_visible=False, margin=dict(t=30))
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            obv_fig = go.Figure(data=[go.Scatter(x=chart_df.index, y=chart_df['OBV'], name='OBV', fill='tozeroy', line=dict(color='purple'))])
            obv_fig.update_layout(height=400, title=f"최근 {chart_len}일간의 OBV(수급 에너지) 흐름", margin=dict(t=40))
            st.plotly_chart(obv_fig, use_container_width=True)