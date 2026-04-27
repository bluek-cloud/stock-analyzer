import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import FinanceDataReader as fdr
from datetime import datetime, timedelta

# ==========================================
# 1. 페이지 설정 및 제목
# ==========================================
st.set_page_config(page_title="실시간 매수매도분석기", layout="wide")

st.title("📈 실시간 매수매도분석기")
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
    # 순수 6자리 코드만 반환
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
    df['BB_Mid'] = close.rolling(window=20).mean()
    df['BB_Std'] = close.rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)

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
    
    # OBV 벡터화 연산 (속도 최적화)
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

def generate_signal_and_comments(df, mode):
    latest, prev = df.iloc[-1], df.iloc[-2]
    close, rsi = float(latest['Close']), float(latest['RSI'])
    macd_curr, sig_curr = float(latest['MACD']), float(latest['Signal'])
    macd_prev, sig_prev = float(prev['MACD']), float(prev['Signal'])
    atr, obv = float(latest['ATR']), float(latest['OBV'])
    vol_ratio = float(latest['Vol_Ratio'])
    
    ma20 = float(latest['MA20'])
    ma200 = float(latest['MA200']) if not pd.isna(latest['MA200']) else close
    bb_lower = float(latest['BB_Lower']) if not pd.isna(latest['BB_Lower']) else close * 0.9

    prev20_close = float(df['Close'].iloc[-20]) if len(df) >= 20 else float(df['Close'].iloc[0])
    prev20_obv = float(df['OBV'].iloc[-20]) if len(df) >= 20 else float(df['OBV'].iloc[0])

    comments = {}
    
    if pd.isna(rsi): comments['RSI'] = "데이터 부족으로 RSI를 계산할 수 없습니다."
    elif rsi >= 70: comments['RSI'] = "🔥 **과매수 (Overbought)**: 단기 과열 구간입니다. 신규 진입은 자제하세요."
    elif 30 < rsi < 45: comments['RSI'] = "📉 **약세 국면**: 매도세가 우세합니다. 바닥 확인이 필요합니다."
    elif rsi <= 30: comments['RSI'] = "❄️ **과매도 (Oversold)**: 공포 구간이나 반등 가능성이 높습니다."
    else: comments['RSI'] = "📈 **정상 범위**: 안정적인 흐름을 유지 중입니다."

    if pd.isna(macd_curr) or pd.isna(sig_curr): comments['MACD'] = "데이터 부족으로 MACD를 계산할 수 없습니다."
    elif macd_curr > sig_curr: comments['MACD'] = "🚀 **상승 추세 유지**" 
    else: comments['MACD'] = "⚠️ **단기 조정/하락세**"

    if pd.isna(prev20_close) or prev20_close == 0 or pd.isna(obv): 
        comments['OBV'] = "수급 데이터를 계산할 수 없습니다."
    else:
        price_change = (close - prev20_close) / prev20_close * 100
        obv_change = obv - prev20_obv
        if price_change <= 0 and obv_change > 0: comments['OBV'] = "🕵️‍♂️ **숨은 매집**: 주가는 하락/횡보하지만 수급(OBV)은 증가 중입니다. 상승 전조일 수 있습니다."
        elif price_change > 0 and obv_change < 0: comments['OBV'] = "🚨 **이탈 징후**: 주가는 오르는데 거래량은 빠지고 있습니다. '가짜 상승'을 주의하세요."
        elif obv_change > 0: comments['OBV'] = "💪 **건강한 상승**: 주가와 매수 거래량이 동반 상승하며 추세를 뒷받침합니다."
        else: comments['OBV'] = "🍂 **수급 악화**: 매도 거래량이 압도하며 자금이 이탈하고 있습니다."

    if pd.isna(vol_ratio): comments['VOL'] = "거래량 데이터를 계산할 수 없습니다."
    elif vol_ratio >= 150: comments['VOL'] = f"🌋 **수급 폭발 ({vol_ratio:.0f}%)**: 최근 거래량이 크게 터졌습니다! 세력 개입 가능성이 높습니다."
    elif vol_ratio >= 110: comments['VOL'] = f"🌊 **거래 활발 ({vol_ratio:.0f}%)**: 시장의 관심이 쏠리며 유의미한 거래량이 유입되고 있습니다."
    else: comments['VOL'] = f"💤 **소외/관망 ({vol_ratio:.0f}%)**: 거래량이 말라붙었습니다. 에너지를 응축 중이거나 관심에서 멀어져 있습니다."

    if pd.isna(atr) or pd.isna(close) or close == 0:
        comments['ATR'] = "변동성 데이터를 계산할 수 없습니다."
    else:
        volatility_pct = (atr / close) * 100
        comments['ATR'] = f"현재 일평균 변동성은 {volatility_pct:.1f}% 수준입니다."

    position, reason = "⚖️ 관망 (Neutral)", "추세 확인 후 진입을 권장합니다."
    t_buy = close * 0.95
    t_sell = close * 1.05
    s_loss = close * 0.90
    
    if not pd.isna(atr):
        if "단기" in mode:
            if not pd.isna(rsi) and rsi < 40 and macd_curr > sig_curr and macd_prev <= sig_prev:
                position, reason = "🔴 적극 매수 (Strong Buy)", "과매도 부근 골든크로스 발생."
            elif not pd.isna(ma20) and close > ma20 and macd_curr > sig_curr:
                position, reason = "🟠 분할 매수 (Buy)", "20일선 위 안정적 상승 추세."
            elif not pd.isna(rsi) and rsi > 70 and macd_curr < sig_curr:
                position, reason = "🔷 적극 매도 (Strong Sell)", "단기 과열 및 데드크로스."
            t_buy, t_sell, s_loss = int(close - atr*0.5), int(close + atr*1.5), int(close - atr*2)
        else:
            if close < bb_lower and not pd.isna(rsi) and rsi < 35:
                position, reason = "🔴 적극 매수 (Strong Buy)", "역사적 저평가 매수 기회."
            elif not pd.isna(ma200) and close > ma200 and macd_curr > sig_curr:
                position, reason = "🟠 비중 확대 (Buy)", "장기 우상향 추세 유지."
            t_buy, t_sell, s_loss = int(close - atr), int(close + atr*4), int(close - atr*3)

    return position, t_buy, t_sell, s_loss, reason, rsi, atr, comments

# ==========================================
# 3. 사이드바 및 실행 UI
# ==========================================
with st.sidebar:
    st.header("⚙️ 분석 설정")
    analyze_mode = st.radio("투자 성향", ["단기/스윙 (6개월 분석)", "장기 투자 (2년 분석)"])
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
        st.error("데이터를 찾을 수 없습니다. 종목명이나 코드를 다시 확인해주세요.")
    else:
        cur_price = df['Close'].iloc[-1]
        diff = cur_price - df['Close'].iloc[-2]
        currency = "원"
        mode_badge = "단기" if "단기" in analyze_mode else "장기"
        
        st.subheader(f"📑 {display_name} 리포트 ({mode_badge})")
        st.metric("현재 주가", f"{cur_price:,.0f} {currency}", f"{diff:,.0f} {currency}")

        q_score = calculate_quant_score(df)
        st.write("### 💯 퀀트 매수 매력도 점수")
        st.progress(q_score / 100)
        st.write(f"현재 점수: **{q_score}점** / 100점")

        pos, buy, sell, stop, reason, rsi, atr, comments = generate_signal_and_comments(df, analyze_mode)
        pts, sup, res = detect_patterns_and_levels(df)
        
        # 🌟 기업 기초 체력을 완전히 제거하고 매매 타이밍과 패턴을 나란히 배치
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown("### 🎯 **종합 매매 타이밍**")
                st.warning(f"**포지션: {pos}**\n\n**의견:** {reason}")
                st.write(f"진입가: {buy:,.0f} | 목표가: {sell:,.0f} | 손절가: {stop:,.0f}")
                
        with col2:
            with st.container(border=True):
                st.markdown("### 🔍 **차트 패턴 및 주요 가격대**")
                p_text = ", ".join(pts) if pts else "특이 패턴 없음"
                st.write(f"📍 **발견된 패턴:** {p_text}")
                st.write(f"🛡️ **심리적 지지선:** {sup:,.0f}\n\n🚧 **강력 저항선:** {res:,.0f}")

        # 🌟 타이틀 자체를 누르는 팝오버 방식 (직관적인 UI)
        with st.expander("🔬 기술적 지표 상세 분석 보기", expanded=True):
            c1, c2 = st.columns([0.15, 0.85])
            with c1.popover("상대 거래량 ❓", use_container_width=True):
                st.info("**상대 거래량(Relative Volume)**\n\n최근 5일 평균 거래량 대비 오늘 거래량이 얼마나 터졌는지를 나타냅니다. 150~200% 이상이면 세력 유입이나 강한 추세 변화의 신호로 봅니다.")
            c2.markdown(comments.get('VOL', '데이터 없음'))
            
            c1, c2 = st.columns([0.15, 0.85])
            with c1.popover("OBV 누적 ❓", use_container_width=True):
                st.info("**OBV(On-Balance Volume)**\n\n거래량은 주가에 선행한다는 원리를 이용한 지표입니다. 주가가 하락해도 OBV가 상승하면 '숨은 매집'으로 판단하며, 반대의 경우 '이탈 징후'로 봅니다.")
            c2.markdown(comments.get('OBV', '데이터 없음'))
            
            c1, c2 = st.columns([0.15, 0.85])
            with c1.popover("RSI 강도 ❓", use_container_width=True):
                st.info("**RSI(상대강도지수)**\n\n주가의 상승 압력과 하락 압력 간의 상대적 강도를 나타냅니다. 70 이상은 '과매수(거품)', 30 이하는 '과매도(저평가)' 구간으로 해석합니다.")
            c2.markdown(comments.get('RSI', '데이터 없음'))
            
            c1, c2 = st.columns([0.15, 0.85])
            with c1.popover("MACD 흐름 ❓", use_container_width=True):
                st.info("**MACD(이동평균 수렴확산)**\n\n단기 추세선과 장기 추세선이 얼마나 가까워지고 멀어지는지를 측정합니다. 골든크로스가 발생하면 상승 추세의 시작으로 봅니다.")
            c2.markdown(comments.get('MACD', '데이터 없음'))
            
            c1, c2 = st.columns([0.15, 0.85])
            with c1.popover("ATR 변동성 ❓", use_container_width=True):
                st.info("**ATR(평균 실변동폭)**\n\n일정 기간 동안 주가가 얼마나 '출렁'거렸는지 변동성을 보여줍니다. ATR이 높을수록 주가가 급등락하기 쉬우므로 위험 관리가 필요합니다.")
            c2.markdown(comments.get('ATR', '데이터 없음'))

        tab1, tab2 = st.tabs(["주가 차트", "수급(OBV) 차트"])
        chart_df = df.tail(120 if "단기" in analyze_mode else 250)
        
        with tab1:
            fig = go.Figure(data=[go.Candlestick(x=chart_df.index, open=chart_df['Open'], high=chart_df['High'], low=chart_df['Low'], close=chart_df['Close'], name='주가')])
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['MA20'], name='20일선', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['MA60'], name='60일선', line=dict(color='green')))
            
            fig.update_layout(height=500, xaxis_rangeslider_visible=False, margin=dict(t=0, b=0, l=0, r=0), dragmode=False)
            fig.update_xaxes(fixedrange=True)
            fig.update_yaxes(fixedrange=True)
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
        with tab2:
            obv_fig = go.Figure(data=[go.Scatter(x=chart_df.index, y=chart_df['OBV'], name='OBV', fill='tozeroy', line=dict(color='purple'))])
            
            obv_fig.update_layout(height=400, title="누적 수급(OBV) 에너지", margin=dict(t=40, b=0, l=0, r=0), dragmode=False)
            obv_fig.update_xaxes(fixedrange=True)
            obv_fig.update_yaxes(fixedrange=True)
            
            st.plotly_chart(obv_fig, use_container_width=True, config={'displayModeBar': False})
