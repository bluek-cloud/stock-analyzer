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

# 다크/라이트 테마 가독성 보강 및 모바일 여백 CSS
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
    
    # 국내 주식 판별 (6자리 숫자 코드)
    if query.isdigit() and len(query) == 6:
        matched = krx_df[krx_df['Code'] == query]
        if not matched.empty:
            name = matched.iloc[0]['Name']
            return f"{name} ({query})", query, query, "원", 0
            
    matched = krx_df[krx_df['Name'] == query]
    if not matched.empty:
        code = matched.iloc[0]['Code']
        return f"{query} ({code})", code, query, "원", 0
        
    # 해외 주식으로 간주 (달러 표기 및 소수점 2자리 설정)
    return f"{query} (해외)", query, query, "$", 2

def calculate_indicators(df):
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
def get_stock_data(code):
    # 5년치 전체 데이터 로드 (자유 탐색 지원)
    start_date = (datetime.now() - timedelta(days=1825)).strftime('%Y-%m-%d')
    try:
        df = fdr.DataReader(code, start=start_date)
        if df.empty: return pd.DataFrame()
        return calculate_indicators(df)
    except: return pd.DataFrame()

# 🌟 듀얼 AI 엔진: 투자 성향(단기 vs 장기)에 따라 완벽히 다른 뷰 제공
def generate_signal_and_comments(df, sup, res, currency, decimals, mode):
    latest = df.iloc[-1]
    close = float(latest['Close'])
    rsi = float(latest['RSI'])
    macd = float(latest['MACD'])
    signal = float(latest['Signal'])
    vol_ratio = float(latest['Vol_Ratio'])
    atr = float(latest['ATR'])
    obv = float(latest['OBV'])
    
    prev20_close = float(df['Close'].iloc[-20]) if len(df) >= 20 else close
    prev20_obv = float(df['OBV'].iloc[-20]) if len(df) >= 20 else obv
    prev20_rsi = float(df['RSI'].iloc[-20]) if len(df) >= 20 else rsi
    prev_obv = float(df['OBV'].iloc[-5]) if len(df) >= 5 else obv

    is_short_term = "단기" in mode

    comments = {}
    
    # 지표 상세 코멘트
    if pd.isna(rsi): comments['RSI'] = "데이터 부족"
    elif rsi >= 70: comments['RSI'] = f"🔥 **과매수 (RSI: {rsi:.1f})**: 단기 과열권입니다. 수익 실현을 고려할 시점입니다."
    elif rsi <= 30: comments['RSI'] = f"❄️ **과매도 (RSI: {rsi:.1f})**: 공포 구간입니다. 기술적 반등 가능성이 높습니다."
    else: comments['RSI'] = f"📈 **정상 범위 (RSI: {rsi:.1f})**: 매수/매도세가 균형을 이루고 있습니다."

    if macd > signal: comments['MACD'] = "🚀 **상승 추세**: MACD가 시그널선을 상향 돌파하여 긍정적입니다."
    else: comments['MACD'] = "⚠️ **하락 추세**: MACD가 시그널선 아래에 위치해 단기 조정 압력이 있습니다."

    obv_status = "상승" if obv > prev_obv else "하락"
    comments['VOL'] = f"🌋 **상대 거래량 ({vol_ratio:.0f}%)**: 유의미한 거래 에너지가 포착되었습니다." if vol_ratio > 150 else f"➖ **보통 거래량 ({vol_ratio:.0f}%)**: 평이한 수준의 거래입니다."
    comments['OBV'] = f"🕵️‍♂️ **매집 확인**: 최근 5일간 누적 OBV가 {obv_status} 중입니다."
    
    volatility_pct = (atr / close) * 100
    comments['ATR'] = f"일평균 **{volatility_pct:.1f}% ({atr:,.{decimals}f}{currency})**의 변동폭을 보입니다."

    # 패턴 및 이격도 분석
    dist_to_sup = (close - sup) / sup * 100 if sup > 0 else 100
    dist_to_res = (res - close) / res * 100 if res > 0 else 100
    near_sup = dist_to_sup <= 5
    near_res = dist_to_res <= 5

    price_down, price_up = close < prev20_close, close > prev20_close
    obv_up, rsi_up = obv > prev20_obv, rsi > prev20_rsi
    
    bullish_div = price_down and (obv_up or rsi_up)
    bearish_div = price_up and (not obv_up or not rsi_up)

    # ==========================================
    # 🤖 AI 판단 로직 (단기 투자 vs 장기 투자 분리)
    # ==========================================
    if is_short_term:
        # 단기 투자 로직 (파동, 다이버전스, 단기 지표 민감도 높음)
        position, reason = "⚖️ 단기 관망", "단기 지표의 방향성이 혼재되어 명확한 타점이 필요합니다."
        if (rsi < 40 and macd > signal) or (near_sup and bullish_div):
            position, reason = "🔴 단기 적극 매수", "강력한 단기 반등 신호(상승 다이버전스/지지선)가 포착되었습니다."
        elif (rsi > 70 and macd < signal) or (near_res and bearish_div):
            position, reason = "🔷 단기 적극 매도", "단기 고점 징후가 뚜렷합니다. 빠른 차익 실현을 권장합니다."
        elif bullish_div or (macd > signal and vol_ratio > 120 and obv > prev_obv):
            position, reason = "🟠 단기 분할 매수", "긍정적인 매수 수급과 짧은 상승 모멘텀이 확인됩니다."
        elif bearish_div or (macd < signal and vol_ratio > 120 and obv < prev_obv):
            position, reason = "🔵 단기 비중 축소", "매도 우위 수급으로 단기 하락 리스크가 있습니다."
            
        ai_opinion = random.choice(["🤖 **StockMap AI 단기 스윙 리포트**\n\n", "⚡ **StockMap AI 단기 트레이딩 진단**\n\n"])
    else:
        # 장기 투자 로직 (큰 추세, 60일선 비교, 깊은 과매도 기준)
        ma60 = df['MA60'].iloc[-1]
        position, reason = "⚖️ 장기 관망", "장기적 관점에서 뚜렷한 큰 추세 전환이 확인되지 않았습니다."
        if (rsi < 35) or (near_sup and close > ma60):
            position, reason = "🔴 비중 확대 (장기)", "장기 우상향 추세 속에서 매력적인 저평가(눌림목) 구간입니다."
        elif rsi > 75 or near_res:
            position, reason = "🔷 비중 축소 (장기)", "장기 저항대에 도달했습니다. 리스크 관리 차원의 익절 구간입니다."
        elif macd > signal and close > ma60:
            position, reason = "🟠 보유 (Hold)", "장기 우상향 추세가 유효합니다. 안정적인 홀딩 구간입니다."
        elif macd < signal and close < ma60:
            position, reason = "🔵 진입 자제 (장기)", "장기 하락 추세가 진행 중이므로 섣부른 물타기를 자제하세요."

        ai_opinion = random.choice(["🤖 **StockMap AI 장기 가치투자 리포트**\n\n", "🌲 **StockMap AI 거시적 뷰 진단**\n\n"])

    # 공통 AI 코멘트 작성
    sr_text, div_text, trend_text = "", "", ""
    if near_sup: sr_text = f"주가가 주요 지지선(**{sup:,.{decimals}f}{currency}**)에 근접(이격도 {dist_to_sup:.1f}%)했습니다. 하방 경직성이 확보될 확률이 높은 자리입니다. "
    elif near_res: sr_text = f"주가가 강력한 저항선(**{res:,.{decimals}f}{currency}**)에 바짝 다가섰습니다. 이 구간 돌파 여부가 향후 흐름을 결정짓습니다. "
    
    if bullish_div: div_text = "💡 **[상승 다이버전스]** 가격은 눌리지만 보조지표는 상승 중입니다. 세력의 숨은 매집 가능성이 있습니다. "
    elif bearish_div: div_text = "⚠️ **[하락 다이버전스]** 주가는 오르지만 거래 수급이 뒤따라오지 못하는 불안정한 상승입니다. "

    if not sr_text and not div_text:
        trend_text = "현재 뚜렷한 특이 패턴은 없으나 MACD 시그널을 상회하며 우상향 모멘텀을 모색 중입니다. " if macd > signal else "큰 모멘텀 없이 조정을 거치는 휴식 국면입니다. "

    ai_opinion += sr_text + div_text + trend_text
    
    # 🌟 투자 성향에 따른 최종 전략 멘트 완전 분리
    if is_short_term:
        if "매수" in position: ai_opinion += "\n\n🎯 **단기 전략:** 단기 바닥 탈출 신호가 강합니다. **빠른 수익 실현과 짧은 손절가**를 기준으로 진입을 권장합니다."
        elif "매도" in position or "축소" in position: ai_opinion += "\n\n🎯 **단기 전략:** 단기 고점 징후가 강합니다. **신속한 차익 실현으로 수익을 지키는 방어적 대응**을 하십시오."
        else: ai_opinion += "\n\n🎯 **단기 전략:** 방향성이 뚜렷하지 않습니다. **확실한 타점이 올 때까지 섣부른 진입을 피하고 관망**하세요."
    else:
        if "확대" in position or "보유" in position: ai_opinion += "\n\n🎯 **장기 전략:** 거시적 바닥 구간이거나 추세가 양호합니다. **단기 잔파동(노이즈)을 무시하고 긴 호흡으로 모아가는 전략**을 권장합니다."
        elif "축소" in position or "자제" in position: ai_opinion += "\n\n🎯 **장기 전략:** 큰 사이클에서 고점 부근이거나 하락 추세입니다. **장기 물량을 일부 덜어내고 펀더멘탈 회복을 기다리십시오.**"
        else: ai_opinion += "\n\n🎯 **장기 전략:** 긴 시계열에서 방향성이 모호합니다. **섣부른 비중 확대보다 관망하며 다음 매크로 환경 변화를 대기**하세요."

    comments['AI'] = ai_opinion
    return position, reason, rsi, atr, comments

# ==========================================
# 3. 사이드바 및 실행 UI
# ==========================================
with st.sidebar:
    st.header("⚙️ 분석 설정")
    # 🌟 듀얼 버전 설정 스위치
    analyze_mode = st.radio("투자 성향 설정", ["단기 투자 (6개월 차트)", "장기 투자 (2년 차트)"])
    new_search = st.text_input("종목명/코드 입력", placeholder="삼성전자, AAPL, NVDA 등")
    target_query = new_search if st.button("🚀 분석 실행", type="primary", use_container_width=True) else None
    st.divider()
    for item in st.session_state.recent_searches:
        if st.button(f"▪️ {item['display_name']}", use_container_width=True): target_query = item['query']

if target_query:
    display_name, ticker_symbol, raw_query, currency_symbol, decimals = parse_query(target_query)
    if {'query': raw_query, 'display_name': display_name} not in st.session_state.recent_searches:
        st.session_state.recent_searches.insert(0, {'query': raw_query, 'display_name': display_name})
        st.session_state.recent_searches = st.session_state.recent_searches[:5]

    # 모드에 따라 스피너 멘트 변경
    sp_text = "단기 트레이딩" if "단기" in analyze_mode else "장기 가치투자"
    with st.spinner(f"📡 '{display_name}' {sp_text} 관점 분석 중..."):
        df = get_stock_data(ticker_symbol)
        
    if df.empty:
        st.error("데이터를 찾을 수 없습니다. 종목명을 다시 확인해주세요.")
    else:
        cur_price = df['Close'].iloc[-1]
        diff = cur_price - df['Close'].iloc[-2]
        
        st.subheader(f"📑 {display_name} 리포트")
        st.metric("현재 주가", f"{cur_price:,.{decimals}f} {currency_symbol}", f"{diff:,.{decimals}f} {currency_symbol}")

        q_score = calculate_quant_score(df)
        st.write(f"### 💯 퀀트 스코어: **{q_score}점**")
        st.progress(q_score / 100)

        pts, sup, res = detect_patterns_and_levels(df)
        pos, reason, rsi, atr, comments = generate_signal_and_comments(df, sup, res, currency_symbol, decimals, analyze_mode)
        
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown("### 🎯 **종합 매매 타이밍**")
                st.warning(f"**포지션: {pos}**\n\n**요약의견:** {reason}")
                st.write(f"진입가: {cur_price*0.95:,.{decimals}f} | 목표가: {cur_price*1.05:,.{decimals}f}")
                
        with col2:
            with st.container(border=True):
                st.markdown("### 🔍 **차트 패턴 및 지지/저항**")
                p_text = ", ".join(pts) if pts else "특이 패턴 없음"
                st.write(f"📍 **패턴:** {p_text}")
                st.write(f"🛡️ **지지:** {sup:,.{decimals}f} | 🚧 **저항:** {res:,.{decimals}f}")

        with st.expander("🔬 지표별 상세 수치 분석", expanded=True):
            for label, info_text, key in [
                ("상대 거래량", "최근 5일 평균 거래량 대비 당일의 거래 에너지를 나타냅니다.", "VOL"),
                ("OBV 누적", "거래량은 주가에 선행한다는 원리를 이용한 세력 매집 지표입니다.", "OBV"),
                ("RSI 강도", "주가의 과매수(70 이상) 및 과매도(30 이하) 상태를 측정합니다.", "RSI"),
                ("MACD 흐름", "단기/장기 추세의 수렴과 확산을 통해 추세 반전을 포착합니다.", "MACD"),
                ("ATR 변동성", "일정 기간 주가의 평균 실질 변동폭을 화폐 단위로 보여줍니다.", "ATR")
            ]:
                c1, c2 = st.columns([0.2, 0.8])
                with c1.popover(label, use_container_width=True): st.info(f"**{label}**\n\n{info_text}")
                c2.markdown(comments.get(key, '데이터 없음'))
            st.divider()
            # 🌟 단기/장기에 맞춰진 맞춤형 AI 리포트 출력
            st.info(comments.get('AI'))

        # ==========================================
        # 🌟 5년 전체 탐색 지원 핀치 줌 차트 (초기 범위 설정)
        # ==========================================
        tab1, tab2 = st.tabs(["주가 차트 (확대/축소)", "수급(OBV) 에너지"])
        view_days = 180 if "단기" in analyze_mode else 730
        initial_start = datetime.now() - timedelta(days=view_days)
        initial_end = datetime.now()
        
        with tab1:
            fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='주가')])
            fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='20일선', line=dict(color='orange', width=1)))
            fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], name='60일선', line=dict(color='green', width=1)))
            fig.update_layout(
                height=450, margin=dict(t=10, b=10, l=0, r=0), dragmode='pan', hovermode='x unified',
                xaxis=dict(range=[initial_start, initial_end], rangeslider=dict(visible=False))
            )
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': False})
            
        with tab2:
            obv_fig = go.Figure(data=[go.Scatter(x=df.index, y=df['OBV'], name='OBV', fill='tozeroy', line=dict(color='purple'))])
            obv_fig.update_layout(height=350, margin=dict(t=10, b=10, l=0, r=0), dragmode='pan', xaxis=dict(range=[initial_start, initial_end]))
            st.plotly_chart(obv_fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': False})
