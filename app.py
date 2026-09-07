import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import FinanceDataReader as fdr
from datetime import datetime, timedelta
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 1. 페이지 설정 및 세션 관리 (상태 꼬임 무한루프 버그 패치)
# ==========================================
st.set_page_config(page_title="StockMap", layout="wide")

if 'target_query' not in st.session_state:
    st.session_state.target_query = None
if 'recent_searches' not in st.session_state:
    st.session_state.recent_searches = []
if 'trigger_search' not in st.session_state:
    st.session_state.trigger_search = False
if 'search_input' not in st.session_state:
    st.session_state.search_input = ""

# 모바일 및 데스크톱 가독성 확대를 위해 글자 포인트 스케일업 스타일 시트 적용
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
        font-size: 0.95rem;
        line-height: 1.6;
        background-color: rgba(255, 165, 0, 0.05);
        border-left: 4px solid #FF8C00;
    }
    [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {
        font-size: 1.05rem !important;
        line-height: 1.6 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 최근 검색어 선택 시 입력 폼 버퍼까지 완벽 동기화하여 레이스 컨디션 차단
def on_recent_click(query):
    st.session_state.target_query = query
    st.session_state.search_input = query
    st.session_state.trigger_search = True

# 엔터키 및 유저 직접 입력 시 독립적으로 데이터 흐름을 제어하는 콜백 함수
def on_search_input_change():
    if st.session_state.search_input:
        st.session_state.target_query = st.session_state.search_input

# ==========================================
# 2. 공통 데이터 처리 함수 (외풍 방어막 유지)
# ==========================================
@st.cache_data(ttl=86400)
def get_krx_data():
    try:
        return fdr.StockListing('KRX')
    except Exception:
        return pd.DataFrame(columns=['Code', 'Name', 'Market', 'Marcap'])

def parse_query(query):
    query = query.strip().upper()
    krx_df = get_krx_data()
    
    if not krx_df.empty:
        if query.isdigit() and len(query) == 6:
            matched = krx_df[krx_df['Code'] == query]
            if not matched.empty:
                return f"{matched.iloc[0]['Name']} ({query})", query, query, "원", 0
        matched = krx_df[krx_df['Name'] == query]
        if not matched.empty:
            code = matched.iloc[0]['Code']
            return f"{query} ({code})", code, query, "원", 0
            
    if query.isdigit() and len(query) == 6:
        return f"국내 종목 ({query})", query, query, "원", 0
    return f"{query} (해외)", query, query, "$", 2

@st.cache_data(ttl=60)
def get_stock_data(code, days=1825):
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    try:
        df = fdr.DataReader(code, start=start_date)
        if df.empty: return pd.DataFrame()
        if df.index.tz is not None:
            try:
                df.index = df.index.tz_convert(None)
            except Exception:
                df.index = df.index.tz_localize(None)
        return df
    except Exception: return pd.DataFrame()

def calculate_indicators(df):
    if df.empty or len(df) < 2: return df
    df = df.copy()  
    close = df['Close'].squeeze()
    
    df['MA20'] = close.rolling(window=20).mean()
    df['MA60'] = close.rolling(window=60).mean()
    
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    tr = pd.concat([df['High'] - df['Low'], (df['High'] - close.shift()).abs(), (df['Low'] - close.shift()).abs()], axis=1).max(axis=1)
    df['ATR'] = tr.ewm(alpha=1/14, adjust=False).mean() 
    
    df['STD'] = close.rolling(window=20).std()
    df['BB_Upper'] = df['MA20'] + (df['STD'] * 2)
    df['BB_Lower'] = df['MA20'] - (df['STD'] * 2)
    df['BBW'] = (df['BB_Upper'] - df['BB_Lower']) / (df['MA20'] + 1e-10) * 100 
    
    direction = np.sign(delta).fillna(0) 
    df['OBV'] = (df['Volume'] * direction).cumsum()
    df['Vol_MA5'] = df['Volume'].rolling(window=5).mean()
    vol_ma5_prev = df['Volume'].shift(1).rolling(window=5).mean()
    df['Vol_Ratio'] = (df['Volume'] / (vol_ma5_prev.fillna(df['Vol_MA5']) + 1e-10)) * 100
    
    high_diff, low_diff = df['High'].diff(), -df['Low'].diff()
    plus_dm = pd.Series(np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0), index=df.index).ewm(alpha=1/14, adjust=False).mean()
    minus_dm = pd.Series(np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0), index=df.index).ewm(alpha=1/14, adjust=False).mean()
    plus_di, minus_di = 100 * (plus_dm / (df['ATR'] + 1e-10)), 100 * (minus_dm / (df['ATR'] + 1e-10))
    df['ADX'] = (100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)).ewm(alpha=1/14, adjust=False).mean() 
    df['+DI'], df['-DI'] = plus_di, minus_di
    
    return df

def calculate_quant_score(df, is_short_term):
    if len(df) < 5: return 0
    latest, prev = df.iloc[-1], df.iloc[-2]
    score = 0
    if is_short_term:
        if not pd.isna(latest['RSI']):
            rsi_val = latest['RSI']
            if 50 <= rsi_val <= 68: score += 25  # 건강한 상승 모멘텀 유지 구간
            elif rsi_val < 35 and latest['Close'] > prev['Close']: score += 25  # 과매도권 양봉 반등 확인 (역발상 타점)
            elif 35 <= rsi_val < 50 and latest['Close'] > prev['Close']: score += 15  # 눌림목 반등 시도
            elif rsi_val < 35: score += 5  # 과매도이나 지속 음봉 하락 중 (떨어지는 칼날 위험)
            else: score += 10  # 과열권(RSI > 68) 추세 유지
        if not pd.isna(latest['MACD']) and not pd.isna(latest['Signal']):
            if latest['MACD'] > latest['Signal']: score += 25
        obv_ref_short = df['OBV'].iloc[-min(5, len(df)-1)]
        if not pd.isna(latest['OBV']) and latest['OBV'] > obv_ref_short: score += 30
        if not pd.isna(latest['Vol_Ratio']):
            if latest['Vol_Ratio'] >= 150 and latest['Close'] > prev['Close']: score += 20
    else:
        if not pd.isna(latest['MA60']) and latest['Close'] > latest['MA60']: score += 30
        highest_60 = df['Close'].tail(60).max()
        if latest['Close'] >= highest_60 * 0.95: score += 20
        if not pd.isna(latest['MACD']) and not pd.isna(latest['Signal']):
            if latest['MACD'] > latest['Signal']: score += 20
        obv_ref_long = df['OBV'].iloc[-min(13, len(df)-1)]
        if not pd.isna(latest['OBV']) and latest['OBV'] > obv_ref_long: score += 20
        if not pd.isna(latest['RSI']):
            if 40 <= latest['RSI'] <= 70: score += 10
    return min(score, 100)

def detect_patterns_and_levels(df):
    if len(df) < 3: return [], 0, 0  
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    patterns = []
    
    body = abs(latest['Open'] - latest['Close'])
    candle_range = latest['High'] - latest['Low']
    lower_shadow = min(latest['Open'], latest['Close']) - latest['Low']
    upper_shadow = latest['High'] - max(latest['Open'], latest['Close'])
    
    # 1. 망치형 / 교수형 정밀 판정 (위치에 따른 분기)
    if candle_range > 0:
        if body <= candle_range * 0.35 and lower_shadow >= candle_range * 0.5 and upper_shadow <= candle_range * 0.15:
            is_pullback = (not pd.isna(latest['MA20']) and latest['Close'] <= latest['MA20']) or (len(df) >= 5 and latest['Close'] < df['Close'].iloc[-5])
            if is_pullback:
                patterns.append("🔨 망치형 (바닥권 반등 신호)")
            else:
                patterns.append("⚠️ 교수형 (고점 경고 신호)")

    # 2. 장악형 패턴 정밀 판정 (상승 장악형 / 하락 장악형)
    prev_body = abs(prev['Open'] - prev['Close'])
    if prev['Close'] < prev['Open'] and latest['Close'] > latest['Open']:  # 전일 음봉, 당일 양봉
        if latest['Open'] <= prev['Open'] and latest['Close'] > prev['Open'] and body >= prev_body:
            patterns.append("🚀 상승 장악형 (추세 반전)")
    elif prev['Close'] > prev['Open'] and latest['Close'] < latest['Open']:  # 전일 양봉, 당일 음봉
        if latest['Open'] >= prev['Open'] and latest['Close'] < prev['Open'] and body >= prev_body:
            patterns.append("🚨 하락 장악형 (하락 반전 경고)")
    
    # 3. 지지선 / 저항선 및 신고가 산출 (최대 250거래일 기준)
    lookback = min(250, len(df))
    past_df = df.iloc[-lookback:-1] if lookback > 1 else df.iloc[:-1]
    if past_df.empty:
        return patterns, latest['Close'] * 0.95, latest['Close'] * 1.05
    
    cur_price = latest['Close']
    tolerance = cur_price * 0.025
    
    def cluster_levels(prices_list):
        if not prices_list: return []
        sorted_prices = sorted(prices_list)
        clusters = []
        for p in sorted_prices:
            matched = False
            for c in clusters:
                if abs(p - c['center']) <= tolerance:
                    c['prices'].append(p)
                    c['center'] = sum(c['prices']) / len(c['prices'])
                    matched = True
                    break
            if not matched:
                clusters.append({'center': p, 'prices': [p]})
        return clusters

    # 저점 후보 (지지): Low 및 Close의 로컬 미니멈
    low_series = past_df['Low']
    low_mask = (low_series <= low_series.shift(1)) & (low_series <= low_series.shift(-1))
    support_candidates = past_df.loc[low_mask, 'Low'].tolist()
    close_low_mask = (past_df['Close'] <= past_df['Close'].shift(1)) & (past_df['Close'] <= past_df['Close'].shift(-1))
    support_candidates.extend(past_df.loc[close_low_mask, 'Close'].tolist())

    # 고점 후보 (저항): High 및 Close의 로컬 맥시멈
    high_series = past_df['High']
    high_mask = (high_series >= high_series.shift(1)) & (high_series >= high_series.shift(-1))
    resistance_candidates = past_df.loc[high_mask, 'High'].tolist()
    close_high_mask = (past_df['Close'] >= past_df['Close'].shift(1)) & (past_df['Close'] >= past_df['Close'].shift(-1))
    resistance_candidates.extend(past_df.loc[close_high_mask, 'Close'].tolist())

    # 지지선 산출: 현재가 이하 클러스터 중 현재가에 가장 가까우면서도 지지 신뢰도가 높은 레벨
    sup_clusters = cluster_levels(support_candidates)
    valid_sups = [c for c in sup_clusters if c['center'] <= cur_price]
    if valid_sups:
        valid_sups.sort(key=lambda c: abs(cur_price - c['center']) / (len(c['prices']) ** 0.5))
        support = valid_sups[0]['center']
    else:
        below_lows = past_df[past_df['Low'] <= cur_price]['Low']
        support = below_lows.max() if not below_lows.empty else past_df['Low'].min()

    # 저항선 산출: 현재가 초과 클러스터 중 현재가에 가장 가까운 저항 레벨
    res_clusters = cluster_levels(resistance_candidates)
    valid_res = [c for c in res_clusters if c['center'] > cur_price]
    if valid_res:
        valid_res.sort(key=lambda c: abs(c['center'] - cur_price) / (len(c['prices']) ** 0.5))
        resistance = valid_res[0]['center']
    else:
        above_highs = past_df[past_df['High'] > cur_price]['High']
        resistance = 0 if above_highs.empty else above_highs.min()

    return patterns, support, resistance

def generate_detailed_opinions(df, sup, res, currency, decimals, is_short_term, time_unit, q_score, weekly_bullish=None):
    md_currency = currency.replace('$', r'\$')
    latest, prev = df.iloc[-1], df.iloc[-2]
    close, rsi, obv, vol_ratio, atr = map(float, [latest['Close'], latest['RSI'], latest['OBV'], latest['Vol_Ratio'], latest['ATR']])
    ma20, ma60, adx, p_di, m_di = map(float, [latest['MA20'], latest['MA60'], latest['ADX'], latest['+DI'], latest['-DI']])
    
    prev_candle_close = float(prev['Close'])
    prev_ma20 = float(prev['MA20']) if not pd.isna(prev['MA20']) else ma20
    
    simple_lookback = min(5, len(df) - 1) if len(df) > 1 else 1
    long_lookback   = min(13, len(df) - 1) if len(df) > 1 else 1
    obv_lookback    = simple_lookback if is_short_term else long_lookback
    simple_prev_obv = float(df['OBV'].iloc[-obv_lookback])
    
    bullish_div = False
    if len(df) >= 15:
        recent_chunk = df.iloc[-4:]
        past_chunk = df.iloc[-30:-4] if len(df) >= 30 else df.iloc[:-4]
        if not past_chunk.empty and not recent_chunk.empty:
            p_min_idx = past_chunk['Low'].idxmin()
            r_min_idx = recent_chunk['Low'].idxmin()
            p_low = float(past_chunk.loc[p_min_idx, 'Low'])
            r_low = float(recent_chunk.loc[r_min_idx, 'Low'])
            # 최근 저가가 이전 저가 이하이거나 거의 유사(신저가 형성)
            if r_low <= p_low * 1.01:
                p_rsi = float(past_chunk.loc[p_min_idx, 'RSI'])
                r_rsi = float(recent_chunk.loc[r_min_idx, 'RSI'])
                p_obv = float(past_chunk.loc[p_min_idx, 'OBV'])
                r_obv = float(recent_chunk.loc[r_min_idx, 'OBV'])
                # 주가는 하락했으나 RSI나 OBV 저점은 뚜렷하게 높아지는 다이버전스
                if (not pd.isna(r_rsi) and not pd.isna(p_rsi) and r_rsi > p_rsi + 2.0) or \
                   (not pd.isna(r_obv) and not pd.isna(p_obv) and r_obv > p_obv):
                    bullish_div = True

    has_valid_adx = not pd.isna(adx)
    has_valid_ma = not pd.isna(ma20) and not pd.isna(ma60)

    is_squeeze = latest['BBW'] <= df['BBW'].iloc[-120:].min() * 1.05 if (len(df) > 120 and not pd.isna(latest['BBW'])) else False
    if is_squeeze: regime = "에너지 응축 (스퀴즈)"
    elif vol_ratio >= 150 and has_valid_adx and adx > df['ADX'].iloc[-2] and adx > 20: regime = "변동성 폭발"
    elif has_valid_adx and adx < 25: regime = "횡보 박스"
    elif has_valid_ma and ma20 >= ma60 and close >= ma60: regime = "강세 추세" if p_di > m_di else "상승 조정"
    elif len(df) < 30: regime = "데이터 수집/안정화 중"
    else: regime = "약세 추세"

    box_pos = ((close - sup) / (res - sup) * 100) if (res > sup and sup > 0) else 100
    drop_pct = ((prev['Close'] - close) / prev['Close'] * 100) if prev['Close'] > 0 else 0
    is_falling_knife = (drop_pct >= 7.0 and vol_ratio >= 120) or (drop_pct >= 10.0)

    macd_diff = float(latest['MACD'] - latest['Signal']) if not pd.isna(latest['MACD']) and not pd.isna(latest['Signal']) else 0
    vol_pct = (atr / close) * 100 if close > 0 else 0

    adx_disp = f"**{adx:.1f}**" if has_valid_adx else "**산출 중**"
    comments = {}
    comments['ADX'] = f"현재 ADX 추세강도 지수는 {adx_disp}이며, 알고리즘은 현재 시장을 **[{regime}]** 국면으로 확정했습니다."
    
    rsi_disp = f"RSI({rsi:.1f})" if not pd.isna(rsi) else "RSI(산출 중)"
    if regime == "에너지 응축 (스퀴즈)":
        comments['RSI'] = f"{rsi_disp}: 볼린저 밴드 수축 국면이므로 RSI의 움직임이 매우 둔화되어 있습니다. 방향성 탐색 중입니다."
        comments['MACD'] = f"MACD({macd_diff:,.{decimals}f}): 이동평균선이 밀집하며 MACD도 0선에 완전히 수렴했습니다. 폭풍 전야의 고요한 상태입니다."
    elif regime == "횡보 박스":
        comments['RSI'] = f"{rsi_disp}: 횡보장에서는 RSI의 신뢰도가 가장 높습니다. " + ("박스권 하단 지지선(과매도) 터치로 기술적 반등이 예상됩니다." if (not pd.isna(rsi) and rsi <= 40) else "박스권 상단 저항선(과매수) 도달로 조정이 예상됩니다." if (not pd.isna(rsi) and rsi >= 60) else "박스권 중간에서 뚜렷한 방향성을 탐색 중입니다.")
        comments['MACD'] = f"MACD({macd_diff:,.{decimals}f}): 뚜렷한 추세가 부재한 박스권이므로 MACD 크로스 신호의 신뢰도는 다소 떨어집니다."
    elif regime == "강세 추세":
        comments['RSI'] = f"{rsi_disp}: 강세장에서는 지표가 쉽게 과열권에 진입합니다. " + ("강한 매수세로 단기 과열(70 이상) 상태이나 추세는 굳건합니다." if (not pd.isna(rsi) and rsi >= 70) else "상승 추세 중 발생한 건전한 눌림목(조정) 타점입니다." if (not pd.isna(rsi) and rsi <= 50) else "안정적인 상승 탄력을 유지하고 있습니다.")
        comments['MACD'] = f"MACD({macd_diff:,.{decimals}f}): 상승 모멘텀이 강하게 유지되며 이평선 정배열 확장을 지지하고 있습니다."
    elif regime == "상승 조정": 
        comments['RSI'] = f"{rsi_disp}: 상승 추세 속에서 조정을 받으며 지표가 식어가고 있습니다. 40~50 부근에서 지지받는지 확인이 필요합니다."
        comments['MACD'] = f"MACD({macd_diff:,.{decimals}f}): 단기적으로 데드크로스가 발생하거나 모멘텀이 둔화되었으나, 장기 상승 추세 베이스는 훼손되지 않았습니다."
    elif regime == "약세 추세":
        comments['RSI'] = f"{rsi_disp}: 약세장에서는 지표가 지속적으로 침체권에 머눕니다. " + ("일시적인 기술적 반등 구간으로 매도를 고려할 시점입니다." if (not pd.isna(rsi) and rsi >= 55) else "극단적 과매도 상태이나, 지속적인 하락 압력을 받고 있으므로 섣부른 진입은 피해야 합니다." if (not pd.isna(rsi) and rsi <= 30) else "지속적인 하락 압력을 받고 있습니다.")
        comments['MACD'] = f"MACD({macd_diff:,.{decimals}f}): 하락 모멘텀이 강하며, 추세 반전을 암시하는 뚜렷한 신호가 아직 없습니다."
    elif regime == "변동성 폭발":
        comments['RSI'] = f"{rsi_disp}: 변동성 폭발로 인해 투심이 한쪽으로 극단적으로 쏠리는 오버슈팅 및 투매 국면입니다."
        comments['MACD'] = f"MACD({macd_diff:,.{decimals}f}): 단기 모멘텀이 평소의 범위를 벗어나 급격하게 방향성을 분출하고 있습니다."
    else:
        comments['RSI'] = f"{rsi_disp}: 데이터 축적 중으로 지표 신뢰도를 검증 중입니다."
        comments['MACD'] = f"MACD({macd_diff:,.{decimals}f}): 추세 형성 초기 단계입니다."

    comments['VOL'] = f"상대 거래량이 평균 대비 **{vol_ratio:.0f}%** 수준입니다. " + ("대량 거래가 터지며 시장의 강한 이목이 집중되었습니다." if vol_ratio > 150 else "평이한 수준의 거래가 이뤄지고 있습니다.")
    comments['OBV'] = f"최근 {obv_lookback}{time_unit}간 누적 수급(OBV)이 **{'상승(자금 유입)' if obv > simple_prev_obv else '하락(자금 이탈)'}** 중입니다."
    comments['ATR'] = f"예상되는 실질 변동폭(ATR)은 주당 평균 **{vol_pct:.1f}% ({atr:,.{decimals}f}{md_currency})** 수준입니다."

    if is_short_term:
        if is_falling_knife: pos, strategy = "🔷 투매 진행 중 (절대 관망)", "대량 거래를 동반한 치명적 급락 발생. '떨어지는 칼날'이므로 하락 진정 시까지 절대 관망하십시오."
        elif res == 0 and close > prev['Close']: pos, strategy = "🔴 신고가 랠리 (강력 홀딩)", "과거 매물대를 모두 뚫어낸 신고가 영역입니다. 추세 훼손 전까지 수익을 극대화하십시오."
        elif regime == "에너지 응축 (스퀴즈)":
            if bullish_div: pos, strategy = "🔴 응축 구간 선취매", "에너지 응축 중 상승 다이버전스 포착. 상방 돌파 확률이 매우 높으므로 선취매가 유효합니다."
            else: pos, strategy = "⚖️ 방향성 대기 (관망)", "볼린저 밴드 극도 수축 상태. 뚜렷한 방향성 분출 전까지 관망하십시오."
        elif regime == "횡보 박스":
            if box_pos <= 35 or bullish_div: pos, strategy = "🟠 박스권 하단 매수", "박스권 하단 지지 확인 및 반전 시그널 발생. 상단을 목표로 한 단기 스윙 전략이 유효합니다."
            elif box_pos >= 65:
                if obv > simple_prev_obv and vol_ratio >= 100: pos, strategy = "🟠 돌파 기대 (보유)", "저항선 근접했으나 긍정적 수급 유입 중. 돌파 여부를 예의주시하며 홀딩을 권장합니다."
                elif obv > simple_prev_obv and vol_ratio < 100: pos, strategy = "⚖️ 돌파 탐색 (관망/분할매도)", "수급은 좋으나 폭발적 거래량이 부족합니다. 돌파 여부 관찰 및 일부 비중 축소를 고려하십시오."
                else: pos, strategy = "🔵 단기 박스권 상단 매도", "저항 돌파를 위한 수급이 부족합니다. 리스크 관리를 위해 적극적인 비중 축소를 권장합니다."
            else: pos, strategy = "⚖️ 단기 관망", "박스권 중간 지대 위치. 뚜렷한 타점 도달 전까지 진입을 자제하십시오."
        elif regime in ["강세 추세", "상승 조정"]:
            if rsi <= 55 or bullish_div: pos, strategy = "🔴 추세 눌림목 적극 매수", "강한 상승 추세 속 건전한 눌림목 발생. 확률 높은 매수 타점으로 평가됩니다."
            elif rsi >= 75: pos, strategy = "🔵 분할 익절", "안정적 추세나 단기 과열권에 진입했습니다. 수익 보호를 위해 보유 비중 분할 실현을 권장합니다."
            else: pos, strategy = "🟠 추세 보유 (홀딩)", "우상향 흐름 진행 중. 상승 추세 이탈 전까지 지속 보유하여 수익을 극대화하십시오."
        elif regime == "약세 추세":
            if rsi >= 45 and close > prev['Close']:
                if obv > simple_prev_obv and vol_ratio > 100: pos, strategy = "🟠 의미 있는 반등 시도", "하락장 속 유의미한 수급/거래량 동반 반등. 추세 전환의 단초가 될 수 있습니다."
                else: pos, strategy = "🔵 데드캣 바운스 경계 (매도)", "수급 뒷받침이 부족한 단순 기술적 반등(속임수)일 확률이 높습니다. 탈출 기회로 삼으십시오."
            elif rsi <= 30 or bullish_div: pos, strategy = "🟠 단기 기술적 반등 공략", "극단적 과매도 및 다이버전스 발생. 짧은 수익을 목표로 한 기술적 반등 매매만 권장합니다."
            else: pos, strategy = "🔷 적극 매도 및 관망", "하락 추세가 지배적입니다. 물타기를 자제하고 현금 비중을 높여 관망하십시오."
        elif regime == "변동성 폭발":
            if close > prev_candle_close:
                pos, strategy = "🔴 돌파 추세 추종", "평균을 상회하는 대량 거래와 함께 상방 돌파 분출. 단기 모멘텀 추종이 유리합니다."
            else:
                pos, strategy = "🔷 하방 변동성 폭발 (적극 관망)", "대량 거래를 동반한 강한 하방 이탈 발생. 추가 낙폭 위험이 크므로 절대 매수를 금지합니다."
        else:
            pos, strategy = "⚖️ 단기 관망", "뚜렷한 추세나 타점이 부재한 변곡점 구간입니다. 명확한 방향성 확인 후 대응하십시오."
    else:
        if is_falling_knife:
            pos, strategy = "🔷 장기 투매 진행 중 (절대 매수금지)", "주봉 기준 대량 거래를 동반한 장대음봉 폭락이 포착되었습니다. 추가 연쇄 하락 위험이 극도로 큽니다."
        elif regime == "변동성 폭발":
            if close > prev_candle_close: pos, strategy = "🔴 장기 대시세 분출 (비중 확대)", "장기 박스권을 상방으로 막대한 거래량과 함께 뚫어내는 대형 우상향 시작 타점입니다."
            else: pos, strategy = "🔷 하방 변동성 폭발 (적극 관망)", "폭발적인 매도 자금 이탈과 함께 중장기 주요 구조선들을 연쇄적으로 이탈하는 초고위험 구간입니다."
        elif regime == "상승 조정" and (box_pos > 50 or obv < simple_prev_obv): pos, strategy = "⚖️ 장기 눌림목 대기", "장기 상승장 내 조정 구간이나, 하락세 진정 및 지지선 확인 전까지 보수적 관망을 권장합니다."
        elif regime in ["강세 추세", "상승 조정"]: pos, strategy = "🔴 비중 확대 (장기)", "대세 상승장에 진입했습니다. 장기적 시각에서 비중 확대 및 홀딩 전략이 유효합니다."
        elif regime == "약세 추세" and rsi < 30: pos, strategy = "🟠 저점 분할 매집", "역사적 저평가 구간 진입. 펀더멘털 확인 후 긴 호흡으로 1차 분할 매집을 고려할 수 있습니다."
        elif regime == "약세 추세": pos, strategy = "🔷 비중 축소 (장기)", "대세 하락장이 지속 중입니다. 포트폴리오 방어를 위해 주식 비중 축소를 권장합니다."
        else: pos, strategy = "⚖️ 장기 관망", "장기 추세의 변곡점이거나 방향성이 불분명한 구간입니다. 확실한 추세 형성 시까지 관망하십시오."

    buy_list = {
        "🔴 신고가 랠리 (강력 홀딩)", "🔴 추세 눌림목 적극 매수", "🔴 돌파 추세 추종",
        "🔴 응축 구간 선취매", "🔴 비중 확대 (장기)", "🔴 장기 대시세 분출 (비중 확대)",
        "🟠 박스권 하단 매수", "🟠 돌파 기대 (보유)", "🟠 의미 있는 반등 시도",
        "🟠 단기 기술적 반등 공략", "🟠 저점 분할 매집", "🟠 추세 보유 (홀딩)"
    }
    sell_list = {
        "🔵 단기 박스권 상단 매도", "🔵 분할 익절", "🔵 데드캣 바운스 경계 (매도)",
        "🔷 투매 진행 중 (절대 관망)", "🔷 장기 투매 진행 중 (절대 매수금지)",
        "🔷 적극 매도 및 관망", "🔷 비중 축소 (장기)", "🔷 하방 변동성 폭발 (적극 관망)",
        "⚖️ 돌파 탐색 (관망/분할매도)"
    }
    
    if pos in buy_list and q_score < 30: pos, strategy = ("⚖️ 단기 관망" if is_short_term else "⚖️ 장기 관망"), f"매수/보유 신호가 포착되었으나 퀀트 스코어({q_score}점)가 다소 낮아 신뢰도가 떨어집니다. 관망을 권장합니다."
    elif pos in sell_list and q_score > 70 and not is_falling_knife: pos, strategy = ("⚖️ 단기 관망" if is_short_term else "⚖️ 장기 관망"), f"매도/비중축소 신호가 포착되었으나 퀀트 스코어({q_score}점)가 양호하여 상충이 발생합니다. 방향성 확인 후 대응하십시오."

    # 🌟 이중 이스케이프 제거: 한 줄씩 예쁘게 개행되도록 순수 \n\n으로 정렬
    mode_str = "단기 스윙" if is_short_term else "장기 가치투자"
    ai_op = f"🤖 **StockMap AI {mode_str} 심층 진단 리포트**\n\n"
    ai_op += f"🔍 **[시장 국면 분류]**\n\n• 현재 해당 종목은 **[{regime}]** 국면에 위치해 있습니다.\n\n"
    
    if is_short_term and weekly_bullish is not None:
        ai_op += f"⏱️ **[MTF 다중 시간대 분석]**\n\n"
        if regime in ["강세 추세", "상승 조정"]: 
            ai_op += "• **장기 흐름:** 주봉(장기) 상승세가 굳건하여 일봉(단기) 수준에서도 강한 지지력을 보입니다.\n\n" if weekly_bullish else "• **장기 흐름:** 단기는 긍정적이나 주봉(장기)은 하락 추세이므로 눈높이를 낮춰 대응하십시오.\n\n"
        elif regime == "약세 추세": 
            ai_op += "• **장기 흐름:** 단기는 부진하나 주봉(장기) 추세는 견고하여 중장기 관점에선 기회일 수 있습니다.\n\n" if weekly_bullish else "• **장기 흐름:** 단기와 장기 모두 완전한 하락 추세(역배열)입니다. 보수적으로 접근하십시오.\n\n"
        elif regime == "횡보 박스": 
            ai_op += "• **장기 흐름:** 주봉(장기) 추세 상승 속 잠시 에너지를 비축하는 단기 횡보 국면입니다.\n\n" if weekly_bullish else "• **장기 흐름:** 주봉(장기) 하락세 속에서 단기적으로 지지선을 형성하며 방어 중인 모습입니다.\n\n"
        else: 
            ai_op += "• **장기 흐름:** 장기 흐름에 동조화되어 에너지가 응축/분출되는 변곡점 구간입니다.\n\n"
    
    ai_op += "💡 **[국면 맞춤형 통합 해석]**\n\n"
    if is_falling_knife: ai_op += "🚨 **[초고위험 투매 경보]** 현재 주가가 비정상적인 속도로 극심하게 급락 중인 '패닉셀' 구간입니다. 어떠한 기술적 반등 신호도 무시하고 철저히 관망할 것을 강력히 권고합니다.\n\n"
    elif res == 0: ai_op += "✨ **[신고가 랠리 분석]** 과거의 모든 악성 매물대를 소화하고 완벽한 신고가(상방 열림) 영역에 진입했습니다. 강력한 추세가 이어질 확률이 높습니다.\n\n"
    elif regime == "에너지 응축 (스퀴즈)": ai_op += "• 변동성이 극도로 응축된 상태입니다. 곧 강한 방향성 분출이 예상됩니다.\n\n"
    elif regime == "횡보 박스":
        if box_pos <= 35: ai_op += f"• 하단 지지선({sup:,.{decimals}f}{md_currency}) 부근으로 단기 매수 매력도가 높습니다.\n\n"
        elif box_pos >= 65: ai_op += f"• 상단 저항선({res:,.{decimals}f}{md_currency}) 부근으로 리스크 관리가 필요한 구간입니다.\n\n"
    elif regime == "강세 추세": ai_op += "• 매수세가 시장을 주도하는 강세장입니다. 추세 이탈 전까지 보유가 유리합니다.\n\n"
    elif regime == "상승 조정": ai_op += "• 상승 흐름 속 건전한 단기 조정(매물 소화)이 진행 중입니다.\n\n"
    elif regime == "약세 추세": ai_op += "• 하락 압력이 지배적이므로 철저한 현금 비중 관리와 보수적 접근이 필수입니다.\n\n"
    elif regime == "변동성 폭발": ai_op += f"• {'상방 대량 거래 폭발 확인. 새로운 대시세의 시작일 수 있으나 추격 매수는 신중하게 접근하십시오.' if close > prev_candle_close else '하방 대량 매도세 폭발 확인. 추가 연쇄 하락 위험이 있으므로 절대 역추세 매수를 자제하십시오.'}\n\n"
    
    ai_op += f"📊 **[수급 및 주요 레벨]**\n\n"
    ai_op += f"• **세력 수급:** 누적 수급(OBV)이 꾸준히 {'유입되며 긍정적' if obv > simple_prev_obv else '이탈하며 부정적'}인 정황이 관찰됩니다.\n\n"
    
    latest_open, latest_high, latest_low = float(latest['Open']), float(latest['High']), float(latest['Low'])
    body = abs(latest_open - close)
    if close > prev_candle_close and close > ma20 and prev_candle_close <= prev_ma20 and not is_falling_knife:
        if vol_ratio < 80 or (latest_high - max(latest_open, close)) > body * 1.5:
            ai_op += "🚨 **[가짜 상승(Bull Trap) 주의]** 저항을 돌파했으나 거래량이 부진하거나 윗꼬리가 깁니다. 섣부른 추격 매수를 자제하십시오.\n\n"
    elif close < prev_candle_close and close < ma20 and prev_candle_close >= prev_ma20:
        if vol_ratio < 70 or (min(latest_open, close) - latest_low) > body * 1.5:
            ai_op += "🚨 **[가짜 하락(Bear Trap) 주의]** 지지를 이탈했으나 하락 물량 방어 흔적(아랫꼬리)이 보입니다. 일시적 충격일 수 있습니다.\n\n"

    ai_op += "📅 **[단기 실전 대응 시나리오 가이드]**\n\n"
    if res == 0: ai_op += "• **상방 추세 시나리오:** 저항 없는 신고가 상태입니다. 추세 꺾임 시까지 수익 극대화 관점.\n\n"
    else: ai_op += f"• **상방 돌파 시나리오:** 1차 저항선인 **{res:,.{decimals}f}{md_currency}** 강하게 돌파 시 새로운 상승 추세로 판단, 매수 관점 접근.\n\n"
    ai_op += f"• **하방 방어 시나리오:** 기계적 손절 라인은 **{max(0, close - atr):,.{decimals}f}{md_currency}** 부근, 핵심 지지선은 **{sup:,.{decimals}f}{md_currency}** 입니다. 이탈 시 즉각적 리스크 관리 우선.\n\n"

    if bullish_div and regime != "약세 추세" and not is_falling_knife: 
        ai_op += "🔥 **[상승 다이버전스 포착]** 보조지표의 저점이 상승하는 긍정적 반전 시그널이 확인되었습니다!\n\n"

    comments['AI'] = f"{ai_op}🎯 **최종 투자 전략 요약:** {strategy} (AI 권장 포지션: **{pos}**)"
    return pos, strategy, comments

# ==========================================
# 3. 신규 스캐너 함수 (200일선 눌림목)
# ==========================================
def scan_200_pullback(top_n=200):
    krx_df = get_krx_data()
    if krx_df.empty: return pd.DataFrame()
    krx_df['Marcap'] = pd.to_numeric(krx_df['Marcap'], errors='coerce')
    target_stocks = krx_df.sort_values('Marcap', ascending=False).head(top_n)
    
    start_date_str = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')
    progress_bar = st.progress(0, text=f"📡 우량주 {top_n}개 종목 고속 병렬 스캔 중...")
    
    def check_stock(row_data):
        name, code = row_data['Name'], row_data['Code']
        try:
            df = fdr.DataReader(code, start=start_date_str)
            if len(df) < 210: return None
            df['MA5'] = df['Close'].rolling(5).mean()
            df['MA200'] = df['Close'].rolling(200).mean()
            latest, prev = df.iloc[-1], df.iloc[-2]
            
            # 1. 200일선 우상향 확인 (최근 10거래일 전 대비 상승 또는 수평)
            if latest['MA200'] < df['MA200'].iloc[-10]: return None
            
            # 2. 200일선 부근 지지/눌림목 확인 (전일 또는 당일 저가가 200일선의 97%~104% 사이)
            near_200 = (0.97 <= prev['Low'] / prev['MA200'] <= 1.04) or (0.97 <= latest['Low'] / latest['MA200'] <= 1.04)
            if not near_200: return None
            
            # 3. 당일 양봉 확인
            if latest['Close'] <= latest['Open']: return None
            
            # 4. 5일선 골든크로스 또는 5일선 지지 돌파
            crossed_5 = (prev['Close'] <= prev['MA5'] and latest['Close'] > latest['MA5']) or (latest['Low'] <= latest['MA5'] and latest['Close'] > latest['MA5'])
            if not crossed_5: return None
            
            disparity = (latest['Close'] / latest['MA200'] - 1) * 100
            diff_pct = ((latest['Close'] - prev['Close']) / prev['Close']) * 100
            return {
                '종목명': name,
                '종목코드': code,
                '현재가': int(latest['Close']),
                '200일선': int(latest['MA200']),
                '200일선 이격도': f"{disparity:+.1f}%",
                '당일 등락률': f"{diff_pct:+.2f}%"
            }
        except Exception:
            return None

    stock_rows = target_stocks[['Name', 'Code']].to_dict('records')
    found_stocks = []
    total = len(stock_rows)
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(check_stock, s): s for s in stock_rows}
        completed = 0
        for future in as_completed(futures):
            res = future.result()
            if res:
                found_stocks.append(res)
            completed += 1
            progress_bar.progress(completed / total, text=f"📡 고속 병렬 스캔 중... ({completed}/{total})")
            
    progress_bar.empty()
    return pd.DataFrame(found_stocks)

# ==========================================
# 4. 사이드바 및 메인 실행 UI (투트랙 메뉴 적용)
# ==========================================
with st.sidebar:
    st.header("📌 메뉴 선택")
    app_menu = st.radio("기능을 선택하세요", ["📊 단일 종목 심층 분석", "🎯 200일선 눌림목 포착"])
    st.divider()

if app_menu == "📊 단일 종목 심층 분석":
    with st.sidebar:
        st.header("⚙️ 분석 설정")
        analyze_mode = st.radio("투자 성향 설정", ["단기 스윙 (6개월 차트/일봉)", "중장기 대세 (2년 차트/주봉)"])
        st.text_input("종목명/코드 입력", placeholder="삼성전자, NVDA 등", key="search_input", on_change=on_search_input_change)
        if st.button("🚀 분석 실행", type="primary") or st.session_state.trigger_search:
            if st.session_state.search_input and not st.session_state.trigger_search:
                st.session_state.target_query = st.session_state.search_input
            st.session_state.trigger_search = False
        st.divider()
        st.subheader("🕒 최근 검색")
        for idx, item in enumerate(st.session_state.recent_searches):
            st.button(f"▪️ {item['display_name']}", key=f"rs_{idx}_{item['query']}", use_container_width=True, on_click=on_recent_click, args=(item['query'],))

    if st.session_state.target_query:
        display_name, ticker_symbol, raw_query, currency, decimals = parse_query(st.session_state.target_query)
        if {'query': raw_query, 'display_name': display_name} not in st.session_state.recent_searches:
            st.session_state.recent_searches.insert(0, {'query': raw_query, 'display_name': display_name})
            st.session_state.recent_searches = st.session_state.recent_searches[:5]
        with st.spinner(f"📡 '{display_name}' 분석 중..."):
            raw_df = get_stock_data(ticker_symbol)
        if raw_df.empty: st.error("데이터를 찾을 수 없습니다.")
        else:
            is_short_term = "단기" in analyze_mode
            time_unit = "일" if is_short_term else "주"
            chart_df_daily = calculate_indicators(raw_df.copy())
            weekly_raw = raw_df.resample('W').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
            chart_df_weekly = calculate_indicators(weekly_raw)
            weekly_bullish = None
            if not chart_df_weekly.empty and len(chart_df_weekly) >= 2:
                w_latest = chart_df_weekly.iloc[-1]
                has_w_ma60 = 'MA60' in chart_df_weekly.columns and not pd.isna(w_latest['MA60'])
                has_w_macd = ('MACD' in chart_df_weekly.columns and 'Signal' in chart_df_weekly.columns and 
                              not pd.isna(w_latest['MACD']) and not pd.isna(w_latest['Signal']))
                if has_w_ma60 and has_w_macd:
                    weekly_bullish = (w_latest['Close'] > w_latest['MA60']) and (w_latest['MACD'] > w_latest['Signal'])
                elif has_w_macd:
                    weekly_bullish = w_latest['MACD'] > w_latest['Signal']
            chart_df = chart_df_daily if is_short_term else chart_df_weekly
            default_days = 180 if is_short_term else 730 
            cur_price = raw_df['Close'].iloc[-1]
            diff = cur_price - raw_df['Close'].iloc[-2] if len(raw_df) > 1 else 0
            st.subheader(f"📑 {display_name} 리포트")
            st.metric("현재 주가", f"{cur_price:,.{decimals}f} {currency}", f"{diff:,.{decimals}f} {currency}")
            q_score = calculate_quant_score(chart_df, is_short_term)
            st.write(f"### 💯 퀀트 스코어: **{q_score}점**")
            st.progress(q_score / 100)
            pts, sup, res = detect_patterns_and_levels(chart_df)
            if len(chart_df) < 5: st.warning("분석에 필요한 데이터가 부족합니다 (최소 5거래일 이상 필요).")
            else:
                pos, strat, comments = generate_detailed_opinions(chart_df, sup, res, currency, decimals, is_short_term, time_unit, q_score, weekly_bullish)
                c1, c2 = st.columns(2)
                with c1:
                    with st.container(border=True):
                        st.markdown("### 🎯 **종합 전략**")
                        st.warning(f"**포지션:** {pos}\n\n**의견:** {strat}")
                with c2:
                    with st.container(border=True):
                        st.markdown("### 🔍 **지지/저항 레벨**")
                        md_curr_ui = currency.replace('$', r'\$')
                        sup_txt = f"{sup:,.{decimals}f} {md_curr_ui}" if sup > 0 else "데이터 부족"
                        res_txt = "✨ 신고가 (저항 없음)" if res == 0 else (f"{res:,.{decimals}f} {md_curr_ui}" if res > 0 else "데이터 부족")
                        st.write(f"🛡️ **지지선:** {sup_txt} | 🚧 **저항선:** {res_txt}")
                        if pts:
                            st.info(f"🕯️ **포착된 캔들 패턴:** {' | '.join(pts)}")
                with st.expander("🔬 지표별 상세 분석", expanded=True):
                    desc = {"ADX 추세강도": "ADX: 추세 파워 측정.", "상대 거래량": "Relative Vol: 거래량 비율.", "OBV 누적": "OBV: 세력 매집 지표.", "RSI 강도": "RSI: 과열/침체 수치.", "MACD 흐름": "MACD: 추세 방향 파악.", "ATR 변동성": "ATR: 실질 변동폭."}
                    for label, key in [("ADX 추세강도", "ADX"), ("상대 거래량", "VOL"), ("OBV 누적", "OBV"), ("RSI 강도", "RSI"), ("MACD 흐름", "MACD"), ("ATR 변동성", "ATR")]:
                        cl, cv = st.columns([0.25, 0.75])
                        with cl.popover(label, use_container_width=True): st.info(desc.get(label))
                        cv.markdown(comments.get(key, '데이터 없음'))
                    st.divider()
                    st.info(comments.get('AI'))
                tab1, tab2 = st.tabs(["📈 차트", "📊 수급(OBV)"])
                f_start = max(chart_df.index[0], datetime.now() - timedelta(days=default_days))
                p_df = chart_df[chart_df.index >= f_start].copy()
                with tab1:
                    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.55, 0.20, 0.25], vertical_spacing=0.03)
                    fig.add_trace(go.Candlestick(x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name='주가'), row=1, col=1)
                    for ma, clr in [('MA20', 'orange'), ('MA60', 'green')]: fig.add_trace(go.Scatter(x=p_df.index, y=p_df[ma], name=ma, line=dict(color=clr, width=1)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['RSI'], name='RSI', line=dict(color='#00BFFF', width=1.5)), row=2, col=1)
                    colors = ['#ff3333' if c >= o else '#3366ff' for c, o in zip(p_df['Close'], p_df['Open'])]
                    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Volume'], name='거래량', marker_color=colors), row=3, col=1)
                    fig.update_layout(height=600, margin=dict(t=10, b=10, l=0, r=0), hovermode='x unified', showlegend=False)
                    fig.update_xaxes(rangeslider=dict(visible=False))
                    if is_short_term:
                        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                with tab2:
                    if 'OBV' in p_df.columns:
                        ofig = go.Figure(data=[go.Scatter(x=p_df.index, y=p_df['OBV'], fill='tozeroy', line=dict(color='purple'))])
                        ofig.update_layout(height=350, margin=dict(t=10, b=10, l=0, r=0))
                        if is_short_term:
                            ofig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
                        st.plotly_chart(ofig, use_container_width=True)
    else: 
        st.info("👈 사이드바에서 종목을 검색하여 분석을 시작하세요.")

elif app_menu == "🎯 200일선 눌림목 포착":
    st.subheader("🎯 200일선 철벽 방어 우량주 스캐너")
    st.markdown("외국인과 기관이 방어하는 1등 주식의 '최후의 보루'를 찾아냅니다.")
    scan_lim = st.selectbox("스캔 범위 설정 (시총 상위)", [100, 200, 300], index=1)
    if st.button("🚀 스캐너 작동", type="primary", use_container_width=True):
        res_df = scan_200_pullback(top_n=scan_lim)
        st.divider()
        if not res_df.empty:
            st.success(f"🎉 {len(res_df)}개의 종목을 포착했습니다.")
            if '현재가' in res_df.columns:
                res_df['현재가'] = res_df['현재가'].apply(lambda x: f"{x:,} 원" if isinstance(x, (int, float)) else str(x))
            if '200일선' in res_df.columns:
                res_df['200일선'] = res_df['200일선'].apply(lambda x: f"{x:,} 원" if isinstance(x, (int, float)) else str(x))
            st.dataframe(res_df, use_container_width=True, hide_index=True)
        else: st.warning("조건에 일치하는 종목이 없습니다.")
