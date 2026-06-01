import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import FinanceDataReader as fdr
from datetime import datetime, timedelta
import numpy as np

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

# 🌟 모바일 및 데스크톱 가독성 향상을 위해 글자 포인트 및 줄간격 스케일업 반영
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
    /* 분석의견 등 마크다운 내 본문/리스트 글자 포인트 확대 적용 */
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
# 2. 공통 데이터 처리 함수 (외풍 방어막 및 가상 흑자 필터)
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
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
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
    df['Vol_Ratio'] = (df['Volume'] / (df['Vol_MA5'] + 1e-10)) * 100
    
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
            if latest['RSI'] < 30: score += 25
            elif latest['RSI'] < 50: score += 15
            elif latest['RSI'] < 70: score += 5
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
    patterns = []
    body = abs(latest['Open'] - latest['Close'])
    lower_shadow, upper_shadow = min(latest['Open'], latest['Close']) - latest['Low'], latest['High'] - max(latest['Open'], latest['Close'])
    
    if lower_shadow > body * 2 and upper_shadow < body: patterns.append("🔨 망치형 (바닥권 반등 신호)")
    if len(df) >= 2 and latest['Close'] > latest['Open'] and latest['Close'] > df['High'].iloc[-2]: patterns.append("🚀 상승 장악형 (추세 전환)")
    
    lookback = min(61, len(df))
    past_df = df.iloc[-lookback:-1] if lookback > 1 else df.iloc[:-1]
    if past_df.empty: return patterns, latest['Close'] * 0.95, latest['Close'] * 1.05
    
    closes = past_df['Close']
    tolerance = closes.mean() * 0.02
    
    def cluster_levels(price_series):
        prices = sorted(price_series.tolist())
        clusters = []
        for p in prices:
            matched = False
            for c in clusters:
                if abs(p - c['center']) <= tolerance:
                    c['prices'].append(p)
                    c['center'] = sum(c['prices']) / len(c['prices'])
                    matched = True
                    break
            if not matched: clusters.append({'center': p, 'prices': [p]})
        clusters.sort(key=lambda x: len(x['prices']), reverse=True)
        return clusters

    low_mask = (closes < closes.shift(1)) & (closes < closes.shift(-1))
    support_candidates = closes[low_mask]
    resistance_candidates = closes[(closes > closes.shift(1)) & (closes > closes.shift(-1))]

    if len(support_candidates) >= 2:
        sup_clusters = cluster_levels(support_candidates)
        valid_sup = [c for c in sup_clusters if c['center'] <= latest['Close']]
        support = valid_sup[0]['center'] if valid_sup else (closes[closes <= latest['Close']].min() if not closes[closes <= latest['Close']].empty else closes.min())
    else:
        below = closes[closes <= latest['Close']]
        support = below.min() if not below.empty else closes.min()

    above = closes[closes > latest['Close']] 
    if above.empty: resistance = 0  
    else:
        if len(resistance_candidates) >= 2:
            res_clusters = cluster_levels(resistance_candidates)
            valid_res = [c for c in res_clusters if c['center'] > latest['Close']]
            resistance = valid_res[0]['center'] if valid_res else above.max()
        else: resistance = above.max()

    return patterns, support, resistance

def generate_detailed_opinions(df, sup, res, currency, decimals, is_short_term, time_unit, q_score, weekly_bullish=None):
    md_currency = currency.replace('$', r'\$')
    latest, prev = df.iloc[-1], df.iloc[-2]
    close, rsi, obv, vol_ratio, atr = map(float, [latest['Close'], latest['RSI'], latest['OBV'], latest['Vol_Ratio'], latest['ATR']])
    ma20, ma60, adx, p_di, m_di = map(float, [latest['MA20'], latest['MA60'], latest['ADX'], latest['+DI'], latest['-DI']])
    
    prev_candle_close = float(prev['Close'])
    prev_ma20 = float(prev['MA20']) if not pd.isna(prev['MA20']) else ma20
    
    simple_lookback = min(5, len(df) - 1) if len(df) > 1 else 1
    obv_lookback = simple_lookback if is_short_term else min(13, len(df) - 1)
    simple_prev_obv = float(df['OBV'].iloc[-obv_lookback])
    
    swing_lookback = min(60, len(df))
    past_df = df.iloc[-swing_lookback:]
    local_min_mask = (past_df['Close'] < past_df['Close'].shift(1)) & (past_df['Close'] < past_df['Close'].shift(2)) & \
                     (past_df['Close'] < past_df['Close'].shift(-1)) & (past_df['Close'] < past_df['Close'].shift(-2))
    local_min_df = past_df[local_min_mask]
    
    bullish_div = False
    if len(local_min_df) >= 2:
        t1, t2 = local_min_df.iloc[-2], local_min_df.iloc[-1]
        if t2['Close'] < t1['Close'] and (t2['OBV'] > t1['OBV'] or t2['RSI'] > t1['RSI']): bullish_div = True

    # 🌟 국면 판단 논리 교정 공식 이식 (하이브 연속 하락 패턴 방어공식 이식)
    is_squeeze = latest['BBW'] <= df['BBW'].iloc[-120:].min() * 1.05 if len(df) > 120 else False
    if is_squeeze: regime = "에너지 응축 (스퀴즈)"
    elif vol_ratio >= 150 and adx > df['ADX'].iloc[-2] and adx > 20: regime = "변동성 폭발"
    elif adx < 25: regime = "횡보 박스"
    elif ma20 >= ma60 and close >= ma60: 
        if close >= ma20: regime = "강세 추세" if p_di > m_di else "상승 조정"
        else: regime = "단기 약세 전환"
    else: regime = "약세 추세"

    box_pos = (close - sup) / (res - sup) * 100 if res > sup else 100
    drop_pct = ((prev['Close'] - close) / prev['Close'] * 100) if prev['Close'] > 0 else 0
    is_falling_knife = (drop_pct >= 7.0 and vol_ratio >= 120) or (drop_pct >= 10.0)

    macd_diff = float(latest['MACD'] - latest['Signal']) if not pd.isna(latest['MACD']) and not pd.isna(latest['Signal']) else 0
    vol_pct = (atr / close) * 100 if close > 0 else 0

    comments = {}
    comments['ADX'] = f"현재 ADX 추세강도 지수는 **{adx:.1f}**이며, 알고리즘은 현재 시장을 **[{regime}]** 국면으로 확정했습니다."
    
    if regime == "에너지 응축 (스퀴즈)":
        comments['RSI'] = f"RSI({rsi:.1f}): 볼린저 밴드 수축 국면이므로 RSI의 움직임이 매우 둔화되어 있습니다. 방향성 탐색 중입니다."
        comments['MACD'] = f"MACD({macd_diff:,.{decimals}f}): 이동평균선이 밀집하며 MACD도 0선에 완전히 수렴했습니다."
    elif regime == "횡보 박스":
        comments['RSI'] = f"RSI({rsi:.1f}): 횡보장에서는 RSI의 신뢰도가 가장 높습니다. " + ("박스권 하단 지지선(과매도) 터치로 기술적 반등이 예상됩니다." if rsi <= 40 else "박스권 상단 저항선(과매수) 도달로 조정이 예상됩니다." if rsi >= 60 else "박스권 중간에서 뚜렷한 방향성을 탐색 중입니다.")
        comments['MACD'] = f"MACD({macd_diff:,.{decimals}f}): 뚜렷한 추세가 부재한 박스권이므로 MACD 크로스 신호의 신뢰도는 다소 떨어집니다."
    elif regime == "강세 추세":
        comments['RSI'] = f"RSI({rsi:.1f}): 강세장에서는 지표가 쉽게 과열권에 진입합니다. " + ("강한 매수세로 단기 과열(70 이상) 상태이나 추세는 굳건합니다." if rsi >= 70 else "상승 추세 중 발생한 건전한 눌림목(조정) 타점입니다." if rsi <= 50 else "안정적인 상승 탄력을 유지하고 있습니다.")
        comments['MACD'] = f"MACD({macd_diff:,.{decimals}f}): 상승 모멘텀이 강하게 유지되며 이평선 정배열 확장을 지지하고 있습니다."
    elif regime == "상승 조정": 
        comments['RSI'] = f"RSI({rsi:.1f}): 상승 추세 속에서 조정을 받으며 지표가 식어가고 있습니다. 40~50 부근에서 지지받는지 확인이 필요합니다."
        comments['MACD'] = f"MACD({macd_diff:,.{decimals}f}): 단기적으로 데드크로스가 발생하거나 모멘텀이 둔화되었으나, 장기 상승 추세 베이스는 훼손되지 않았습니다."
    elif regime == "단기 약세 전환":
        comments['RSI'] = f"RSI({rsi:.1f}): 주가가 단기 지지선(20일선)을 이탈함에 따라 투심이 빠르게 냉각되고 있습니다."
        comments['MACD'] = f"MACD({macd_diff:,.{decimals}f}): 지표가 시그널선을 하향 돌파한 뒤 하락 확산세를 보이며 단기 매도 압력이 가중되고 있습니다."
    elif regime == "약세 추세":
        comments['RSI'] = f"RSI({rsi:.1f}): 약세장에서는 지표가 지속적으로 침체권에 머눕니다. " + ("일시적인 기술적 반등 구간으로 매도를 고려할 시점입니다." if rsi >= 55 else "극단적 과매도 상태이나, 지속적인 하락 압력을 받고 있으므로 섣부른 진입은 피해야 합니다." if rsi <= 30 else "지속적인 하락 압력을 받고 있습니다.")
        comments['MACD'] = f"MACD({macd_diff:,.{decimals}f}): 하락 모멘텀이 강하며, 추세 반전을 암시하는 뚜렷한 신호가 아직 없습니다."
    elif regime == "변동성 폭발":
        comments['RSI'] = f"RSI({rsi:.1f}): 변동성 폭발로 인해 투심이 한쪽으로 극단적으로 쏠리는 오버슈팅 및 투매 국면입니다."
        comments['MACD'] = f"MACD({macd_diff:,.{decimals}f}): 단기 모멘텀이 평소의 범위를 벗어나 급격하게 방향성을 분출하고 있습니다."

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
        elif regime == "단기 약세 전환":
            if box_pos <= 35 or bullish_div: pos, strategy = "🟠 단기 기술적 반등 공략", "단기 지지선은 깨졌으나 하방 구조적 지지선에 도달했습니다. 반등 신호를 노린 짧은 트레이딩이 가능합니다."
            else: pos, strategy = "⚖️ 단기 관망", "주가가 20일선을 깨고 밀리고 있습니다. 60일 장기 지지선 부근까지 추가 낙폭을 열어두고 확실한 바닥을 다질 때까지 관망하십시오."
        elif regime == "약세 추세":
            if rsi >= 45 and close > prev['Close']:
                if obv > simple_prev_obv and vol_ratio > 100: pos, strategy = "🟠 의미 있는 반등 시도", "하락장 속 유의미한 수급/거래량 동반 반등. 추세 전환의 단초가 될 수 있습니다."
                else: pos, strategy = "🔵 데드캣 바운스 경계 (매도)", "수급 뒷받침이 부족한 단순 기술적 반등(속임수)일 확률이 높습니다. 탈출 기회로 삼으십시오."
            elif rsi <= 30 or bullish_div: pos, strategy = "🟠 단기 기술적 반등 공략", "극단적 과매도 및 다이버전스 발생. 짧은 수익을 목표로 한 기술적 반등 매매만 권장합니다."
            else: pos, strategy = "🔷 적극 매도 및 관망", "하락 추세가 지배적입니다. 물타기를 자제하고 현금 비중을 높여 관망하십시오."
        else: pos, strategy = "🔴 돌파 추세 추종", "평균을 상회하는 대량 거래와 상방 돌파 발생. 새로운 추세 형성 초입으로 적극 추종이 유리합니다."
    else:
        if regime == "상승 조정" and (box_pos > 50 or obv < simple_prev_obv): pos, strategy = "⚖️ 장기 눌림목 대기", "장기 상승장 내 조정 구간이나, 하락세 진정 및 지지선 확인 전까지 보수적 관망을 권장합니다."
        elif regime == "단기 약세 전환": pos, strategy = "⚖️ 장기 지지선 테스트", "장기 상승추세의 이평 정배열 마지노선인 60일선 지지력을 시험하는 구간입니다. 보수적 진입을 권장합니다."
        elif regime iN ["강세 추세", "변동성 폭발", "상승 조정"]: pos, strategy = "🔴 비중 확대 (장기)", "대세 상승장에 진입했습니다. 장기적 시각에서 비중 확대 및 홀딩 전략이 유효합니다."
        elif regime == "약세 추세" and rsi < 30: pos, strategy = "🟠 저점 분할 매집", "역사적 저평가 구간 진입. 펀더멘털 확인 후 긴 호흡으로 1차 분할 매집을 고려할 수 있습니다."
        elif regime == "약세 추세": pos, strategy = "🔷 비중 축소 (장기)", "대세 하락장이 지속 중입니다. 포트폴리오 방어를 위해 주식 비중 축소를 권장합니다."
        else: pos, strategy = "⚖️ 장기 관망", "장기 추세의 변곡점이거나 방향성이 불분명한 구간입니다. 확실한 추세 형성 시까지 관망하십시오."

    # 🌟 이스케이프가 중복 처리되지 않도록 마크다운 표준 문법 정렬 (줄바꿈 원상 복구)
    mode_str = "단기 스윙" if is_short_term else "장기 가치투자"
    ai_op = f"🤖 **StockMap AI {mode_str} 심층 진단 리포트**\n\n"
    ai_op += f"🔍 **[시장 국면 분류]**\n\n• 현재 해당 종목은 **[{regime}]** 국면에 위치해 있습니다.\n\n"
    
    if is_short_term and weekly_bullish is not None:
        ai_op += f"⏱️ **[MTF 다중 시간대 분석]**\n\n"
        if regime in ["강세 추세", "상승 조정"]: 
            ai_op += "• **장기 흐름:** 주봉(장기) 상승세가 굳건하여 일봉(단기) 수준에서도 강한 지지력을 보입니다.\n\n" if weekly_bullish else "• **장기 흐름:** 단기는 긍정적이나 주봉(장기)은 하락 추세이므로 눈높이를 낮춰 대응하십시오.\n\n"
        elif regime == "단기 약세 전환":
            ai_op += "• **장기 흐름:** 일봉상 단기 파동은 훼손되었으나, 주봉 기준 대세 상승선은 깨지지 않았습니다. 든든한 하단 지지 매물대 근처 진입 기회를 엿볼 수 있습니다.\n\n" if weekly_bullish else "• **장기 흐름:** 단기와 장기 대세 시그널이 동시에 하방 압력을 호소하고 있으므로 대단히 보수적으로 방어해야 합니다.\n\n"
        elif regime == "약세 추세": 
            ai_op += "• **장기 흐름:** 단기는 부진하나 주봉(장기) 추세는 견고하여 중장기 관점에선 기회일 수 있습니다.\n\n" if weekly_bullish else "• **장기 흐름:** 단기와 장기 모두 완전한 하락 추세(역배열)입니다. 보수적으로 접근하십시오.\n\n"
        elif regime == "횡보 박스": 
            ai_op += "• **장기 흐름:** 주봉(장기) 추세 상승 속 잠시 에너지를 비축하는 단기 횡보 국면입니다.\n\n" if weekly_bullish else "• **장기 흐름:** 주봉(장기) 하락세 속에서 단기적으로 지지선을 형성하며 방어 중인 모습입니다.\n\n"
        else: 
            ai_op += "• **장기 흐름:** 장기 흐름에 동조화되어 에너지가 응축/분출되는 변곡점 구간입니다.\n\n"
    
    ai_op += f"💡 **[국면 맞춤형 통합 해석]**\n\n"
    if is_falling_knife: ai_op += "🚨 **[초고위험 투매 경보]** 현재 주가가 비정상적인 속도로 극심하게 급락 중인 '패닉셀' 구간입니다. 어떠한 기술적 반등 신호도 무시하고 철저히 관망할 것을 강력히 권고합니다.\n\n"
    elif res == 0: ai_op += "✨ **[신고가 랠리 분석]** 과거의 모든 악성 매물대를 소화하고 완벽한 신고가(상방 열림) 영역에 진입했습니다. 강력한 추세가 이어질 확률이 높습니다.\n\n"
    elif regime == "에너지 응축 (스퀴즈)": ai_op += "• 변동성이 극도로 응축된 상태입니다. 곧 강한 방향성 분출이 예상됩니다.\n\n"
    elif regime == "횡보 박스":
        if box_pos <= 35: ai_op += f"• 하단 지지선({sup:,.{decimals}f}{md_currency}) 부근으로 단기 매수 매력도가 높습니다.\n\n"
        elif box_pos >= 65 and res > 0: 
            if obv > simple_prev_obv: ai_op += f"• 현재 주가가 상단 저항선 부근에 위치했으나, 긍정적인 수급이 유입되고 있어 돌파 가능성을 예의주시할 필요가 있습니다.\n\n"
            else: ai_op += f"• 현재 주가가 상단 저항선 부근에 위치하여 단기 차익 실현 및 비중 축소를 고려해야 할 구간입니다.\n\n"
    elif regime == "강세 추세": ai_op += "• 매수세가 시장을 주도하는 강세장입니다. 추세 이탈 전까지 보유가 유리합니다.\n\n"
    elif regime == "상승 조정": ai_op += "• 상승 흐름 속 건전한 단기 조정(매물 소화)이 진행 중입니다.\n\n"
    elif regime == "단기 약세 전환": ai_op += "• 가파른 직전 랠리를 뒤로하고 생명선(20일선)이 무너지며 위험 단기 조정에 진입했습니다. 무조건적인 매수보단 관망 시점입니다.\n\n"
    elif regime == "약세 추세": ai_op += "• 하락 압력이 지배적이므로 철저한 현금 비중 관리와 보수적 접근이 필수입니다.\n\n"
    
    ai_op += f"📊 **[수급 및 주요 레벨]**\n\n"
    ai_op += f"• **세력 수급:** 누적 수급(OBV)이 꾸준히 {'유입되며 긍정적' if obv > simple_prev_obv else '이탈하며 부정적'}인 정황이 관찰됩니다.\n\n"
    
    if close > prev_candle_close and close > ma20 and prev_candle_close <= prev_ma20 and not is_falling_knife:
        if vol_ratio < 80 or (latest_high - max(latest_open, close)) > body * 1.5:
            ai_op += "🚨 **[가짜 상승(Bull Trap) 주의]** 저항을 돌파했으나 거래량이 부진하거나 윗꼬리가 깁니다. 섣부른 추격 매수를 자제하십시오.\n\n"
    elif close < prev_candle_close and close < ma20 and prev_candle_close >= prev_ma20:
        if vol_ratio < 70 or (min(latest_open, close) - latest_low) > body * 1.5:
            ai_op += "🚨 **[가짜 하락(Bear Trap) 주의]** 지지를 이탈했으나 하락 물량 방어 흔적(아랫꼬리)이 보입니다. 일시적 충격일 수 있습니다.\n\n"

    playbook_text = f"📅 **[단기 실전 대응 시나리오 가이드]**\n\n"
    if res == 0: playbook_text += f"• **상방 추세 시나리오:** 저항 없는 신고가 상태입니다. 추세 꺾임 시까지 수익 극대화 관점.\n\n"
    else: playbook_text += f"• **상방 돌파 시나리오:** 1차 저항선인 **{res:,.{decimals}f}{md_currency}** 강하게 돌파 시 새로운 상승 추세로 판단, 매수 관점 접근.\n\n"
    playbook_text += f"• **하방 방어 시나리오:** 기계적 손절 라인은 **{max(0, close - atr):,.{decimals}f}{md_currency}** 부근, 핵심 지지선은 **{sup:,.{decimals}f}{md_currency}** 입니다. 이탈 시 즉각적 리스크 관리 우선.\n\n"
    ai_op += playbook_text

    if bullish_div and regime != "약세 추세" and not is_falling_knife: 
        ai_op += "🔥 **[상승 다이버전스 포착]** 보조지표의 저점이 상승하는 긍정적 반전 시그널이 확인되었습니다!\n\n"

    comments['AI'] = f"{ai_op}🎯 **최종 투자 전략 요약:** {strategy} (AI 권장 포지션: **{pos}**)"
    return pos, strategy, comments

# ==========================================
# 3. 신규 스캐너 함수 (200일선 눌림목)
# ==========================================
def scan_200_pullback(top_n=200):
    krx_df = get_krx_data()
    if krx_df.empty:
        return pd.DataFrame()
        
    krx_df['Marcap'] = pd.to_numeric(krx_df['Marcap'], errors='coerce')
    target_stocks = krx_df.sort_values('Marcap', ascending=False).head(top_n)
    
    found_stocks = []
    progress_bar = st.progress(0, text="📡 우량주 차트 데이터 수집 및 조건 스캔 중...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=400) 
    
    for i, (idx, row) in enumerate(target_stocks.iterrows()):
        progress_bar.progress((i + 1) / top_n, text=f"📡 스캔 중... ({i+1}/{top_n}) - {row['Name']}")
        try:
            df = fdr.DataReader(row['Code'], start=start_date.strftime('%Y-%m-%d'))
            if len(df) < 210: continue
            
            df['MA5'] = df['Close'].rolling(5).mean()
            df['MA200'] = df['Close'].rolling(200).mean()
            
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            ma200_diff = df['MA200'].diff().tail(10)
            if (ma200_diff <= 0).any(): continue
            
            if not (0.98 <= prev['Low'] / prev['MA200'] <= 1.03): continue
            
            if latest['Close'] <= latest['Open']: continue
            
            if not (prev['Close'] <= prev['MA5'] and latest['Close'] > latest['MA5']): continue
            
            found_stocks.append({
                '종목명': row['Name'], 
                '종목코드': row['Code'], 
                '현재가': int(latest['Close']),
                '200일선': int(latest['MA200'])
            })
        except Exception:
            pass
            
    progress_bar.empty()
    return pd.DataFrame(found_stocks)

# ==========================================
# 4. 사이드바 및 메인 실행 UI (투트랙 메뉴 적용)
# ==========================================
with st.sidebar:
    st.header("📌 메뉴 선택")
    # 🌟 요구사항 반영: ' (NEW)' 텍스트 완벽 제거 및 스캐너 독립 탑재
    app_menu = st.radio("기능을 선택하세요", ["📊 단일 종목 심층 분석", "🎯 200일선 눌림목 포착"])
    st.divider()

if app_menu == "📊 단일 종목 심층 분석":
    with st.sidebar:
        st.header("⚙️ 분석 설정")
        analyze_mode = st.radio("투자 성향 설정", ["단기 스윙 (6개월 차트/일봉)", "중장기 대세 (2년 차트/주봉)"])
        
        new_query = st.text_input("종목명/코드 입력", placeholder="삼성전자, NVDA 등", key="search_input", on_change=on_search_input_change)
        
        if st.button("🚀 분석 실행", type="primary") or st.session_state.trigger_search:
            if st.session_state.search_input and not st.session_state.trigger_search:
                st.session_state.target_query = st.session_state.search_input
            st.session_state.trigger_search = False
        
        st.markdown("""<div class="style-box"><b>🔍 분석 모드 가이드</b><br>• <b>단기 스윙</b>: 최근 6개월 일봉 파동 파악.<br>• <b>중장기 대세</b>: 최근 2년 주봉 대세 판별.</div>""", unsafe_allow_html=True)
        st.divider()
        st.subheader("🕒 최근 검색")
        for idx, item in enumerate(st.session_state.recent_searches):
            st.button(f"▪️ {item['display_name']}", key=f"rs_{idx}_{item['query']}", use_container_width=True, on_click=on_recent_click, args=(item['query'],))

    if st.session_state.target_query:
        display_name, ticker_symbol, raw_query, currency, decimals = parse_query(st.session_state.target_query)
        if {'query': raw_query, 'display_name': display_name} not in st.session_state.recent_searches:
            st.session_state.recent_searches.insert(0, {'query': raw_query, 'display_name': display_name})
            st.session_state.recent_searches = st.session_state.recent_searches[:5]

        with st.spinner(f"📡 '{display_name}' 심층 리포트를 분석 중입니다..."):
            raw_df = get_stock_data(ticker_symbol)
            
        if raw_df.empty: st.error("해당 종목의 데이터를 찾을 수 없습니다.")
        else:
            is_short_term = "단기" in analyze_mode
            time_unit = "일" if is_short_term else "주"
            chart_df_daily = calculate_indicators(raw_df.copy())
            weekly_raw = raw_df.resample('W').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
            chart_df_weekly = calculate_indicators(weekly_raw)
            
            weekly_bullish = None
            if not chart_df_weekly.empty:
                w_latest = chart_df_weekly.iloc[-1]
                weekly_bullish = (w_latest['Close'] > w_latest['MA60']) and (w_latest['MACD'] > w_latest['Signal'])

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

            if len(chart_df) < 2: st.warning("분석을 위한 과거 데이터가 충분하지 않습니다.")
            else:
                pos, strat, comments = generate_detailed_opinions(chart_df, sup, res, currency, decimals, is_short_term, time_unit, q_score, weekly_bullish)
                col1, col2 = st.columns(2)
                with col1:
                    with st.container(border=True):
                        st.markdown("### 🎯 **종합 전략**")
                        st.warning(f"**포지션:** {pos}\n\n**의견:** {strat}")
                with col2:
                    with st.container(border=True):
                        st.markdown("### 🔍 **차트 패턴 및 지지/저항 레벨**")
                        p_text = ", ".join(pts) if pts else "포착된 특이 패턴이 없습니다."
                        st.write(f"📍 **패턴:** {p_text}")
                        md_currency_ui = currency.replace('$', r'\$')
                        sup_text = f"{sup:,.{decimals}f} {md_currency_ui}" if sup > 0 else "데이터 부족"
                        if res == 0: res_text = "✨ 신고가 돌파 (저항 없음)"
                        elif res > 0: res_text = f"{res:,.{decimals}f} {md_currency_ui}"
                        else: res_text = "데이터 부족"
                        st.write(f"🛡️ **주요 지지선:** {sup_text} | 🚧 **주요 저항선:** {res_text}")

                with st.expander("🔬 지표별 상세 수치 분석 (용어를 클릭하시면 설명이 나타납니다)", expanded=True):
                    desc = {
                        "ADX 추세강도": "**ADX**\n\n추세의 '파워' 자체를 측정합니다. 25 이상이면 강한 추세.",
                        "상대 거래량": "**상대 거래량**\n\n최근 5일 평균 대비 현재 거래량의 비율입니다.",
                        "OBV 누적": "**OBV**\n\n세력 매집 판단 지표입니다.",
                        "RSI 강도": "**RSI**\n\n과열/침체를 수치화한 지표 (70이상 과매수, 30이하 과매도).",
                        "MACD 흐름": "**MACD**\n\n이평선의 차이를 이용해 추세 방향 파악.",
                        "ATR 변동성": "**ATR**\n\n실질적인 주가 변동폭 평균."
                    }
                    for label, key in [("ADX 추세강도", "ADX"), ("상대 거래량", "VOL"), ("OBV 누적", "OBV"), ("RSI 강도", "RSI"), ("MACD 흐름", "MACD"), ("ATR 변동성", "ATR")]:
                        col_lbl, col_val = st.columns([0.25, 0.75])
                        with col_lbl.popover(label, use_container_width=True): st.info(desc.get(label))
                        col_val.markdown(comments.get(key, '데이터 없음'))
                    st.divider()
                    st.info(comments.get('AI'))

                tab1, tab2 = st.tabs(["📈 차트 & 보조지표", "📊 수급 에너지(OBV)"])
                final_start_date = max(chart_df.index[0], datetime.now() - timedelta(days=default_days))
                plot_df = chart_df[chart_df.index >= final_start_date].copy()
                
                if not plot_df.empty:
                    c_min, c_max = plot_df[['Low', 'MA20', 'MA60']].min().min(), plot_df[['High', 'MA20', 'MA60']].max().max()
                    if pd.isna(c_min) or c_min == c_max: c_min, c_max = plot_df['Low'].min(), plot_df['High'].max()
                    padding = (c_max - c_min) * 0.05
                    y_range = [c_min - padding, c_max + padding]
                else: y_range = None

                with tab1:
                    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.55, 0.20, 0.25], vertical_spacing=0.03)
                    fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name='주가'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BB_Upper'], name='BB상단', line=dict(color='rgba(173, 216, 230, 0.4)', width=1, dash='dot')), row=1, col=1)
                    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BB_Lower'], name='BB하단', line=dict(color='rgba(173, 216, 230, 0.4)', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(173, 216, 230, 0.1)'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA20'], name=f'MA20', line=dict(color='orange', width=1)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA60'], name=f'MA60', line=dict(color='green', width=1)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['RSI'], name='RSI', line=dict(color='#00BFFF', width=1.5)), row=2, col=1)
                    fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=1, row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=1, row=2, col=1)
                    fig.add_hrect(y0=30, y1=70, fillcolor="gray", opacity=0.1, line_width=0, row=2, col=1)
                    colors = ['#ff3333' if c >= o else '#3366ff' for c, o in zip(plot_df['Close'], plot_df['Open'])]
                    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['Volume'], name='거래량', marker_color=colors), row=3, col=1)
                    fig.update_layout(height=600, margin=dict(t=10, b=10, l=0, r=0), dragmode=False, hovermode='x unified', showlegend=False)
                    fig.update_xaxes(rangeslider=dict(visible=False), fixedrange=True)
                    fig.update_yaxes(range=y_range, fixedrange=True, row=1, col=1)
                    fig.update_yaxes(range=[0, 100], fixedrange=True, row=2, col=1)
                    fig.update_yaxes(fixedrange=True, row=3, col=1)
                    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': False, 'displayModeBar': False})
                    
                with tab2:
                    if 'OBV' in plot_df.columns:
                        obv_fig = go.Figure(data=[go.Scatter(x=plot_df.index, y=plot_df['OBV'], name='OBV', fill='tozeroy', line=dict(color='purple'))])
                        obv_fig.update_layout(height=350, margin=dict(t=10, b=10, l=0, r=0), dragmode=False, hovermode='x unified')
                        obv_fig.update_xaxes(range=[final_start_date, datetime.now()], fixedrange=True)
                        obv_fig.update_yaxes(fixedrange=True)
                        st.plotly_chart(obv_fig, use_container_width=True, config={'scrollZoom': False, 'displayModeBar': False})
    else:
        st.info("👈 사이드바에서 종목을 검색하여 분석을 시작하세요.")

# ==========================================
# 5. 신규 메뉴: 200일선 눌림목 포착 스캐너
# ==========================================
elif app_menu == "🎯 200일선 눌림목 포착":
    st.subheader("🎯 200일선 철벽 방어 우량주 스캐너")
    st.markdown("""
    **외국인과 기관이 방어하는 1등 주식의 '최후의 보루'를 찾아냅니다.**
    * **조건 A, B:** 상장폐지 위험 최소화 (시가총액 상위 우량주 스캔)
    * **조건 C:** 대세 상승장 확인 (200일선 10일 연속 상승)
    * **조건 E:** 정확한 지지 확인 (1봉전 저가가 200일선의 98% ~ 103% 이내)
    * **조건 G, H:** 완벽한 턴어라운드 타점 (오늘 양봉 & 종가 5일선 상향 돌파)
    """)
    
    scan_limit = st.selectbox("스캔 범위 설정 (시가총액 상위 기준)", [100, 200, 300], index=1, help="스캔 범위가 넓을수록 탐색 시간이 오래 걸립니다.")
    
    if st.button("🚀 스캐너 작동 (우량주 조건 탐색)", type="primary", use_container_width=True):
        krx_check = get_krx_data()
        if krx_check.empty:
            st.error("⚠️ 한국거래소(KRX) 서버 통신 지연으로 실시간 스캔을 시작할 수 없습니다. 잠시 후 다시 시도해 주세요.")
        else:
            result_df = scan_200_pullback(top_n=scan_limit)
            
            st.divider()
            if not result_df.empty:
                st.success(f"🎉 축하합니다! 완벽한 눌림목 타점 종목 {len(result_df)}개를 포착했습니다.")
                result_df['현재가'] = result_df['현재가'].apply(lambda x: f"{x:,} 원")
                result_df['200일선'] = result_df['200일선'].apply(lambda x: f"{x:,} 원")
                
                st.dataframe(result_df, use_container_width=True, hide_index=True)
                st.info("💡 위 종목 코드를 복사하여 좌측 메뉴의 **[단일 종목 심층 분석]**에서 상세 전략을 확인해 보세요!")
            else:
                st.warning(f"⚠️ 현재 시가총액 상위 {scan_limit}개 종목 중, 200일선 눌림목 조건과 일치하는 종목이 없습니다.")
                st.info("이 조건식은 매우 깐깐한 '안전 제일주의' 로직입니다. 포착된 종목이 없다는 것은 현재 주도 우량주 중 확실한 턴어라운드 지점에 온 종목이 없음을 의미합니다. 내일 다시 스캔해 보세요!")
