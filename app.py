import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import FinanceDataReader as fdr
from datetime import datetime, timedelta

# ==========================================
# 1. 페이지 설정 및 제목 (모바일 최적화)
# ==========================================
st.set_page_config(page_title="StockMap", layout="wide")

st.markdown("""
    <style>
    /* 🌟 모바일 환경 '당겨서 새로고침(Pull-to-refresh)' 방지 코팅 */
    html, body {
        overscroll-behavior-y: none;
    }
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
        font-size: 0.85rem;
        line-height: 1.5;
        background-color: rgba(255, 165, 0, 0.05);
        border-left: 4px solid #FF8C00;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 StockMap")
st.markdown("---")

if 'recent_searches' not in st.session_state:
    st.session_state.recent_searches = []

if 'search_query' not in st.session_state:
    st.session_state.search_query = ""
if 'trigger_search' not in st.session_state:
    st.session_state.trigger_search = False

def on_recent_click(query):
    st.session_state.search_query = query
    st.session_state.trigger_search = True

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
    df = df.copy()  
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
    
    df['STD'] = close.rolling(window=20).std()
    df['BB_Upper'] = df['MA20'] + (df['STD'] * 2)
    df['BB_Lower'] = df['MA20'] - (df['STD'] * 2)
    df['BBW'] = (df['BB_Upper'] - df['BB_Lower']) / df['MA20'] * 100 
    
    high_diff = df['High'].diff()
    low_diff = -df['Low'].diff()
    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)
    plus_dm[(high_diff > low_diff) & (high_diff > 0)] = high_diff[(high_diff > low_diff) & (high_diff > 0)]
    minus_dm[(low_diff > high_diff) & (low_diff > 0)] = low_diff[(low_diff > high_diff) & (low_diff > 0)]
    
    plus_di = 100 * (plus_dm.rolling(window=14).mean() / (df['ATR'] + 1e-10))
    minus_di = 100 * (minus_dm.rolling(window=14).mean() / (df['ATR'] + 1e-10))
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    df['ADX'] = dx.rolling(window=14).mean()
    df['+DI'] = plus_di
    df['-DI'] = minus_di
    
    direction = (delta > 0).astype(int) - (delta < 0).astype(int)
    df['OBV'] = (df['Volume'] * direction).cumsum()
    df['Vol_MA5'] = df['Volume'].rolling(window=5).mean()
    df['Vol_Ratio'] = (df['Volume'] / df['Vol_MA5']) * 100
    
    return df

def calculate_quant_score(df, is_short_term):
    if len(df) < 5: return 0
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0
    
    if is_short_term:
        if not pd.isna(latest['RSI']):
            if latest['RSI'] < 30: score += 25
            elif latest['RSI'] < 50: score += 15
            elif latest['RSI'] < 70: score += 5
        if not pd.isna(latest['MACD']) and not pd.isna(latest['Signal']):
            if latest['MACD'] > latest['Signal']: score += 25
        if not pd.isna(latest['OBV']) and latest['OBV'] > df['OBV'].iloc[-5]: score += 30
        if not pd.isna(latest['Vol_Ratio']):
            if latest['Vol_Ratio'] >= 150 and latest['Close'] > prev['Close']: score += 20
    else:
        if not pd.isna(latest['MA60']) and latest['Close'] > latest['MA60']: score += 30
        highest_60 = df['Close'].tail(60).max()
        if latest['Close'] >= highest_60 * 0.95: score += 20
        if not pd.isna(latest['MACD']) and not pd.isna(latest['Signal']):
            if latest['MACD'] > latest['Signal']: score += 20
        if not pd.isna(latest['OBV']) and latest['OBV'] > df['OBV'].iloc[-5]: score += 20
        if not pd.isna(latest['RSI']):
            if 40 <= latest['RSI'] <= 70: score += 10
            
    return min(score, 100)

def detect_patterns_and_levels(df):
    if len(df) < 3: return [], 0, 0  
    latest = df.iloc[-1]
    patterns = []
    body = abs(latest['Open'] - latest['Close'])
    lower_shadow = min(latest['Open'], latest['Close']) - latest['Low']
    upper_shadow = latest['High'] - max(latest['Open'], latest['Close'])
    
    if lower_shadow > body * 2 and upper_shadow < body: patterns.append("🔨 망치형 (바닥권 반등 신호)")
    if len(df) >= 2 and latest['Close'] > latest['Open'] and latest['Close'] > df['High'].iloc[-2]: patterns.append("🚀 상승 장악형 (추세 전환)")
    
    lookback = min(61, len(df))
    past_df = df.iloc[-lookback:-1] if lookback > 1 else df.iloc[:-1]
    if past_df.empty:
        return patterns, latest['Close'] * 0.95, latest['Close'] * 1.05
    
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
            if not matched:
                clusters.append({'center': p, 'prices': [p]})
        clusters.sort(key=lambda x: len(x['prices']), reverse=True)
        return clusters

    low_mask = (closes < closes.shift(1)) & (closes < closes.shift(-1))
    support_candidates = closes[low_mask]

    high_mask = (closes > closes.shift(1)) & (closes > closes.shift(-1))
    resistance_candidates = closes[high_mask]

    if len(support_candidates) >= 2:
        sup_clusters = cluster_levels(support_candidates)
        valid_sup = [c for c in sup_clusters if c['center'] <= latest['Close']]
        if valid_sup:
            support = valid_sup[0]['center']  
        else:
            support = closes[closes <= latest['Close']].min() if not closes[closes <= latest['Close']].empty else closes.min()
    else:
        below = closes[closes <= latest['Close']]
        support = below.min() if not below.empty else closes.min()

    if len(resistance_candidates) >= 2:
        res_clusters = cluster_levels(resistance_candidates)
        valid_res = [c for c in res_clusters if c['center'] >= latest['Close']]
        if valid_res:
            resistance = valid_res[0]['center']  
        else:
            resistance = closes[closes >= latest['Close']].max() if not closes[closes >= latest['Close']].empty else closes.max()
    else:
        above = closes[closes >= latest['Close']]
        resistance = above.max() if not above.empty else closes.max()

    return patterns, support, resistance

@st.cache_data(ttl=60)
def get_stock_data(code):
    start_date = (datetime.now() - timedelta(days=1825)).strftime('%Y-%m-%d')
    try:
        df = fdr.DataReader(code, start=start_date)
        if df.empty: return pd.DataFrame()
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        return df
    except: return pd.DataFrame()

def generate_detailed_opinions(df, sup, res, currency, decimals, is_short_term, time_unit, weekly_bullish=None):
    latest = df.iloc[-1]
    close = float(latest['Close'])
    rsi = float(latest['RSI']) if not pd.isna(latest['RSI']) else 50
    macd = float(latest['MACD']) if not pd.isna(latest['MACD']) else 0
    signal = float(latest['Signal']) if not pd.isna(latest['Signal']) else 0
    vol_ratio = float(latest['Vol_Ratio']) if not pd.isna(latest['Vol_Ratio']) else 100
    atr = float(latest['ATR']) if not pd.isna(latest['ATR']) else 0
    obv = float(latest['OBV']) if not pd.isna(latest['OBV']) else 0
    
    vol_pct = (atr / close) * 100 if close > 0 else 0
    
    adx = float(latest['ADX']) if not pd.isna(latest['ADX']) else 0
    p_di = float(latest['+DI']) if not pd.isna(latest['+DI']) else 0
    m_di = float(latest['-DI']) if not pd.isna(latest['-DI']) else 0
    ma20 = float(latest['MA20']) if not pd.isna(latest['MA20']) else close
    ma60 = float(latest['MA60']) if not pd.isna(latest['MA60']) else close
    macd_diff = macd - signal
    
    simple_lookback = min(5, len(df) - 1) if len(df) > 1 else 1
    long_lookback   = min(13, len(df) - 1) if len(df) > 1 else 1
    obv_lookback    = simple_lookback if is_short_term else long_lookback
    simple_prev_obv = float(df['OBV'].iloc[-obv_lookback])
    
    swing_lookback = min(60, len(df) - 1) if len(df) > 1 else 1
    prev_close, prev_obv, prev_rsi = close, obv, rsi  
    if swing_lookback > 5:
        past_df = df.iloc[-swing_lookback:-1]
        local_min_mask = (
            (past_df['Close'] < past_df['Close'].shift(1)) &
            (past_df['Close'] < past_df['Close'].shift(2)) &
            (past_df['Close'] < past_df['Close'].shift(-1)) &
            (past_df['Close'] < past_df['Close'].shift(-2))
        )
        local_min_df = past_df[local_min_mask]
        if len(local_min_df) >= 2:
            trough1_idx = local_min_df.index[-2]  
            trough2_idx = local_min_df.index[-1]  
            prev_close = float(df.loc[trough1_idx, 'Close'])
            prev_obv   = float(df.loc[trough1_idx, 'OBV'])
            prev_rsi   = float(df.loc[trough1_idx, 'RSI']) if not pd.isna(df.loc[trough1_idx, 'RSI']) else 50
            trough2_close = float(df.loc[trough2_idx, 'Close'])
            trough2_obv   = float(df.loc[trough2_idx, 'OBV'])
            trough2_rsi   = float(df.loc[trough2_idx, 'RSI']) if not pd.isna(df.loc[trough2_idx, 'RSI']) else 50
            prev_close, prev_obv, prev_rsi = trough2_close, trough2_obv, trough2_rsi
        elif len(local_min_df) == 1:
            min_idx = local_min_df.index[-1]
            prev_close = float(df.loc[min_idx, 'Close'])
            prev_obv   = float(df.loc[min_idx, 'OBV'])
            prev_rsi   = float(df.loc[min_idx, 'RSI']) if not pd.isna(df.loc[min_idx, 'RSI']) else 50

    squeeze_lookback = 120 if is_short_term else 24
    past_bbw = df['BBW'].iloc[-squeeze_lookback:-1] if len(df) > squeeze_lookback else df['BBW'].iloc[:-1]
    current_bbw = float(latest['BBW']) if not pd.isna(latest['BBW']) else 100
    is_squeeze = not past_bbw.empty and (current_bbw <= past_bbw.min() * 1.05) 

    prev_adx = float(df['ADX'].iloc[-2]) if len(df) > 1 and not pd.isna(df['ADX'].iloc[-2]) else 0
    
    if is_squeeze:
        regime = "에너지 응축 (스퀴즈)"
    elif vol_ratio >= 150 and adx > prev_adx and adx > 20:
        regime = "변동성 폭발"
    elif adx < 25:
        regime = "횡보 박스"
    elif p_di > m_di and close >= ma20:
        regime = "강세 추세"
    else:
        regime = "약세 추세"

    comments = {}
    
    comments['ADX'] = f"현재 ADX 추세강도 지수는 **{adx:.1f}**이며, 알고리즘은 현재 시장을 **[{regime}]** 국면으로 확정했습니다."
    
    if regime == "에너지 응축 (스퀴즈)":
        comments['RSI'] = f"RSI({rsi:.1f}): 볼린저 밴드 수축 국면이므로 RSI의 움직임이 매우 둔화되어 있습니다. 방향성 탐색 중입니다."
        comments['MACD'] = f"MACD({macd_diff:,.{decimals}f}): 이동평균선이 밀집하며 MACD도 0선에 완전히 수렴했습니다. 폭풍 전야의 고요한 상태입니다."
    elif regime == "횡보 박스":
        comments['RSI'] = f"RSI({rsi:.1f}): 횡보장에서는 RSI의 신뢰도가 가장 높습니다. " + ("박스권 하단 지지선(과매도) 터치로 기술적 반등이 예상됩니다." if rsi <= 40 else "박스권 상단 저항선(과매수) 도달로 조정이 예상됩니다." if rsi >= 60 else "박스권 중간에서 뚜렷한 방향성을 탐색 중입니다.")
        comments['MACD'] = f"MACD({macd_diff:,.{decimals}f}): 뚜렷한 추세가 부재한 박스권이므로 MACD 크로스 신호의 신뢰도는 다소 떨어집니다."
    elif regime == "강세 추세":
        comments['RSI'] = f"RSI({rsi:.1f}): 강세장에서는 지표가 쉽게 과열권에 진입합니다. " + ("강한 매수세로 단기 과열(70 이상) 상태이나 추세는 굳건합니다." if rsi >= 70 else "상승 추세 중 발생한 건전한 눌림목(조정) 타점입니다." if rsi <= 50 else "안정적인 상승 탄력을 유지하고 있습니다.")
        comments['MACD'] = f"MACD({macd_diff:,.{decimals}f}): 상승 모멘텀이 강하게 유지되며 이평선 정배열 확장을 지지하고 있습니다."
    elif regime == "약세 추세":
        comments['RSI'] = f"RSI({rsi:.1f}): 약세장에서는 지표가 지속적으로 침체권에 머뭅니다. " + ("일시적인 기술적 반등(데드캣 바운스) 구간으로 매도를 고려할 시점입니다." if rsi >= 55 else "극단적 과매도 상태이나, 하락장의 '떨어지는 칼날'일 수 있으니 섣부른 진입은 위험합니다." if rsi <= 30 else "지속적인 하락 압력을 받고 있습니다.")
        comments['MACD'] = f"MACD({macd_diff:,.{decimals}f}): 하락 모멘텀이 강하며, 추세 반전을 암시하는 뚜렷한 신호가 아직 없습니다."
    elif regime == "변동성 폭발":
        comments['RSI'] = f"RSI({rsi:.1f}): 변동성 폭발로 인해 투심이 한쪽으로 극단적으로 쏠리는 오버슈팅/패닉셀 국면입니다."
        comments['MACD'] = f"MACD({macd_diff:,.{decimals}f}): 단기 모멘텀이 평소의 범위를 벗어나 급격하게 방향성을 분출하고 있습니다."

    comments['VOL'] = f"상대 거래량이 평균 대비 **{vol_ratio:.0f}%** 수준입니다. " + ("대량 거래가 터지며 시장의 강한 이목이 집중되었습니다." if vol_ratio > 150 else "평이한 수준의 횡보성 거래가 이뤄지고 있습니다.")
    comments['OBV'] = f"최근 {obv_lookback}{time_unit}간 누적 수급(OBV)이 **{'상승(자금 유입)' if obv > simple_prev_obv else '하락(자금 이탈)'}** 중입니다."
    comments['ATR'] = f"예상되는 실질 변동폭(ATR)은 주당 평균 **{vol_pct:.1f}% ({atr:,.{decimals}f}{currency})** 수준입니다."

    dist_to_sup = (close - sup) / sup * 100 if sup > 0 else 100
    near_sup = abs(dist_to_sup) <= 5 
    bullish_div = (close < prev_close) and (obv > prev_obv or rsi > prev_rsi)
    
    if is_short_term:
        if regime == "에너지 응축 (스퀴즈)":
            pos, strategy = "⚖️ 방향성 대기 (관망)", "볼린저 밴드가 극도로 수축되었습니다. 상방 돌파 시 추격 매수, 하방 이탈 시 즉각 손절(관망) 준비를 하세요."
        elif regime == "횡보 박스":
            if near_sup or rsi <= 40: pos, strategy = "🟠 단기 박스권 하단 매수", "박스권 하단 지지선을 확인했습니다. 상단 저항선까지의 핑퐁 반등 매매가 유효합니다."
            elif (res - close) / close * 100 <= 5 or rsi >= 65: pos, strategy = "🔵 단기 박스권 상단 매도", "박스권 상단 저항에 도달했습니다. 뚫지 못할 확률이 높으므로 비중 축소를 권장합니다."
            else: pos, strategy = "⚖️ 단기 관망", "박스권 중간 지대입니다. 어설픈 진입보다는 지지선/저항선 도달을 기다리세요."
        elif regime == "강세 추세":
            if rsi <= 55: pos, strategy = "🔴 추세 눌림목 적극 매수", "강한 상승 추세 속에서 건전한 조정(눌림목)이 발생한 훌륭한 진입 타점입니다."
            elif rsi >= 75: pos, strategy = "🔵 분할 익절", "강한 추세가 유지 중이나 단기 과열권입니다. 리스크 관리를 위해 수익을 분할 실현하세요."
            else: pos, strategy = "🟠 추세 홀딩", "안정적인 우상향 흐름이 진행 중입니다. 달리는 말에서 섣불리 내리지 마세요."
        elif regime == "약세 추세":
            if rsi <= 30 and near_sup: pos, strategy = "🟠 데드캣 바운스 노림", "투매가 나온 과매도 상태이나 하락장이므로, '짧은 기술적 반등(데드캣)'만 노리고 빠르게 빠져나와야 합니다."
            else: pos, strategy = "🔷 적극 매도 및 관망", "하락 추세가 지배적입니다. 섣부른 물타기를 절대 금지하고 현금을 관망하세요."
        elif regime == "변동성 폭발":
            if close > prev_close: pos, strategy = "🔴 돌파 추세 추종", "대량 거래와 함께 상방으로 변동성이 터졌습니다. 새로운 대시세 랠리 시작 가능성이 높습니다."
            else: pos, strategy = "🔷 패닉셀 회피 (적극 매도)", "대량 거래를 동반한 하방 변동성 폭발입니다. 추가 급락을 막기 위해 즉각적인 리스크 관리가 필요합니다."
    else:
        if regime == "에너지 응축 (스퀴즈)":
            pos, strategy = "⚖️ 장기 관망", "장기적인 에너지가 응축되고 있습니다. 박스권 돌파 방향이 1~2년의 대시세를 결정할 것입니다."
        elif regime in ["강세 추세", "변동성 폭발"] and close > ma60:
            pos, strategy = "🔴 비중 확대 (장기)", "대세 상승장에 진입했으며 추세와 모멘텀이 모두 훌륭합니다."
        elif regime == "약세 추세" and close < ma60:
            if rsi < 30: pos, strategy = "🟠 저점 분할 매집", "역사적 저평가 구간입니다. 장기적 안목에서 1차 분할 매집이 유효합니다."
            else: pos, strategy = "🔷 비중 축소 (장기)", "대세 하락장이 진행 중이므로 장기 현금 비중을 늘려야 합니다."
        else:
            pos, strategy = "⚖️ 장기 관망", "대세 추세가 전환되는 변곡점 또는 횡보 구간입니다. 뚜렷한 방향성을 대기하세요."

    q_score = calculate_quant_score(df, is_short_term)
    buy_positions  = {"🔴 단기 적극 매수", "🟠 단기 분할 매수", "🔴 추세 눌림목 적극 매수", "🟠 단기 박스권 하단 매수", "🔴 비중 확대 (장기)", "🟠 저점 분할 매집", "🔴 돌파 추세 추종"}
    sell_positions = {"🔷 단기 적극 매도", "🔵 단기 분할 매도", "🔵 단기 박스권 상단 매도", "🔷 비중 축소 (장기)", "🔷 적극 매도 및 관망", "🔷 패닉셀 회피 (적극 매도)"}
    
    if pos in buy_positions and q_score < 30:
        pos      = "⚖️ 단기 관망" if is_short_term else "⚖️ 장기 관망"
        strategy = f"매수 신호가 감지되었으나 전체 퀀트 스코어({q_score}점)가 너무 낮아 신뢰도가 떨어집니다. 추가 확인 후 진입하세요."
    elif pos in sell_positions and q_score > 70:
        pos      = "⚖️ 단기 관망" if is_short_term else "⚖️ 장기 관망"
        strategy = f"매도 신호가 감지되었으나 전체 퀀트 스코어({q_score}점)가 높아 신호가 상충합니다. 방향성 확인 후 대응하세요."

    mode_str = "단기 스윙" if is_short_term else "장기 가치투자"
    
    ai_op = f"🤖 **StockMap AI {mode_str} 심층 진단 리포트**\n\n"
    
    ai_op += f"🔍 **[시장 국면 분류]**\n\n"
    ai_op += f"• ADX 추세 강도({adx:.1f})와 이평선 배열을 종합 분석한 결과, 현재 이 종목은 **[{regime}]** 국면에 있습니다.\n\n"
    
    if is_short_term and weekly_bullish is not None:
        ai_op += f"⏱️ **[MTF 다중 시간대 분석]**\n\n"
        if regime == "강세 추세":
            if weekly_bullish:
                ai_op += "• **완벽한 정배열:** 주봉(장기)과 일봉(단기)이 모두 완벽한 상승 추세입니다. 대세 상승장 속의 훌륭한 매수 타점입니다.\n\n"
            else:
                ai_op += "• **단기 반등 주의:** 일봉은 강세이나, 주봉(장기)은 여전히 하락장(역배열)입니다. 장기 저항선에 부딪힐 수 있으니 목표 수익률을 짧게 잡으세요.\n\n"
        elif regime == "약세 추세":
            if weekly_bullish:
                ai_op += "• **장기 상승장 속 눌림목:** 일봉은 약세이나 주봉(장기)은 굳건한 상승장입니다. 장기 투자자에게는 매력적인 할인(눌림목) 구간이 될 수 있습니다.\n\n"
            else:
                ai_op += "• **완벽한 역배열:** 주봉과 일봉 모두 하락장입니다. 바닥을 섣불리 예측하지 말고 철저히 현금을 관망하세요.\n\n"
    
    ai_op += f"💡 **[국면 맞춤형 통합 해석]**\n\n"
    if regime == "에너지 응축 (스퀴즈)":
        ai_op += "• 현재 볼린저 밴드의 폭이 최근 6개월 내 최저 수준으로 극도로 압축된 **[변동성 응축(Squeeze)]** 상태입니다. 조만간 위든 아래든 거대한 폭발이 임박했으니, 돌파 방향을 예의주시하세요.\n\n"
    elif regime == "횡보 박스":
        ai_op += "• 뚜렷한 방향성이 없이 에너지를 응축하는 횡보장입니다. 지표의 '과열/침체' 신호를 역발상으로 활용하는 박스권 매매가 유리합니다.\n\n"
        if near_sup: ai_op += f"• 현재 박스권 하단 지지선({sup:,.{decimals}f}{currency})에 근접하여 반등 매수 매력도가 매우 높습니다.\n\n"
    elif regime == "강세 추세":
        ai_op += "• 매수세가 시장을 장악한 강세장입니다. 보조지표가 다소 과열되더라도 추세가 꺾이지 않는 한 홀딩하는 것이 수익률을 극대화합니다.\n\n"
        if rsi < 55: ai_op += "• 특히 현재는 가파른 상승 중 일시적으로 쉬어가는 '눌림목' 패턴이 포착되어 매우 좋은 진입 기회입니다.\n\n"
    elif regime == "약세 추세":
        ai_op += "• 하락 압력이 지배적인 역배열 약세장입니다. 어설픈 지지선은 쉽게 붕괴되므로 보수적인 관망과 현금 관리가 생명입니다.\n\n"
        if rsi > 55: ai_op += "• 현재 나타나는 반등은 추세 전환이 아닌 일시적인 데드캣 바운스일 확률이 높으므로 탈출 기회로 삼으시길 권장합니다.\n\n"
    elif regime == "변동성 폭발":
        ai_op += "• 평소 대비 막대한 자금이 몰리며 주가가 위아래로 거칠게 요동치고 있습니다. 방향이 확정되면 걷잡을 수 없는 큰 시세가 나올 수 있습니다.\n\n"

    ai_op += f"📊 **[수급 및 주요 레벨]**\n\n"
    if obv > simple_prev_obv: ai_op += "• **세력수급:** 겉보기 주가 흐름 이면에 누적 수급(OBV)이 꾸준히 유입되며 긍정적인 '매집' 정황이 관찰됩니다.\n\n"
    else: ai_op += "• **세력수급:** 누적 수급이 지속적으로 이탈 중이므로 어설픈 '가짜 반등'에 속지 않도록 유의해야 합니다.\n\n"
    if regime == "강세 추세" and obv < simple_prev_obv:
        ai_op += "⚠️ **[지표 충돌 경고]** 추세는 상승 중이나 수급(OBV)이 은밀히 이탈 중입니다. 수급 없는 억지 상승일 수 있으므로 주의하세요.\n\n"
        
    ai_op += f"• **핵심레벨:** 1차 지지선은 **{sup:,.{decimals}f}{currency}**, 1차 저항선은 **{res:,.{decimals}f}{currency}**에 튼튼하게 형성되어 있습니다.\n\n"
    ai_op += f"• **손절기준:** 현재 실질 변동폭(ATR)인 **{vol_pct:.1f}%**를 감안하여 휩소(속임수)에 털리지 않게 넉넉히 리스크를 설정하세요.\n\n"

    if bullish_div and regime != "약세 추세": 
        ai_op += "🔥 **[특급 패턴: 상승 다이버전스 포착]** 스윙 로우 대비 주가는 내렸으나 지표가 오르는 강력한 반전(매수) 시그널이 확인되었습니다!\n\n"

    comments['AI'] = f"{ai_op}🎯 **최종 투자 전략:** {strategy} (AI 권장 포지션: **{pos}**)"
    
    return pos, strategy, comments

# ==========================================
# 3. 사이드바 및 실행 UI
# ==========================================
with st.sidebar:
    st.header("⚙️ 분석 설정")
    analyze_mode = st.radio("투자 성향 설정", ["단기 투자 (6개월 차트/일봉)", "장기 투자 (2년 차트/주봉)"])
    
    new_search = st.text_input("종목명/코드 입력", placeholder="삼성전자, NVDA 등", key="search_query")
    run_btn = st.button("🚀 분석 실행", type="primary", use_container_width=True)
    
    st.markdown(f"""
    <div class="style-box">
    <b>🔍 분석 모드 가이드</b><br>
    • <b>단기</b>: 최근 6개월의 일봉 파동을 읽어 단기 타점을 포착합니다.<br>
    • <b>장기</b>: 최근 2년의 <b>주봉(Weekly)</b> 대세 흐름을 판별합니다.
    </div>
    """, unsafe_allow_html=True)
    
    target_query = None
    if run_btn or st.session_state.trigger_search:
        target_query = st.session_state.search_query
        st.session_state.trigger_search = False
        
        if target_query:
            display_name, ticker_symbol, raw_query, currency, decimals = parse_query(target_query)
            if {'query': raw_query, 'display_name': display_name} not in st.session_state.recent_searches:
                st.session_state.recent_searches.insert(0, {'query': raw_query, 'display_name': display_name})
                st.session_state.recent_searches = st.session_state.recent_searches[:5]
    
    st.divider()
    st.subheader("🕒 최근 검색")
    for idx, item in enumerate(st.session_state.recent_searches):
        st.button(f"▪️ {item['display_name']}", key=f"rs_{idx}_{item['query']}", use_container_width=True, on_click=on_recent_click, args=(item['query'],))

# ==========================================
# 4. 메인 화면 분석 결과 출력
# ==========================================
if target_query:
    display_name, ticker_symbol, raw_query, currency, decimals = parse_query(target_query)

    with st.spinner(f"📡 '{display_name}' 딥다이브 리포트 생성 중..."):
        raw_df = get_stock_data(ticker_symbol)
        
    if raw_df.empty:
        st.error("데이터를 찾을 수 없습니다. 종목명을 다시 확인해주세요.")
    else:
        is_short_term = "단기" in analyze_mode
        time_unit = "일" if is_short_term else "주"
        
        chart_df_daily = calculate_indicators(raw_df.copy())
        
        weekly_raw = raw_df.resample('W').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
        chart_df_weekly = calculate_indicators(weekly_raw)
        
        weekly_bullish = None
        if not chart_df_weekly.empty:
            w_latest = chart_df_weekly.iloc[-1]
            weekly_bullish = (w_latest['Close'] > w_latest['MA60']) and (w_latest['MACD'] > w_latest['Signal'])

        if not is_short_term:
            chart_df = chart_df_weekly
            default_days = 730 
        else:
            chart_df = chart_df_daily
            default_days = 180 

        cur_price = raw_df['Close'].iloc[-1]
        diff = cur_price - raw_df['Close'].iloc[-2] if len(raw_df) > 1 else 0
        st.subheader(f"📑 {display_name} 리포트")
        st.metric("현재 주가", f"{cur_price:,.{decimals}f} {currency}", f"{diff:,.{decimals}f} {currency}")

        q_score = calculate_quant_score(chart_df, is_short_term)
        st.write(f"### 💯 퀀트 스코어: **{q_score}점**")
        st.progress(q_score / 100)

        pts, sup, res = detect_patterns_and_levels(chart_df)

        if len(chart_df) < 2:
            st.warning("데이터가 부족하여 상세 분석을 수행할 수 없습니다.")
        else:
            pos, strat, comments = generate_detailed_opinions(chart_df, sup, res, currency, decimals, is_short_term, time_unit, weekly_bullish)
        
            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    st.markdown("### 🎯 **종합 전략**")
                    st.warning(f"**포지션: {pos}**\n\n**의견:** {strat}")
            with col2:
                with st.container(border=True):
                    st.markdown("### 🔍 **차트 패턴 및 지지/저항**")
                    p_text = ", ".join(pts) if pts else "특이 패턴 없음"
                    st.write(f"📍 **패턴:** {p_text}")
                    sup_text = f"{sup:,.{decimals}f} {currency}" if sup > 0 else "데이터 부족"
                    res_text = f"{res:,.{decimals}f} {currency}" if res > 0 else "데이터 부족"
                    st.write(f"🛡️ **지지:** {sup_text} | 🚧 **저항:** {res_text}")

            with st.expander("🔬 지표별 상세 수치 분석 (용어 클릭)", expanded=True):
                indicator_descriptions = {
                    "ADX 추세강도": "**ADX (평균방향성지수)**\n\n주가의 상승/하락 방향과 무관하게, 현재 진행 중인 추세의 '파워(속도)' 자체를 측정합니다. 25 이상이면 강한 추세가 진행 중임을 뜻하며, 25 미만이면 에너지가 빠진 횡보(박스권) 장세를 의미합니다.",
                    "상대 거래량": "**상대 거래량 (Relative Volume)**\n\n최근 5일(주) 평균 거래량 대비 현재 거래량의 비율입니다. 150% 이상일 경우 평소보다 많은 자금이 유입되며 의미 있는 변동성이 발생하고 있음을 뜻합니다.",
                    "OBV 누적": "**OBV (On Balance Volume)**\n\n주가가 상승한 날의 거래량은 더하고 하락한 날의 거래량은 빼서 누적한 수급 지표입니다. 주가가 횡보/하락함에도 OBV가 상승하면 세력의 '매집'으로, 반대의 경우 '분산(이탈)'으로 해석합니다.",
                    "RSI 강도": "**RSI (상대강도지수)**\n\n주가의 상승폭과 하락폭을 바탕으로 과열 상태를 수치화한 모멘텀 지표입니다.\n• **70 이상**: 과매수 (단기 고점 징후, 수익실현 고려)\n• **30 이하**: 과매도 (단기 바닥 징후, 반등/진입 고려)",
                    "MACD 흐름": "**MACD (이동평균수렴확산지수)**\n\n단기 이동평균선과 장기 이동평균선의 차이를 이용해 추세의 방향과 힘을 파악합니다. MACD 선이 시그널 선을 상향 돌파(골든크로스)하면 매수, 하향 돌파(데드크로스)하면 매도 신호로 해석합니다.",
                    "ATR 변동성": "**ATR (평균진정범위)**\n\n고점과 저점, 전일 종가를 모두 고려한 '실질적인 주가 변동폭'의 평균입니다. 이 수치가 높을수록 위아래 흔들림이 크다는 의미이며, 자신의 투자 성향에 맞는 손절가(Stop Loss)를 설정할 때 유용하게 쓰입니다."
                }
                
                for label, key in [("ADX 추세강도", "ADX"), ("상대 거래량", "VOL"), ("OBV 누적", "OBV"), ("RSI 강도", "RSI"), ("MACD 흐름", "MACD"), ("ATR 변동성", "ATR")]:
                    c1, c2 = st.columns([0.25, 0.75])
                    with c1.popover(label, use_container_width=True): 
                        st.info(indicator_descriptions.get(label, f"**{label}**"))
                    c2.markdown(comments.get(key, '데이터 없음'))
                st.divider()
                st.info(comments.get('AI'))

        # ==========================================
        # 🌟 완전 고정형 및 능동 반응형 차트 (이동/확대 완전 차단 유지)
        # ==========================================
        tab1, tab2 = st.tabs(["📈 주가 & RSI 차트", "📊 수급 에너지(OBV)"])
        
        data_start_date = chart_df.index[0]
        calculated_start_date = datetime.now() - timedelta(days=default_days)
        final_start_date = max(data_start_date, calculated_start_date)
        
        plot_df = chart_df[chart_df.index >= final_start_date]
        
        if not plot_df.empty:
            min_vals = plot_df[['Low', 'MA20', 'MA60']].min(skipna=True)
            max_vals = plot_df[['High', 'MA20', 'MA60']].max(skipna=True)
            c_min = min_vals.min()
            c_max = max_vals.max()
            if pd.isna(c_min) or pd.isna(c_max) or c_min == c_max:
                c_min = plot_df['Low'].min()
                c_max = plot_df['High'].max()
            padding = (c_max - c_min) * 0.05
            y_range = [c_min - padding, c_max + padding]
        else:
            y_range = None

        with tab1:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
            
            fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name='주가'), row=1, col=1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BB_Upper'], name='BB 상단', line=dict(color='rgba(173, 216, 230, 0.4)', width=1, dash='dot')), row=1, col=1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BB_Lower'], name='BB 하단', line=dict(color='rgba(173, 216, 230, 0.4)', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(173, 216, 230, 0.1)'), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA20'], name=f'20{time_unit}선', line=dict(color='orange', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA60'], name=f'60{time_unit}선', line=dict(color='green', width=1)), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['RSI'], name='RSI', line=dict(color='#00BFFF', width=1.5)), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=1, row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=1, row=2, col=1)
            fig.add_hrect(y0=30, y1=70, fillcolor="gray", opacity=0.1, line_width=0, row=2, col=1)

            fig.update_layout(
                height=550, 
                margin=dict(t=10, b=10, l=0, r=0), 
                dragmode=False, 
                hovermode='x unified', showlegend=False
            )
            
            fig.update_xaxes(range=[final_start_date, datetime.now()], rangeslider=dict(visible=False), fixedrange=True, row=1, col=1)
            fig.update_xaxes(rangeslider=dict(visible=False), fixedrange=True, row=2, col=1)
            fig.update_yaxes(range=y_range, fixedrange=True, row=1, col=1)
            fig.update_yaxes(range=[0, 100], fixedrange=True, row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': False, 'displayModeBar': False})
            
        with tab2:
            if 'OBV' in plot_df.columns:
                obv_fig = go.Figure(data=[go.Scatter(x=plot_df.index, y=plot_df['OBV'], name='OBV', fill='tozeroy', line=dict(color='purple'))])
                obv_fig.update_layout(
                    height=350, margin=dict(t=10, b=10, l=0, r=0), 
                    dragmode=False, 
                    hovermode='x unified'
                )
                obv_fig.update_xaxes(range=[final_start_date, datetime.now()], fixedrange=True)
                obv_fig.update_yaxes(fixedrange=True)
                st.plotly_chart(obv_fig, use_container_width=True, config={'scrollZoom': False, 'displayModeBar': False})
else:
    st.info("👈 사이드바에서 종목을 검색하여 분석을 시작하세요.")
