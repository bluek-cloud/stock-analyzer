import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import FinanceDataReader as fdr
from datetime import datetime, timedelta
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import os, io, requests
from llm_analyst import generate_rag_analyst_report
from backtest_engine import match_current_setup, run_stock_backtest

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
# 2. 공통 데이터 처리 함수 (3중 안전망 KRX 데이터베이스)
# ==========================================
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'krx_cache.csv')

@st.cache_data(ttl=86400)
def get_krx_data():
    today = datetime.now()
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    # 1. GitHub 캐시 (오늘부터 과거 14일 역추적하여 존재하는 최신 파일 로드)
    for i in range(14):
        dt_str = (today - timedelta(days=i)).strftime('%Y-%m-%d')
        url = f'https://raw.githubusercontent.com/FinanceData/fdr_krx_data_cache/refs/heads/master/data/listing/krx/{dt_str}.csv'
        try:
            r = requests.head(url, headers=headers, timeout=2)
            if r.status_code == 200:
                df = pd.read_csv(url, dtype={'Code': str, 'Dept': str, 'ChangeCode': str, 'MarketId': str})
                if not df.empty:
                    df['Code'] = df['Code'].astype(str).str.zfill(6)
                    try:
                        df[['Code', 'Name', 'Market', 'Marcap']].to_csv(CACHE_FILE, index=False)
                    except Exception:
                        pass
                    return df[['Code', 'Name', 'Market', 'Marcap']]
        except Exception:
            continue
            
    # 2. 한국거래소 KIND 공식 상장회사 목록 다운로드 (100% 실시간 작동)
    try:
        url = 'http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13'
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            dfs = pd.read_html(io.StringIO(r.text), header=0)
            df_kind = dfs[0].rename(columns={'회사명': 'Name', '종목코드': 'Code'})
            df_kind['Code'] = df_kind['Code'].astype(str).str.zfill(6)
            df_kind['Market'] = 'KRX'
            df_kind['Marcap'] = 0
            try:
                df_kind[['Code', 'Name', 'Market', 'Marcap']].to_csv(CACHE_FILE, index=False)
            except Exception:
                pass
            return df_kind[['Code', 'Name', 'Market', 'Marcap']]
    except Exception:
        pass

    # 3. 로컬 디스크 캐시 파일 로드 (오프라인/네트워크 장애 대비)
    if os.path.exists(CACHE_FILE):
        try:
            df_local = pd.read_csv(CACHE_FILE, dtype={'Code': str})
            df_local['Code'] = df_local['Code'].astype(str).str.zfill(6)
            return df_local[['Code', 'Name', 'Market', 'Marcap']]
        except Exception:
            pass

    raise ConnectionError("KRX 종목 목록을 가져올 수 없습니다.")

def _get_krx_data_safe():
    """KRX 데이터 로드 실패 시에도 크래시 없이 로컬 캐시 또는 빈 DataFrame 반환"""
    try:
        return get_krx_data()
    except Exception:
        if os.path.exists(CACHE_FILE):
            try:
                df_local = pd.read_csv(CACHE_FILE, dtype={'Code': str})
                df_local['Code'] = df_local['Code'].astype(str).str.zfill(6)
                return df_local[['Code', 'Name', 'Market', 'Marcap']]
            except Exception:
                pass
        return pd.DataFrame(columns=['Code', 'Name', 'Market', 'Marcap'])

def parse_query(query):
    raw_query = query.strip()
    query_upper = raw_query.upper()
    query_nospace = query_upper.replace(' ', '')
    is_korean = any('\uac00' <= char <= '\ud7a3' for char in raw_query)
    
    krx_df = _get_krx_data_safe()
    
    # 1. 6자리 숫자 코드 입력 (예: 005930)
    if query_upper.isdigit() and len(query_upper) == 6:
        if not krx_df.empty:
            matched = krx_df[krx_df['Code'] == query_upper]
            if not matched.empty:
                return f"{matched.iloc[0]['Name']} ({query_upper})", query_upper, raw_query, "원", 0
        return f"국내 종목 ({query_upper})", query_upper, raw_query, "원", 0
        
    # 2. 국내 종목명 매칭 (완전일치 -> 공백제거일치 -> 접두사일치 -> 부분일치)
    if not krx_df.empty:
        matched = krx_df[krx_df['Name'].str.upper() == query_upper]
        if matched.empty:
            matched = krx_df[krx_df['Name'].str.replace(' ', '').str.upper() == query_nospace]
        if matched.empty:
            matched = krx_df[krx_df['Name'].str.upper().str.startswith(query_upper)]
        if matched.empty and len(query_upper) >= 2:
            matched = krx_df[krx_df['Name'].str.upper().str.contains(query_upper, regex=False)]
            
        if not matched.empty:
            code = matched.iloc[0]['Code']
            name = matched.iloc[0]['Name']
            return f"{name} ({code})", code, raw_query, "원", 0

    # 3. 한글이 포함된 경우 (절대 해외로 보내지 않음)
    if is_korean:
        return f"{raw_query} (국내 종목 미확인)", raw_query, raw_query, "원", 0

    # 4. 영문 티커 (해외 주식)
    return f"{query_upper} (해외)", query_upper, raw_query, "$", 2

@st.cache_data(ttl=60)
def get_stock_data(code, days=1825):
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    df = fdr.DataReader(code, start=start_date)
    if df.empty:
        raise ConnectionError(f"'{code}' 데이터 수신 실패")
    if df.index.tz is not None:
        try:
            df.index = df.index.tz_convert(None)
        except Exception:
            df.index = df.index.tz_localize(None)
    return df

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

# 4. 규칙 엔진 기반 상세 의견 및 AI 심층 진단 리포트 생성 (규칙 엔진 모듈 연동)
from rule_engine import generate_detailed_opinions


# ==========================================
# 3. 신규 스캐너 함수 (200일선 눌림목)
# ==========================================
def scan_200_pullback(top_n=200):
    krx_df = _get_krx_data_safe()
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
        
        st.divider()
        with st.expander("🤖 Gemini AI 설정 (무료)", expanded=False):
            st.caption("구글 AI 스튜디오에서 발급받은 무료 API Key를 입력하시면 RAG 기반 심층 리포트를 생성할 수 있습니다. (미입력 시에도 룰 엔진은 100% 작동)")
            user_gemini_key = st.text_input("Gemini API Key", type="password", key="user_gemini_key", placeholder="AIzaSy...")
            st.markdown("[👉 Google AI Studio에서 무료 키 발급 (10초 소요)](https://aistudio.google.com/app/apikey)")

    if st.session_state.target_query:
        display_name, ticker_symbol, raw_query, currency, decimals = parse_query(st.session_state.target_query)
        if {'query': raw_query, 'display_name': display_name} not in st.session_state.recent_searches:
            st.session_state.recent_searches.insert(0, {'query': raw_query, 'display_name': display_name})
            st.session_state.recent_searches = st.session_state.recent_searches[:5]
        with st.spinner(f"📡 '{display_name}' 분석 중..."):
            try:
                raw_df = get_stock_data(ticker_symbol)
            except Exception:
                raw_df = pd.DataFrame()
        if raw_df.empty: st.error("⚠️ 데이터를 불러올 수 없습니다. 종목명/코드를 확인하거나, 잠시 후 다시 시도해 주세요. (데이터 서버 일시 장애 가능성)")
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
                try:
                    pos, strat, comments = generate_detailed_opinions(chart_df, sup, res, currency, decimals, is_short_term, time_unit, q_score, pts, weekly_bullish)
                except TypeError:
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
                
                # ==========================================
                # 통계적 백테스트 검증 및 종목별 시뮬레이터 카드
                # ==========================================
                regime_label = comments.get('regime_raw', comments.get('ADX', '').split('[')[-1].split(']')[0] if '[' in comments.get('ADX', '') else '횡보')
                market_ctx_dict = {
                    'regime': regime_label,
                    'patterns': pts,
                    'bullish_div': comments.get('bullish_div_raw', False),
                    'is_falling_knife': comments.get('is_falling_knife_raw', False),
                    'is_short_term': is_short_term
                }
                matched_key, matched_stats = match_current_setup(market_ctx_dict, patterns=pts)

                if matched_stats:
                    with st.container(border=True):
                        st.markdown("### 📊 **통계적 백테스트 검증 (Statistical Edge)**")
                        st.caption(f"💡 현재 감지된 패턴/셋업은 **[{matched_stats['name']}]**에 해당합니다. ({matched_stats['benchmark_note']})")
                        
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("20거래일 보유 승률", f"{matched_stats['win_rate_20d']}%", f"5일: {matched_stats['win_rate_5d']}%")
                        m2.metric("손익비 (Profit Factor)", f"{matched_stats['profit_factor']} : 1")
                        m3.metric("평균 기대 수익률", f"+{matched_stats['avg_return_20d']}%", f"최대 반등: +{matched_stats['avg_mfe']}%")
                        m4.metric("검증 표본 수", f"{matched_stats['sample_count']:,} 건")
                        
                        st.info(f"🎯 **실전 통계 가이드:** 권장 손절폭 **-{matched_stats['recommended_sl_pct']}%** | 1차 목표 익절 **+{matched_stats['recommended_tp_pct']}%** (평균 최대 낙폭: -{matched_stats['avg_mae']}%)")

                        # 인터랙티브 종목별 실전 시뮬레이션
                        with st.expander(f"🎯 이 종목에서의 [{matched_stats['name']}] 과거 실전 승률 즉석 시뮬레이션", expanded=False):
                            st.caption(f"'{display_name}'의 과거 전체 차트(최대 5년)에서 **[{matched_stats['name']}]** 타점이 발생했을 때의 실제 성과를 실시간 계산합니다.")
                            sim_btn = st.button("🚀 과거 실전 승률 계산 실행", key="btn_run_sim", use_container_width=True)
                            sim_cache_key = f"sim_{ticker_symbol}_{matched_key}_{is_short_term}"

                            if sim_btn:
                                with st.spinner("⏳ 과거 전체 차트 스캔 및 타점 역추적 시뮬레이션 중..."):
                                    sim_result = run_stock_backtest(chart_df, setup_type=matched_key, hold_days=20)
                                    if sim_result.get('total_trades', 0) == 0:
                                        # 종목 특성상 해당 단독 패턴 표본이 적을 경우 유사 반등 셋업 전체(종합)로 자동 확장
                                        fallback_sim = run_stock_backtest(chart_df, setup_type="AUTO", hold_days=20)
                                        if fallback_sim.get('total_trades', 0) > 0:
                                            fallback_sim['fallback_note'] = f"현재 종목에서는 [{matched_stats['name']}] 단독 표본이 적어, 유사 반등 셋업 전체(종합)로 자동 확장 시뮬레이션했습니다."
                                            sim_result = fallback_sim
                                    st.session_state[sim_cache_key] = sim_result

                            if sim_cache_key in st.session_state:
                                sim_res = st.session_state[sim_cache_key]
                                if 'error' in sim_res:
                                    st.warning(sim_res['error'])
                                elif sim_res.get('total_trades', 0) == 0:
                                    st.info(sim_res.get('message', '타점이 포착되지 않았습니다.'))
                                else:
                                    if 'fallback_note' in sim_res:
                                        st.caption(f"💡 {sim_res['fallback_note']}")
                                    sc1, sc2, sc3, sc4 = st.columns(4)
                                    sc1.metric("과거 총 타점", f"{sim_res['total_trades']} 회")
                                    sc2.metric("실제 승률", f"{sim_res['win_rate']}%", f"{sim_res['win_trades']}승 {sim_res['loss_trades']}패")
                                    sc3.metric("평균 수익률", f"{sim_res['avg_return']:+.2f}%")
                                    sc4.metric("손익비", f"{sim_res['profit_factor']} : 1")

                                    if 'recent_trades' in sim_res and sim_res['recent_trades']:
                                        hold_label = "20거래일" if is_short_term else "20주"
                                        st.markdown(f"##### 📋 최근 과거 타점 상세 내역 ({hold_label} 보유 기준)")
                                        trade_rows = []
                                        for t in sim_res['recent_trades']:
                                            trade_rows.append({
                                                '진입일': t['entry_date'],
                                                '진입가': f"{t['entry_price']:,.{decimals}f} {currency}",
                                                '청산일': t['exit_date'],
                                                '청산가': f"{t['exit_price']:,.{decimals}f} {currency}",
                                                '수익률': f"{t['return_pct']:+.2f}%",
                                                '최대반등(MFE)': f"+{t['mfe_pct']}%",
                                                '결과': "✅ 승리" if t['is_win'] else "❌ 패배"
                                            })
                                        st.dataframe(pd.DataFrame(trade_rows), use_container_width=True, hide_index=True)
                
                # ==========================================
                # RAG 기반 월가 수석 애널리스트 심층 리포트 카드
                # ==========================================
                with st.container(border=True):
                    c_rag_l, c_rag_r = st.columns([0.75, 0.25])
                    with c_rag_l:
                        st.markdown("### 🧠 **월가 수석 애널리스트 RAG 심층 진단**")
                        st.caption("전문 트레이딩 지식 베이스(캔들 역학, 볼린저, 다이버전스, 리스크 관리)를 실시간 검색(RAG)하여 Gemini AI가 종합 분석합니다.")
                    with c_rag_r:
                        gen_btn = st.button("🚀 AI 심층 리포트 생성", key="btn_rag_report", type="primary", use_container_width=True)

                    rag_cache_key = f"rag_report_{ticker_symbol}_{is_short_term}"
                    if gen_btn:
                        with st.spinner("📚 전문 지식 베이스 검색 및 월가 수석 애널리스트 리포트 작성 중..."):
                            regime_label = comments.get('regime_raw', comments.get('ADX', '').split('[')[-1].split(']')[0] if '[' in comments.get('ADX', '') else '횡보')
                            stock_info_dict = {
                                'name': display_name.split(' (')[0],
                                'code': ticker_symbol,
                                'current_price': cur_price,
                                'currency': currency,
                                'quant_score': q_score,
                                'support': sup,
                                'resistance': res,
                                'rsi': float(chart_df['RSI'].iloc[-1]) if 'RSI' in chart_df.columns and not pd.isna(chart_df['RSI'].iloc[-1]) else 50.0,
                                'macd_diff': float(chart_df['MACD'].iloc[-1] - chart_df['Signal'].iloc[-1]) if 'MACD' in chart_df.columns and not pd.isna(chart_df['MACD'].iloc[-1]) else 0.0,
                                'vol_ratio': float(chart_df['Vol_Ratio'].iloc[-1]) if 'Vol_Ratio' in chart_df.columns and not pd.isna(chart_df['Vol_Ratio'].iloc[-1]) else 100.0,
                                'atr': float(chart_df['ATR'].iloc[-1]) if 'ATR' in chart_df.columns and not pd.isna(chart_df['ATR'].iloc[-1]) else 0.0,
                                'rule_position': pos
                            }
                            market_ctx_dict = {
                                'regime': regime_label,
                                'patterns': pts,
                                'bullish_div': comments.get('bullish_div_raw', False),
                                'is_falling_knife': comments.get('is_falling_knife_raw', False),
                                'is_short_term': is_short_term
                            }
                            api_key_to_use = user_gemini_key if user_gemini_key else None
                            rag_result = generate_rag_analyst_report(stock_info_dict, market_ctx_dict, api_key=api_key_to_use)
                            st.session_state[rag_cache_key] = rag_result

                    if rag_cache_key in st.session_state:
                        st.markdown("---")
                        st.markdown(st.session_state[rag_cache_key])
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
