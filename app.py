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
    if len(df) < 60: return [], 0, 0
    latest = df.iloc[-1]
    patterns = []
    body = abs(latest['Open'] - latest['Close'])
    lower_shadow = min(latest['Open'], latest['Close']) - latest['Low']
    upper_shadow = latest['High'] - max(latest['Open'], latest['Close'])
    
    if lower_shadow > body * 2 and upper_shadow < body: patterns.append("🔨 망치형 (바닥권 반등 신호)")
    if latest['Close'] > latest['Open'] and latest['Close'] > df['High'].iloc[-2]: patterns.append("🚀 상승 장악형 (추세 전환)")
    
    support = df['Close'].tail(60).min()
    resistance = df['Close'].tail(60).max()
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

def generate_detailed_opinions(df, sup, res, currency, decimals, is_short_term, time_unit):
    latest = df.iloc[-1]
    close = float(latest['Close'])
    rsi = float(latest['RSI']) if not pd.isna(latest['RSI']) else 50
    macd = float(latest['MACD']) if not pd.isna(latest['MACD']) else 0
    signal = float(latest['Signal']) if not pd.isna(latest['Signal']) else 0
    vol_ratio = float(latest['Vol_Ratio']) if not pd.isna(latest['Vol_Ratio']) else 100
    atr = float(latest['ATR']) if not pd.isna(latest['ATR']) else 0
    obv = float(latest['OBV']) if not pd.isna(latest['OBV']) else 0
    
    simple_lookback = min(20, len(df) - 1) if len(df) > 1 else 1
    simple_prev_obv = float(df['OBV'].iloc[-simple_lookback])
    
    swing_lookback = min(60, len(df) - 1) if len(df) > 1 else 1
    if swing_lookback > 1:
        past_df = df.iloc[-swing_lookback:-1]
        min_idx = past_df['Close'].idxmin()
        prev_close = float(df.loc[min_idx, 'Close'])
        prev_obv = float(df.loc[min_idx, 'OBV'])
        prev_rsi = float(df.loc[min_idx, 'RSI']) if not pd.isna(df.loc[min_idx, 'RSI']) else 50
    else:
        prev_close, prev_obv, prev_rsi = close, obv, rsi

    comments = {}
    
    rsi_status = "과매수" if rsi >= 70 else ("과매도" if rsi <= 30 else "안정")
    comments['RSI'] = f"현재 RSI 지수가 **{rsi:.1f}**를 기록하며 **{rsi_status}** 구간에 있습니다. "
    if rsi >= 70: comments['RSI'] += "이는 단기 고점 신호를 보내는 것으로, 신규 진입보다는 수익 실현을 우선적으로 고려해야 합니다."
    elif rsi <= 30: comments['RSI'] += "매도세가 쏠린 공포 구간입니다. 기술적 반등이 강하게 나타나는 타점이며, 진입 기회로 분석됩니다."
    else: comments['RSI'] += "현재 매수와 매도 세력이 팽팽하게 맞서고 있습니다. 기존 추세나 박스권 흐름을 이어갈 가능성이 큽니다."

    macd_diff = macd - signal
    macd_status = "상승" if macd > signal else "하락"
    comments['MACD'] = f"MACD 지표는 시그널선 대비 **{macd_diff:,.{decimals}f}** 차이로 **{macd_status} 추세**를 가리킵니다. "
    if macd > signal: comments['MACD'] += "단기 이평선이 장기선을 뚫고 올라온 상태로, 상승 모멘텀 확장 구간입니다."
    else: comments['MACD'] += "단기 모멘텀이 둔화되면서 하락 압력이 우세한 상황입니다. 조정 폭이 깊어질 수 있습니다."

    comments['VOL'] = f"상대 거래량이 최근 5{time_unit} 평균 대비 **{vol_ratio:.0f}%** 수준입니다. "
    if vol_ratio > 150: comments['VOL'] += "평소보다 대량 거래가 수반되고 있습니다. 에너지가 분출되며 변동성이 극대화될 확률이 높습니다."
    else: comments['VOL'] += "평이한 수준의 거래가 이뤄지고 있습니다. 횡보 장세가 예상됩니다."

    obv_trend = "유입" if obv > simple_prev_obv else "이탈"
    comments['OBV'] = f"최근 {simple_lookback}{time_unit}간 누적 OBV 수치가 **{'상승' if obv > simple_prev_obv else '하락'}**하며 자금이 **{obv_trend}**되고 있습니다. "
    if obv > simple_prev_obv: comments['OBV'] += "주가 움직임보다 앞서 수급이 개선되는 긍정적인 '숨은 매집' 단계일 가능성이 큽니다."
    else: comments['OBV'] += "주가가 버티더라도 스마트 머니는 이탈 중일 수 있으니 주의가 필요합니다."

    vol_pct = (atr / close) * 100 if close > 0 else 0
    comments['ATR'] = f"현재 {time_unit}당 평균 **{vol_pct:.1f}% ({atr:,.{decimals}f}{currency})**의 실질 변동폭을 보입니다. "
    comments['ATR'] += f"예상치 못한 노이즈를 피하기 위한 손절가 설정의 기준으로 참고하세요."

    dist_to_sup = (close - sup) / sup * 100 if sup > 0 else 100
    near_sup = dist_to_sup <= 5
    bullish_div = (close < prev_close) and (obv > prev_obv or rsi > prev_rsi)
    
    # 🌟 핵심 수정: 포지션 판별 로직 강화 (지표 교차 검증)
    if is_short_term:
        pos, strategy = "⚖️ 단기 관망", "뚜렷한 방향성이 확인되지 않아 관망을 권장합니다."
        if (rsi <= 40 and macd > signal) or (near_sup and bullish_div): 
            pos, strategy = "🔴 단기 적극 매수", "바닥권 수급 개선 및 기술적 반등 시그널이 명확합니다."
        elif (rsi >= 70 and macd < signal): 
            pos, strategy = "🔷 단기 적극 매도", "고점 과열 및 모멘텀 둔화로 리스크 관리가 시급합니다."
        elif macd > signal and obv > simple_prev_obv and rsi < 65: 
            pos, strategy = "🟠 단기 분할 매수", "수급(OBV)과 모멘텀(MACD)이 동반 개선되는 우상향 흐름입니다."
        elif macd < signal and obv < simple_prev_obv and rsi > 35: 
            pos, strategy = "🔵 단기 분할 매도", "수급 이탈과 모멘텀 약화가 진행 중이므로 비중 축소가 필요합니다."
    else:
        ma60 = latest['MA60'] if not pd.isna(latest['MA60']) else close
        pos, strategy = "⚖️ 장기 관망", "대세 전환 에너지를 응축하며 대기 중입니다."
        if close > ma60 and macd > signal and obv > simple_prev_obv: 
            pos, strategy = "🔴 비중 확대 (장기)", "주봉상 대세 상승장에 진입하여 수익 극대화가 가능합니다."
        elif close < ma60 and rsi < 35: 
            pos, strategy = "🟠 저점 매수 (장기)", "역사적 저평가 구간으로 분할 매집이 유효합니다."
        elif rsi > 75 or (close < ma60 and macd < signal): 
            pos, strategy = "🔷 비중 축소 (장기)", "추세 이탈 및 모멘텀 약화로 리스크 관리가 필요합니다."

    mode_str = "단기 스윙" if is_short_term else "장기 가치투자"
    trend_dir = "강한 상승 탄력" if macd > signal else "하락 압력 및 매물 소화"
    
    ai_op = f"🤖 **StockMap AI {mode_str} 심층 진단 리포트**\n\n"
    
    ai_op += f"📈 **[추세 및 모멘텀]**\n현재 이 종목은 {time_unit}봉 기준 **{trend_dir}** 국면에 놓여 있습니다. "
    if rsi >= 70: ai_op += "RSI가 과열권(70 이상)에 진입하여 단기 고점 징후가 뚜렷하므로, 신규 진입보다는 분할 수익 실현이 유리합니다.\n"
    elif rsi <= 30: ai_op += "RSI가 침체권(30 이하)에 머물며 투매가 진정되는 낙폭과대 구간으로, 강한 기술적 반등이 기대됩니다.\n"
    else: ai_op += f"RSI({rsi:.1f})는 중립 수준으로, 뚜렷한 과열이나 침체 없이 위아래로 방향성을 탐색하고 있습니다.\n"

    ai_op += f"\n📊 **[수급 및 에너지]**\n"
    if obv > simple_prev_obv: ai_op += "최근 주가의 등락과 무관하게 누적 수급(OBV)이 우상향하고 있어 세력의 '숨은 매집' 트렌드가 관찰됩니다. "
    else: ai_op += "누적 수급(OBV)이 하향 이탈하고 있어 자금 유출 우려가 있으니 방어적인 접근이 필요합니다. "
    if vol_ratio > 150: ai_op += f"또한 최근 거래량이 평균 대비 {vol_ratio:.0f}%로 폭증하며 의미 있는 에너지가 분출되었습니다.\n"
    else: ai_op += "현재 거래량은 평이한 수준을 유지 중입니다.\n"

    ai_op += f"\n🎯 **[타점 및 리스크 관리]**\n"
    if near_sup: 
        ai_op += f"현재 주가가 강력한 핵심 지지선(**{sup:,.{decimals}f}{currency}**)에 5% 이내로 근접하여 하방 경직성이 확보되었습니다. 손익비가 훌륭한 진입 타점입니다. "
    else:
        dist_to_res = (res - close) / close * 100 if close > 0 else 100
        if dist_to_res <= 5: ai_op += f"주요 저항선(**{res:,.{decimals}f}{currency}**) 돌파를 목전에 두고 있습니다. 저항 돌파 실패 시 실망 매물이 나올 수 있으니 안착을 확인하세요. "
        else: ai_op += f"현재 주요 지지선(**{sup:,.{decimals}f}{currency}**)과 저항선(**{res:,.{decimals}f}{currency}**) 사이의 중간 지대에 위치해 있습니다. "

    if bullish_div: 
        ai_op += "\n\n💡 **[핵심 패턴: 상승 다이버전스]** 스윙 로우(최근 저점) 대비 주가는 하락했으나 보조지표(RSI/OBV)는 오히려 상승하는 다이버전스가 포착되었습니다! 이는 추세 반전을 암시하는 매우 신뢰도 높은 매수 시그널입니다."

    comments['AI'] = f"{ai_op}\n\n🔥 **최종 투자 전략:** {strategy} (AI 권장 포지션: **{pos}**)"
    
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
        
        if not is_short_term:
            chart_df = raw_df.resample('W').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
            chart_df = calculate_indicators(chart_df)
            default_days = 730 # 장기: 2년
        else:
            chart_df = calculate_indicators(raw_df.copy())
            default_days = 180 # 단기: 6개월

        cur_price = raw_df['Close'].iloc[-1]
        diff = cur_price - raw_df['Close'].iloc[-2] if len(raw_df) > 1 else 0
        st.subheader(f"📑 {display_name} 리포트")
        st.metric("현재 주가", f"{cur_price:,.{decimals}f} {currency}", f"{diff:,.{decimals}f} {currency}")

        q_score = calculate_quant_score(chart_df, is_short_term)
        st.write(f"### 💯 퀀트 스코어: **{q_score}점**")
        st.progress(q_score / 100)

        pts, sup, res = detect_patterns_and_levels(chart_df)
        pos, strat, comments = generate_detailed_opinions(chart_df, sup, res, currency, decimals, is_short_term, time_unit)
        
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
                st.write(f"🛡️ **지지:** {sup:,.{decimals}f} {currency} | 🚧 **저항:** {res:,.{decimals}f} {currency}")

        with st.expander("🔬 지표별 상세 수치 분석 (용어 클릭)", expanded=True):
            for label, key in [("상대 거래량", "VOL"), ("OBV 누적", "OBV"), ("RSI 강도", "RSI"), ("MACD 흐름", "MACD"), ("ATR 변동성", "ATR")]:
                c1, c2 = st.columns([0.25, 0.75])
                with c1.popover(label, use_container_width=True): st.info(f"**{label}**")
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
            min_vals = plot_df[['Low', 'MA20', 'MA60']].min()
            max_vals = plot_df[['High', 'MA20', 'MA60']].max()
            c_min = min_vals.min()
            c_max = max_vals.max()
            padding = (c_max - c_min) * 0.05
            y_range = [c_min - padding, c_max + padding]
        else:
            y_range = None

        with tab1:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
            
            fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name='주가'), row=1, col=1)
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
