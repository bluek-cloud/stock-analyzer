import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import FinanceDataReader as fdr
from datetime import datetime, timedelta
import random # 🌟 AI 멘트 다변화를 위한 모듈 추가

# ==========================================
# 1. 페이지 설정 및 제목 (모바일 최적화)
# ==========================================
st.set_page_config(page_title="StockMap", layout="wide")

# 다크/라이트 테마 모두에서 글씨가 선명하게 보이도록 반투명 스타일 적용
st.markdown("""
    <style>
    .reportview-container .main .block-container { padding-top: 1rem; }
    [data-testid="stMetric"] { background-color: rgba(128, 128, 128, 0.1); padding: 15px; border-radius: 10px; border: 1px solid rgba(128, 128, 128, 0.2); }
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
    close = df['Close'].squeeze()
    high = df['High'].squeeze()
    low = df['Low'].squeeze()
    vol = df['Volume'].squeeze()

    df['MA20'] = close.rolling(window=20).mean()
    df['MA60'] = close.rolling(window=60).mean()
    df['MA200'] = close.rolling(window=200).mean()
    
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
    # 🌟 줌 아웃(확장 탐색)을 위해 기본적으로 5년치(1825일) 데이터를 한 번에 로드
    start_date = (datetime.now() - timedelta(days=1825)).strftime('%Y-%m-%d')
    try:
        df = fdr.DataReader(code, start=start_date)
        if df.empty: return pd.DataFrame()
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        return calculate_indicators(df)
    except: return pd.DataFrame()

# 🌟 궁극의 AI 엔진 탑재: 지지/저항 및 다이버전스 분석 포함
def generate_signal_and_comments(df, mode, sup, res):
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

    comments = {}
    
    # 기본 지표 설명 세팅
    if pd.isna(rsi): comments['RSI'] = "데이터 부족"
    elif rsi >= 70: comments['RSI'] = f"🔥 **과매수 (RSI: {rsi:.1f})**: 지수가 70을 넘어 단기 과열권입니다. 수익 실현을 고려할 시점입니다."
    elif rsi <= 30: comments['RSI'] = f"❄️ **과매도 (RSI: {rsi:.1f})**: 지수가 30 이하로 공포 구간입니다. 기술적 반등 가능성이 매우 높습니다."
    else: comments['RSI'] = f"📈 **정상 범위 (RSI: {rsi:.1f})**: 매수/매도세가 균형을 이루고 있습니다."

    if macd > signal:
        comments['MACD'] = "🚀 **상승 추세**: MACD가 시그널선을 상향 돌파하여 긍정적인 흐름을 유지하고 있습니다."
    else:
        comments['MACD'] = "⚠️ **하락 추세**: MACD가 시그널선 아래에 위치하여 단기 조정 압력이 존재합니다."

    obv_status = "상승" if obv > prev_obv else "하락"
    comments['VOL'] = f"🌋 **상대 거래량 ({vol_ratio:.0f}%)**: 평소 대비 유의미한 거래 에너지가 포착되었습니다." if vol_ratio > 150 else f"➖ **보통 거래량 ({vol_ratio:.0f}%)**: 평이한 수준의 거래가 이뤄지고 있습니다."
    comments['OBV'] = f"🕵️‍♂️ **매집 확인**: 최근 5일간 누적 OBV가 {obv_status}하며 자금의 흐름을 보여줍니다."
    
    volatility_pct = (atr / close) * 100
    comments['ATR'] = f"현재 주가는 일평균 **{volatility_pct:.1f}% ({int(atr):,}원)** 정도의 변동폭을 보이며 움직이고 있습니다."

    # 포지션 설정
    position, reason = "⚖️ 관망", "주요 지표의 방향성이 엇갈리고 있습니다."
    t_buy, t_sell, s_loss = close * 0.95, close * 1.05, close * 0.90
    
    if rsi < 40 and macd > signal:
        position, reason = "🔴 적극 매수", f"RSI({rsi:.1f}) 저평가 구간에서 MACD 골든크로스가 발생했습니다."
    elif rsi > 70 and macd < signal:
        position, reason = "🔷 적극 매도", f"RSI({rsi:.1f}) 과열권에서 추세가 꺾이기 시작했습니다."

    # ==========================================
    # 🤖 AI 딥다이브 분석 엔진 (LLM급 알고리즘)
    # ==========================================
    intro = random.choice([
        "🤖 **StockMap AI 심층 분석 리포트**\n\n",
        "🧠 **StockMap AI 퀀트 다이그노스틱**\n\n",
        "📊 **StockMap AI 알고리즘 종합 판단**\n\n"
    ])
    
    # 1. Price Action (지지/저항 융합 분석)
    dist_to_sup = (close - sup) / sup * 100
    dist_to_res = (res - close) / res * 100
    
    sr_text = ""
    if dist_to_sup <= 5:
        sr_text = random.choice([
            f"현재 주가는 심리적 주요 지지선(**{sup:,.0f}원**)에 매우 근접(이격도 {dist_to_sup:.1f}%)해 있습니다. 하방 경직성이 강하게 확보될 가능성이 높은 매력적인 자리입니다. ",
            f"주요 지지 레벨(**{sup:,.0f}원**) 부근에서 바닥 다지기가 진행 중입니다. 손절폭이 매우 짧아 신규 진입하기에 '가성비'가 뛰어난 구간으로 분석됩니다. "
        ])
    elif dist_to_res <= 5:
        sr_text = random.choice([
            f"현재 주가는 강력한 저항선(**{res:,.0f}원**)에 바짝 다가섰습니다(이격도 {dist_to_res:.1f}%). 이 구간의 완벽한 돌파 여부가 향후 단기 랠리를 결정지을 핵심 변수입니다. ",
            f"저항대(**{res:,.0f}원**) 돌파를 앞두고 치열한 매물 소화 과정이 관찰됩니다. 단번에 돌파하지 못할 경우 단기 차익 매물이 쏟아질 수 있습니다. "
        ])
    
    # 2. 고급 패턴: 다이버전스 (Divergence) 분석
    div_text = ""
    price_down = close < prev20_close
    price_up = close > prev20_close
    obv_up = obv > prev20_obv
    rsi_up = rsi > prev20_rsi
    
    if price_down and (obv_up or rsi_up):
        div_text = random.choice([
            "🔥 **[히든 시그널: 상승 다이버전스]** 겉보기엔 주가가 눌리고 있으나, 내부 보조지표(OBV/RSI)는 오히려 저점을 높이고 있습니다. 이는 세력의 '숨은 매집'을 암시하며 조만간 위로 크게 방향을 틀 확률이 큽니다. ",
            "💡 **[핵심 포착]** 가격은 하락/횡보 중인데 매수 에너지는 축적되는 전형적인 '상승 다이버전스' 현상이 포착되었습니다. 주가의 추세 반전이 임박했습니다. "
        ])
    elif price_up and (not obv_up or not rsi_up):
        div_text = random.choice([
            "⚠️ **[위험 신호: 하락 다이버전스]** 주가는 오르고 있지만 거래 수급이나 펀더멘탈 강도는 뒤따라오지 못하고 있습니다. 일명 '가짜 상승(속임수)'일 가능성이 있으니 강한 경계가 필요합니다. ",
            "🚨 **[주의 요망]** 표면적인 상승 추세와 달리 내부 에너지가 이탈하는 '하락 다이버전스'가 감지되었습니다. 추격 매수는 자칫 고점에 물릴 수 있는 매우 위험한 자리입니다. "
        ])

    # 3. 텍스트 조합
    ai_opinion = intro + sr_text + div_text
    
    if not sr_text and not div_text:
         ai_opinion += f"현재 주가는 지지선(**{sup:,.0f}원**)과 저항선(**{res:,.0f}원**) 사이의 밴드 내에서 안정적인 흐름을 이어가고 있습니다. "
         if macd > signal:
             ai_opinion += "MACD 지표가 시그널선을 상회하며 점진적인 우상향 모멘텀을 모색하는 단계로 평가됩니다. "
         else:
             ai_opinion += "MACD 데드크로스의 여파로 뚜렷한 방향성을 상실한 채 에너지를 비축하는 휴식 국면입니다. "

    if vol_ratio >= 150:
         ai_opinion += random.choice([
             f"\n\n특히 눈에 띄는 점은 최근 평소 대비 **{vol_ratio:.0f}%에 달하는 폭발적 거래량**이 유입되었다는 것입니다. 이는 시장의 강력한 주목을 받고 있다는 증거로, 현재의 추세를 가속화할 강력한 촉매제입니다.",
             f"\n\n여기에 거래량이 **{vol_ratio:.0f}% 수준으로 급증**한 점이 매우 고무적입니다. 스마트 머니(큰 손)의 개입이나 주요 재료 노출이 의심되니 변동성에 적극적으로 편승할 준비가 필요합니다."
         ])

    # 4. 결론 (투자 의견)
    if "매수" in position:
        ai_opinion += random.choice([
            "\n\n🎯 **최종 퀀트 전략:** 전반적인 보조지표들이 바닥권 탈출과 강세 전환을 한목소리로 지지하고 있습니다. **현 구간에서의 적극적인 분할 매수**는 통계적으로 승률이 매우 뛰어난 전략입니다.",
            "\n\n🎯 **최종 퀀트 전략:** 상승 랠리를 향한 퍼즐 조각들이 완벽히 맞춰지고 있습니다. **비중을 과감히 늘려가는 공세적인 포지션 구축**을 강력히 권장합니다."
        ])
    elif "매도" in position:
        ai_opinion += random.choice([
            "\n\n🎯 **최종 퀀트 전략:** 단기 과열 및 상승 추세 이탈 징후가 곳곳에서 감지됩니다. **신규 진입을 철저히 배제하고, 기존 보유 물량은 차익 실현**을 통해 계좌의 리스크를 방어하십시오.",
            "\n\n🎯 **최종 퀀트 전략:** 하방 압력이 거세질 수 있는 불안정한 위치입니다. **현금 비중을 대폭 늘리고 시장의 소나기를 피하는 보수적 접근**이 절대적으로 필요합니다."
        ])
    else:
        ai_opinion += random.choice([
            "\n\n🎯 **최종 퀀트 전략:** 상승과 하락 신호가 혼재되어 시장이 뚜렷한 답을 주지 않고 있습니다. **명확한 지지선 방어나 저항선 돌파가 확인될 때까지 관망**하는 것이 가장 훌륭한 투자입니다.",
            "\n\n🎯 **최종 퀀트 전략:** 현재는 폭발을 위해 에너지를 응축하는 지루한 횡보 구간입니다. **무리하게 배팅하기보다는 관망**하며 다음 결정적인 시그널을 기다리시길 바랍니다."
        ])

    comments['AI'] = ai_opinion
    return position, t_buy, t_sell, s_loss, reason, rsi, atr, comments

# ==========================================
# 3. 사이드바 및 실행 UI
# ==========================================
with st.sidebar:
    st.header("⚙️ 분석 설정")
    analyze_mode = st.radio("투자 성향 (초기 차트 뷰)", ["단기/스윙 (6개월)", "장기 투자 (2년)"])
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

    with st.spinner(f"📡 '{display_name}' 퀀트 딥다이브 분석 중..."):
        df = get_stock_data(ticker_symbol)
        
    if df.empty:
        st.error("데이터를 찾을 수 없습니다. 종목명이나 코드를 다시 확인해주세요.")
    else:
        cur_price = df['Close'].iloc[-1]
        diff = cur_price - df['Close'].iloc[-2]
        currency = "원"
        
        st.subheader(f"📑 {display_name} 리포트")
        st.metric("현재 주가", f"{cur_price:,.0f} {currency}", f"{diff:,.0f} {currency}")

        q_score = calculate_quant_score(df)
        st.write(f"### 💯 퀀트 스코어: **{q_score}점**")
        st.progress(q_score / 100)

        # 🌟 지지/저항을 먼저 뽑아내어 AI 코멘트에 전달
        pts, sup, res = detect_patterns_and_levels(df)
        pos, buy, sell, stop, reason, rsi, atr, comments = generate_signal_and_comments(df, analyze_mode, sup, res)
        
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown("### 🎯 **종합 매매 타이밍**")
                st.warning(f"**포지션: {pos}**\n\n**요약의견:** {reason}")
                st.write(f"진입가: {buy:,.0f} | 목표가: {sell:,.0f}\n\n손절가: {stop:,.0f}")
                
        with col2:
            with st.container(border=True):
                st.markdown("### 🔍 **차트 패턴 및 지지/저항**")
                p_text = ", ".join(patterns) if (patterns := pts) else "특이 패턴 없음"
                st.write(f"📍 **패턴:** {p_text}")
                st.write(f"🛡️ **지지:** {sup:,.0f} | 🚧 **저항:** {res:,.0f}")

        with st.expander("🔬 지표별 상세 수치 분석", expanded=True):
            # 팝오버를 통한 개념 설명 및 우측 수치 출력
            c1, c2 = st.columns([0.2, 0.8])
            with c1.popover("상대 거래량", use_container_width=True):
                st.info("**상대 거래량(Relative Volume)**\n\n최근 5일 평균 거래량 대비 오늘 거래량이 얼마나 터졌는지를 나타냅니다. 150~200% 이상이면 세력 유입이나 강한 추세 변화의 신호로 봅니다.")
            c2.markdown(comments.get('VOL'))
            
            c1, c2 = st.columns([0.2, 0.8])
            with c1.popover("OBV 누적", use_container_width=True):
                st.info("**OBV(On-Balance Volume)**\n\n거래량은 주가에 선행한다는 원리를 이용한 지표입니다. 주가가 하락해도 OBV가 상승하면 '숨은 매집'으로 판단하며, 반대의 경우 '이탈 징후'로 봅니다.")
            c2.markdown(comments.get('OBV'))
            
            c1, c2 = st.columns([0.2, 0.8])
            with c1.popover("RSI 강도", use_container_width=True):
                st.info("**RSI(상대강도지수)**\n\n주가의 상승 압력과 하락 압력 간의 상대적 강도를 나타냅니다. 70 이상은 '과매수(거품)', 30 이하는 '과매도(저평가)' 구간으로 해석합니다.")
            c2.markdown(comments.get('RSI'))
            
            c1, c2 = st.columns([0.2, 0.8])
            with c1.popover("MACD 흐름", use_container_width=True):
                st.info("**MACD(이동평균 수렴확산)**\n\n단기 추세선과 장기 추세선이 얼마나 가까워지고 멀어지는지를 측정합니다. 골든크로스가 발생하면 상승 추세의 시작으로 봅니다.")
            c2.markdown(comments.get('MACD'))
            
            c1, c2 = st.columns([0.2, 0.8])
            with c1.popover("ATR 변동성", use_container_width=True):
                st.info("**ATR(평균 실변동폭)**\n\n일정 기간 동안 주가가 얼마나 '출렁'거렸는지 변동성을 보여줍니다. ATR이 높을수록 주가가 급등락하기 쉬우므로 위험 관리가 필요합니다.")
            c2.markdown(comments.get('ATR'))
            
            st.divider()
            
            # 🌟 궁극의 AI 분석 리포트 출력 구역
            st.info(comments.get('AI'))

        # ==========================================
        # 🌟 5년 전체 탐색 지원 핀치 줌 차트 세팅
        # ==========================================
        tab1, tab2 = st.tabs(["주가 차트", "수급(OBV) 차트"])
        chart_df = df # 데이터 자르기(.tail) 없이 5년 치 통째로 매핑
        
        # 사용자의 투자 성향에 따라 초기 뷰포트(줌 레벨) 설정
        view_days = 180 if "단기" in analyze_mode else 730
        initial_start = datetime.now() - timedelta(days=view_days)
        initial_end = datetime.now()
        
        with tab1:
            fig = go.Figure(data=[go.Candlestick(x=chart_df.index, open=chart_df['Open'], high=chart_df['High'], low=chart_df['Low'], close=chart_df['Close'], name='주가')])
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['MA20'], name='20일선', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['MA60'], name='60일선', line=dict(color='green')))
            
            fig.update_layout(
                height=450, 
                xaxis=dict(
                    range=[initial_start, initial_end], # 초기엔 6개월/2년만 보임 (손가락으로 줌아웃 시 5년 탐색 가능)
                    rangeslider=dict(visible=False)
                ),
                margin=dict(t=10, b=10, l=0, r=0),
                dragmode='pan', # 모바일 친화적 드래그
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True, config={
                'scrollZoom': True, # 핀치 줌 완벽 지원
                'displayModeBar': False,
                'doubleClick': 'reset+autosize'
            })
            
        with tab2:
            obv_fig = go.Figure(data=[go.Scatter(x=chart_df.index, y=chart_df['OBV'], name='OBV', fill='tozeroy', line=dict(color='purple'))])
            obv_fig.update_layout(
                height=350, 
                xaxis=dict(range=[initial_start, initial_end]), # 위 차트와 뷰포트 동기화
                margin=dict(t=10, b=10, l=0, r=0), 
                dragmode='pan'
            )
            st.plotly_chart(obv_fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': False})
