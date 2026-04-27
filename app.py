import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import FinanceDataReader as fdr
import requests
import re
from pykrx import stock
from datetime import datetime

# ==========================================
# 1. 페이지 설정 및 제목
# ==========================================
st.set_page_config(page_title="실시간 매수매도분석기", layout="wide")

st.title("📈 실시간 매수매도분석기")
st.markdown("---")

if 'recent_searches' not in st.session_state:
    st.session_state.recent_searches = []

# 🌟 봇 차단 우회를 위한 브라우저 헤더
REQ_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
}

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

@st.cache_data(ttl=3600)
def get_fundamentals(ticker):
    sector = 'KOR Equity'
    roe = None
    debt_ratio = None
    per = None
    pbr = None
    div_yield = None
    peer_per = None
    
    # 1. 야후 파이낸스 데이터 (클라우드 환경에서 에러 잦음 -> 독립 예외처리)
    try:
        info = yf.Ticker(ticker).info
        if info:
            sector = info.get('sector', 'KOR Equity')
            roe = info.get('returnOnEquity')
            if roe is not None: roe = round(roe * 100, 2)
            debt_ratio = info.get('debtToEquity')
            if debt_ratio is not None: debt_ratio = round(debt_ratio, 2)
            
            # 해외 주식 전용
            if not (ticker.endswith('.KS') or ticker.endswith('.KQ')):
                per = info.get('trailingPE') or info.get('forwardPE')
                pbr = info.get('priceToBook')
                div_yield = info.get('dividendYield')
                if div_yield: div_yield = round(div_yield * 100, 2)
                peer_per = info.get('trailingPE') 
    except:
        pass # 에러 무시하고 네이버 크롤링으로 넘어감

    # 2. 네이버 금융 데이터 (국내 주식 최우선 적용)
    if ticker.endswith('.KS') or ticker.endswith('.KQ'):
        try:
            code = ticker.split('.')[0]
            url = f"https://finance.naver.com/item/main.naver?code={code}"
            res = requests.get(url, headers=REQ_HEADERS)
            text = res.text
            
            per_match = re.search(r'<em id="_per">([\d.,]+)</em>', text)
            pbr_match = re.search(r'<em id="_pbr">([\d.,]+)</em>', text)
            div_match = re.search(r'<em id="_dvr">([\d.,]+)</em>', text)
            peer_per_match = re.search(r'동일업종 PER.*?<em>([\d.,]+)</em>', text, re.DOTALL)
            
            if per_match: per = float(per_match.group(1).replace(',', ''))
            if pbr_match: pbr = float(pbr_match.group(1).replace(',', ''))
            if div_match: div_yield = float(div_match.group(1).replace(',', ''))
            if peer_per_match: peer_per = float(peer_per_match.group(1).replace(',', ''))

            # 🌟 핵심 패치: 야후가 막혀도 네이버 표에서 ROE와 부채비율을 무조건 긁어오는 로직
            def extract_last_valid_float(row_html):
                tds = re.findall(r'<td[^>]*>(.*?)</td>', row_html, re.DOTALL)
                valid_vals = []
                for td in tds:
                    clean_str = re.sub(r'<[^>]+>', '', td).strip().replace(',', '')
                    try:
                        valid_vals.append(float(clean_str))
                    except ValueError:
                        pass
                return valid_vals[-1] if valid_vals else None

            roe_row = re.search(r'<th[^>]*>ROE\(%\)</th>(.*?)</tr>', text, re.DOTALL | re.IGNORECASE)
            if roe_row:
                val = extract_last_valid_float(roe_row.group(1))
                if val is not None: roe = val

            debt_row = re.search(r'<th[^>]*>부채비율</th>(.*?)</tr>', text, re.DOTALL | re.IGNORECASE)
            if debt_row:
                val = extract_last_valid_float(debt_row.group(1))
                if val is not None: debt_ratio = val

        except:
            pass
            
    return per, pbr, sector, div_yield, peer_per, roe, debt_ratio

@st.cache_data(ttl=3600)
def get_investor_trend(ticker):
    if not (ticker.endswith('.KS') or ticker.endswith('.KQ')):
        return None, None
        
    try:
        code = ticker.split('.')[0]
        
        hist = yf.Ticker(ticker).history(period="10d")
        if len(hist) < 5: return None, None
            
        start_date = hist.index[-5].strftime("%Y%m%d")
        end_date = hist.index[-1].strftime("%Y%m%d")
        
        df = stock.get_market_trading_volume_by_investor(start_date, end_date, code)
        inst_sum = int(df.loc['기관합계', '순매수'])
        fore_sum = int(df.loc['외국인', '순매수'])
        
        return inst_sum, fore_sum
    except:
        return None, None

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
    df['OBV'] = (vol * (close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)))).cumsum()
    
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
def get_stock_data(ticker, mode):
    period = "2y" if "장기" in mode else "1y"
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df.empty and ticker.endswith('.KS'):
            df = yf.Ticker(ticker.replace('.KS', '.KQ')).history(period=period)
        if df.empty: return pd.DataFrame()
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        return calculate_indicators(df)
    except: return pd.DataFrame()

def generate_signal_and_comments(df, mode, per, pbr, sector, peer_per, roe, debt_ratio, inst_sum, fore_sum):
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
    elif vol_ratio >= 150: comments['VOL'] = f"🌋 **수급 폭발 ({vol_ratio:.0f}%)**: 거래량이 평소보다 크게 터졌습니다! 세력 개입 가능성이 높습니다."
    elif vol_ratio >= 110: comments['VOL'] = f"🌊 **거래 활발 ({vol_ratio:.0f}%)**: 유의미한 거래량이 유입되고 있습니다."
    else: comments['VOL'] = f"💤 **소외/관망 ({vol_ratio:.0f}%)**: 거래량이 말라붙었습니다. 관심에서 멀어져 있습니다."

    if pd.isna(atr) or pd.isna(close) or close == 0:
        comments['ATR'] = "변동성 데이터를 계산할 수 없습니다."
    else:
        volatility_pct = (atr / close) * 100
        comments['ATR'] = f"현재 일평균 변동성은 {volatility_pct:.1f}% 수준입니다."

    f_lines = []
    if pbr is not None:
        pbr_threshold = 2.0 if any(s in str(sector).lower() for s in ['technology', 'healthcare', 'software', 'bio']) else 1.2
        if pbr < pbr_threshold * 0.8: f_lines.append(f"✅ **PBR({pbr:.2f})**: 자산 가치 대비 **저평가** 매력이 있습니다.")
        elif pbr > pbr_threshold * 1.5: f_lines.append(f"⚠️ **PBR({pbr:.2f})**: 장부 가치 대비 **고평가** 프리미엄이 상당합니다.")
        else: f_lines.append(f"➖ **PBR({pbr:.2f})**: 자산 가치 대비 적정 수준입니다.")
        
    if per is not None:
        if per < 0: f_lines.append("🚨 **PER(적자)**: 현재 순이익이 적자 상태입니다. 실적 턴어라운드 확인이 필요합니다.")
        elif peer_per is not None and peer_per > 0:
            if per < peer_per * 0.8: f_lines.append(f"✅ **PER({per:.2f})**: 동일업종 평균({peer_per:.2f}배) 대비 확실히 **저평가**되어 있습니다!")
            elif per > peer_per * 1.2: f_lines.append(f"🔥 **PER({per:.2f})**: 동일업종 평균({peer_per:.2f}배) 대비 주가가 **고평가**를 받고 있습니다.")
            else: f_lines.append(f"➖ **PER({per:.2f})**: 업종 내 경쟁사들과 **비슷한 평균 밸류에이션**을 적용받고 있습니다.")
            
    if roe is not None:
        if roe >= 15.0: f_lines.append(f"🏅 **ROE({roe:.2f}%)**: 워런 버핏의 기준(15%)을 통과한 **초우량 수익성**입니다.")
        elif roe >= 8.0: f_lines.append(f"✅ **ROE({roe:.2f}%)**: 안정적이고 양호한 수익성을 보여주고 있습니다.")
        else: f_lines.append(f"🚨 **ROE({roe:.2f}%)**: 자본 대비 수익성이 낮거나 적자 상태입니다.")
        
    if debt_ratio is not None:
        if debt_ratio <= 100.0: f_lines.append(f"🛡️ **부채비율({debt_ratio:.2f}%)**: 빚이 적고 재무가 **매우 건전**합니다.")
        elif debt_ratio > 200.0: f_lines.append(f"⚠️ **부채비율({debt_ratio:.2f}%)**: 부채비율이 높아 **재무 리스크**가 존재합니다.")

    comments['FUNDAMENTAL'] = f_lines if f_lines else ["해당 종목의 재무 데이터를 불러올 수 없습니다."]

    position, reason = "⚖️ 관망 (Neutral)", "추세 확인 후 진입을 권장합니다."
    t_buy, t_sell, s_loss = close * 0.95, close * 1.05, close * 0.90
    
    if not pd.isna(atr):
        if "단기" in mode:
            if not pd.isna(rsi) and rsi < 40 and macd_curr > sig_curr:
                position, reason = "🔴 적극 매수 (Strong Buy)", "과매도 부근 골든크로스 발생."
            elif not pd.isna(ma20) and close > ma20 and macd_curr > sig_curr:
                position, reason = "🟠 분할 매수 (Buy)", "20일선 위 안정적 상승 추세."
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

    with st.spinner(f"📡 '{display_name}' 분석 중..."):
        df = get_stock_data(ticker_symbol, analyze_mode)
        per, pbr, sector, div_yield, peer_per, roe, debt_ratio = get_fundamentals(ticker_symbol)
        inst_sum, fore_sum = get_investor_trend(ticker_symbol)
        
    if df.empty:
        st.error("데이터를 찾을 수 없습니다.")
    else:
        cur_price = df['Close'].iloc[-1]
        diff = cur_price - df['Close'].iloc[-2]
        currency = "원" if cur_price > 1000 else "USD"
        mode_badge = "단기" if "단기" in analyze_mode else "장기"
        
        st.subheader(f"📑 {display_name} 리포트 ({mode_badge})")
        st.metric("현재 주가", f"{cur_price:,.0f} {currency}" if cur_price > 1000 else f"{cur_price:,.2f} {currency}", f"{diff:,.0f} {currency}")

        q_score = calculate_quant_score(df)
        st.write("### 💯 퀀트 매수 매력도 점수")
        st.progress(q_score / 100)
        st.write(f"현재 점수: **{q_score}점** / 100점")

        pos, buy, sell, stop, reason, rsi, atr, comments = generate_signal_and_comments(
            df, analyze_mode, per, pbr, sector, peer_per, roe, debt_ratio, inst_sum, fore_sum
        )
        
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown("### 🏢 **기업 기초 체력**")
                peer_display = f" (업종평균: {peer_per}배)" if peer_per else ""
                roe_str = f"{roe}%" if roe is not None else "N/A"
                debt_str = f"{debt_ratio}%" if debt_ratio is not None else "N/A"
                st.markdown(f"**업종:** {sector} | **PER:** {per}배{peer_display} | **PBR:** {pbr}배<br>**ROE:** {roe_str} | **부채비율:** {debt_str} | **배당:** {div_yield}%", unsafe_allow_html=True)
                st.markdown("---")
                for fund_line in comments.get('FUNDAMENTAL', []):
                    st.markdown(fund_line)
                
        with col2:
            with st.container(border=True):
                st.markdown("### 🎯 **종합 매매 타이밍**")
                st.warning(f"**포지션: {pos}**\n\n**의견:** {reason}")
                st.write(f"진입가: {buy:,} | 목표가: {sell:,} | 손절가: {stop:,}")

        pts, sup, res = detect_patterns_and_levels(df)
        with st.container(border=True):
            st.markdown("### 🔍 **차트 패턴 및 주요 가격대**")
            p_text = ", ".join(pts) if pts else "특이 패턴 없음"
            st.write(f"📍 **발견된 패턴:** {p_text}")
            st.write(f"🛡️ **심리적 지지선:** {sup:,.0f} | 🚧 **강력 저항선:** {res:,.0f}")

        with st.expander("🔬 기술적 지표 상세 분석 보기", expanded=True):
            st.markdown(f"**[상대 거래량]** {comments.get('VOL', '데이터 없음')}")
            st.markdown(f"**[OBV 누적]** {comments.get('OBV', '데이터 없음')}")
            st.markdown(f"**[RSI 강도]** {comments.get('RSI', '데이터 없음')}")
            st.markdown(f"**[MACD 흐름]** {comments.get('MACD', '데이터 없음')}")
            st.markdown(f"**[ATR 변동성]** {comments.get('ATR', '데이터 없음')}")

        tab1, tab2 = st.tabs(["주가 차트", "수급(OBV) 차트"])
        chart_df = df.tail(120 if "단기" in analyze_mode else 250)
        
        with tab1:
            fig = go.Figure(data=[go.Candlestick(x=chart_df.index, open=chart_df['Open'], high=chart_df['High'], low=chart_df['Low'], close=chart_df['Close'], name='주가')])
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['MA20'], name='20일선', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['MA60'], name='60일선', line=dict(color='green')))
            fig.update_layout(height=500, xaxis_rangeslider_visible=False, margin=dict(t=0, b=0, l=0, r=0), dragmode=False)
            fig.update_xaxes(fixedrange=True); fig.update_yaxes(fixedrange=True)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
        with tab2:
            obv_fig = go.Figure(data=[go.Scatter(x=chart_df.index, y=chart_df['OBV'], name='OBV', fill='tozeroy', line=dict(color='purple'))])
            obv_fig.update_layout(height=400, title="누적 수급(OBV) 에너지", margin=dict(t=40, b=0, l=0, r=0), dragmode=False)
            obv_fig.update_xaxes(fixedrange=True); obv_fig.update_yaxes(fixedrange=True)
            st.plotly_chart(obv_fig, use_container_width=True, config={'displayModeBar': False})
