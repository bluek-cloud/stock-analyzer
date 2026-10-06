import requests
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import FinanceDataReader as fdr
from setup_signals import normalize_price_data

def get_investor_net_buys(code):
    """
    네이버 증권 모바일 API를 통해 최근 20영업일 외국인/기관(사모펀드 포함) 순매수 수량 조회
    """
    try:
        url = f"https://m.stock.naver.com/api/stock/{code}/trend?pageSize=20"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        r = requests.get(url, headers=headers, timeout=3)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and len(data) >= 20:
                frgn_sum = 0
                inst_sum = 0
                for row in data[:20]:
                    f_val = str(row['foreignerPureBuyQuant']).replace(',', '')
                    i_val = str(row['organPureBuyQuant']).replace(',', '')
                    frgn_sum += int(f_val)
                    inst_sum += int(i_val)
                return frgn_sum, inst_sum, (frgn_sum + inst_sum)
    except Exception:
        pass
    return None


def filter_universe_candidates(krx_df, scan_scope=500):
    """
    Phase 1. 유니버스 필터링 (기본 건전성 및 종목 제외)
    - 시가총액: 최소 1,000억 원 이상
    - 제외: ETF, ETN, SPAC(스팩), 리츠(REITs), 우선주, 관리종목/환기종목
    """
    if krx_df.empty:
        return []

    df = krx_df.copy()
    df['Marcap'] = pd.to_numeric(df['Marcap'], errors='coerce')
    
    # Unknown market caps cannot satisfy the minimum-size condition.
    df = df[df['Marcap'].notna() & (df['Marcap'] >= 1e11)]

    # 2. 우선주 제외 (코드 끝자리 0이 아니거나 종목명에 우선주 식별자 포함)
    df = df[df['Code'].astype(str).str.endswith('0')]
    # A normal company can end in '우' (e.g. 성우); code filtering already
    # excludes ordinary preferred-share codes. Keep only explicit name markers.
    pref_patterns = ['우B', '우C', '1우', '2우B', '3우B', '우선주']
    pref_regex = '|'.join([f"{p}$" for p in pref_patterns]) + '|우선주'
    df = df[~df['Name'].str.contains(pref_regex, regex=True, na=False)]

    # 3. ETF, ETN, SPAC, 리츠, 인프라 펀드 제외
    exclude_keywords = ['스팩', 'SPAC', '리츠', 'REIT', '투융자', 'ETF', 'ETN', '맥쿼리인프라']
    fund_brands = ['KODEX', 'TIGER', 'KBSTAR', 'ACE', 'ARIRANG',
        'SOL', 'PLUS', 'HANARO', 'TIMEFOLIO', 'HERO', 'KOSEF',
        '파워', '마이티', 'FOCUS', 'UNIE', 'KoAct', 'WON'
    ]
    ex_pattern = '|'.join(map(re.escape, exclude_keywords)) + r'|^(?:' + '|'.join(map(re.escape, fund_brands)) + r')\s'
    df = df[~df['Name'].str.contains(ex_pattern, case=False, na=False)]

    # 4. 관리종목 / 투자주의환기종목 제외
    if 'Dept' in df.columns:
        df = df[~df['Dept'].fillna('').astype(str).str.contains('관리|환기', na=False)]
    managed_name_pattern = r'\(관리\)|\(환기\)|\(경고\)|\(주의\)|\(위험\)|\(정지\)'
    df = df[~df['Name'].str.contains(managed_name_pattern, regex=True, na=False)]

    # 시가총액 순 정렬 후 스캔 범위 지정
    df = df.sort_values('Marcap', ascending=False)
    if scan_scope and scan_scope > 0 and scan_scope < len(df):
        df = df.head(scan_scope)

    return df.to_dict('records')


def evaluate_stock_accumulation_df(df, code="", name="", marcap=0, min_accum_candles=2, check_investor=True):
    """
    기존 주가 데이터(DataFrame)를 기반으로 Phase 1 ~ Phase 4 세력 매집 알고리즘을 정밀 검증
    """
    if len(df) < 100:
        return None
    df = normalize_price_data(df)

    # -------------------------------------------------------------
    # Phase 1. 유동성 및 거래정지 체크
    # -------------------------------------------------------------
    # 1-1. 거래정지 체크 (최근 거래량 0 또는 비정상 데이터)
    if df['Volume'].iloc[-1] <= 0:
        return None

    # 1-2. 최근 20영업일 평균 거래대금 10억 원 이상
    recent_20_amount = (df['Close'].iloc[-20:] * df['Volume'].iloc[-20:]).mean()
    if recent_20_amount < 1e9:  # 10억 원 미만 탈락
        return None

    # -------------------------------------------------------------
    # Phase 2 & 3 모드별 정밀 임계값 설정 (다이아몬드 엄격 vs 유망 후보)
    # -------------------------------------------------------------
    is_strict = (min_accum_candles >= 2)
    max_pct_low = 40.0 if is_strict else 45.0
    max_ma_disp = 6.0 if is_strict else 6.5
    cum_ret_min = -5.0
    cum_ret_max = 10.0 if is_strict else 12.0
    min_vol_growth = 95.0
    min_spike_mult = 2.5

    # -------------------------------------------------------------
    # Phase 2. 가격 위치 및 이평선 수렴 조건
    # -------------------------------------------------------------
    cur_close = float(df['Close'].iloc[-1])

    # 2-1. 장기 바닥권: 현재 종가가 최근 1년(250영업일) 최저가 대비 안정적 위치
    lookback = min(250, len(df))
    min_250 = float(df['Low'].iloc[-lookback:].min())
    pct_from_low = ((cur_close - min_250) / (min_250 + 1e-10)) * 100
    if pct_from_low > max_pct_low:
        return None

    # 2-2. 이평선 수렴: MA20과 MA60의 이격도 밀집
    ma20 = float(df['Close'].rolling(20).mean().iloc[-1])
    ma60 = float(df['Close'].rolling(60).mean().iloc[-1])
    if pd.isna(ma20) or pd.isna(ma60):
        return None
    ma_disp = abs(ma20 - ma60) / (ma60 + 1e-10) * 100
    if ma_disp > max_ma_disp:
        return None

    # 2-3. 기간 변동폭 억제: 최근 20영업일 누적 등락률 유지
    ref_close_20 = float(df['Close'].iloc[-20])
    cum_ret = ((cur_close - ref_close_20) / (ref_close_20 + 1e-10)) * 100
    if not (cum_ret_min <= cum_ret <= cum_ret_max):
        return None

    # -------------------------------------------------------------
    # Phase 3. 거래량 폭증 및 매집봉 감지
    # -------------------------------------------------------------
    # 3-1. 평균 거래량 대비: 최근 20영업일 평균 거래량이 직전 60영업일 대비 유지 확인
    vol_20 = float(df['Volume'].iloc[-20:].mean())
    vol_prev_60 = float(df['Volume'].iloc[-80:-20].mean()) if len(df) >= 80 else float(df['Volume'].iloc[:-20].mean())
    vol_growth = (vol_20 / (vol_prev_60 + 1e-10)) * 100

    # 3-2. 매집봉 조건 (최근 20영업일 이내 최소 N회 이상)
    accum_candles = []
    has_super_spike = False
    n = len(df)
    for idx in range(n - 20, n):
        c_open = float(df['Open'].iloc[idx])
        c_high = float(df['High'].iloc[idx])
        c_low = float(df['Low'].iloc[idx])
        c_close = float(df['Close'].iloc[idx])
        c_vol = float(df['Volume'].iloc[idx])

        # a) 당일 거래량이 직전 20일 평균 거래량 대비 250% 이상 (2.5배 이상 폭증)
        prior_20_vol_avg = float(df['Volume'].iloc[max(0, idx-20):idx].mean())
        if prior_20_vol_avg <= 0 or c_vol < prior_20_vol_avg * min_spike_mult:
            continue

        c_range = c_high - c_low
        # b) 캔들 형태: 양봉(C >= O) 또는 윗꼬리 도지 형태, 장대 음봉 제외
        is_bullish = (c_close >= c_open)
        is_doji = (c_range > 0 and abs(c_close - c_open) <= c_range * 0.20 and (c_high - max(c_open, c_close)) >= c_range * 0.35)
        if not (is_bullish or is_doji):
            continue
        # 장대 음봉(고점 대비 크게 밀려 저가 부근에서 마감한 음봉) 철저 배제
        if c_close < c_open:
            if abs(c_open - c_close) > c_range * 0.4:
                continue
            if (c_close - c_low) / (c_range + 1e-10) < 0.25:
                continue

        # c) 지지 확인: 해당 매집봉 발생 이후 현재까지 종가가 매집봉의 저가를 하회한 적 없을 것
        if idx < n - 1:
            subsequent_min_close = float(df['Close'].iloc[idx+1:].min())
            if subsequent_min_close < c_low:
                continue

        date_str = df.index[idx].strftime('%Y-%m-%d') if hasattr(df.index[idx], 'strftime') else str(df.index[idx])
        vol_ratio_spike = round((c_vol / prior_20_vol_avg) * 100)
        if vol_ratio_spike >= 300:
            has_super_spike = True
        accum_candles.append({
            'date': date_str,
            'vol_ratio': vol_ratio_spike,
            'candle_low': int(c_low)
        })

    # 3-1 거래량 검증: 평균 거래량 증가율 확인 (또는 300% 이상 단일 초강력 매집봉 터진 경우 인정)
    if vol_growth < min_vol_growth and not has_super_spike:
        return None

    if len(accum_candles) < min_accum_candles:
        return None

    # -------------------------------------------------------------
    # Phase 4. 수급 가점 (외국인 + 기관 순매수 조회)
    # -------------------------------------------------------------
    frgn_sum, inst_sum, total_smart_money = (None, None, None)
    badge = "⚪ 수급 미확인"
    has_smart_money = False

    investor_data = get_investor_net_buys(code) if check_investor and code else None
    if investor_data is not None:
        frgn_sum, inst_sum, total_smart_money = investor_data
        has_smart_money = total_smart_money > 0
        if frgn_sum > 0 and inst_sum > 0:
            badge = "💎 외인·기관 쌍끌이"
        elif total_smart_money > 0:
            badge = "🔥 수급 일치"
        elif inst_sum > 0:
            badge = "🔶 기관 순매수"
        elif frgn_sum > 0:
            badge = "🔷 외인 순매수"
        else:
            badge = "⚪ 외인·기관 순매수 없음"

    return {
        'code': code,
        'name': name,
        'marcap_eok': round(marcap / 1e8) if marcap > 0 else 0,
        'current_price': int(cur_close),
        'amount_20_eok': round(recent_20_amount / 1e8, 1),
        'pct_from_low': round(pct_from_low, 1),
        'ma_disp': round(ma_disp, 1),
        'cum_ret': round(cum_ret, 1),
        'vol_growth': round(vol_growth, 1),
        'accum_count': len(accum_candles),
        'accum_details': accum_candles,
        'accum_dates_str': ", ".join([f"{c['date']}({c['vol_ratio']}%)" for c in accum_candles]),
        'frgn_sum': frgn_sum,
        'inst_sum': inst_sum,
        'total_smart_money': total_smart_money,
        'has_smart_money': has_smart_money,
        'investor_data_available': investor_data is not None,
        'badge': badge
    }


def evaluate_stock_accumulation(stock, start_date, min_accum_candles=2):
    """
    단일 종목 딕셔너리를 받아 FDR로 다운로드 후 검증 수행
    """
    code = stock['Code']
    name = stock['Name']
    marcap = stock.get('Marcap', 0)

    df = fdr.DataReader(code, start=start_date)
    if df.empty:
        raise ValueError('시세 응답이 비어 있습니다.')
    return evaluate_stock_accumulation_df(df, code=code, name=name, marcap=marcap, min_accum_candles=min_accum_candles, check_investor=True)


def scan_smart_money_stocks(krx_df, scan_scope=500, min_accum_candles=2, progress_bar=None, status_text=None):
    """
    전체 유니버스 대상 병렬 고속 스캔 실행 함수
    """
    candidates = filter_universe_candidates(krx_df, scan_scope=scan_scope)
    total_count = len(candidates)
    if total_count == 0:
        empty = pd.DataFrame()
        empty.attrs['scan_summary'] = {'total': 0, 'succeeded': 0, 'failed': 0, 'errors': []}
        return empty

    start_date = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')
    results = []
    completed = 0
    errors = []

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(evaluate_stock_accumulation, stock, start_date, min_accum_candles): stock for stock in candidates}
        for future in as_completed(futures):
            try:
                res = future.result()
            except Exception as exc:
                stock = futures[future]
                errors.append({'종목코드': stock['Code'], '종목명': stock['Name'],
                               '오류': str(exc) if isinstance(exc, ValueError) else type(exc).__name__})
                res = None
            if res is not None:
                results.append(res)
            completed += 1
            if progress_bar is not None:
                progress_bar.progress(completed / total_count)
            if status_text is not None:
                status_text.caption(f"📡 세력 매집 알고리즘 고속 병렬 스캔 중... ({completed}/{total_count} 종목 완료)")

    # Phase 4. 수급 가점 상위 랭킹 정렬
    # 1순위: 외인+기관 합산 순매수 양수(True) 여부
    # 2순위: 매집봉 발생 횟수 (많은 순)
    # 3순위: 외인+기관 합산 순매수량
    # 4순위: 거래량 증가율 (높은 순)
    results.sort(key=lambda x: (
        1 if x['has_smart_money'] else 0,
        x['accum_count'],
        x['total_smart_money'] if x['total_smart_money'] is not None else float('-inf'),
        x['vol_growth']
    ), reverse=True)

    output = pd.DataFrame(results)
    output.attrs['scan_summary'] = {'total': total_count, 'succeeded': total_count-len(errors),
                                    'failed': len(errors), 'errors': errors}
    return output
