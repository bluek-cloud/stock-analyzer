import os
import yaml
import pandas as pd
import numpy as np
from backtest_engine import match_current_setup, format_stats_for_report

# ==========================================
# 1. 룰셋 로더 및 캐시 관리
# ==========================================
RULE_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rules', 'trading_rules.yaml')
_CACHED_RULES = None

def load_rules(force_reload=False):
    """YAML 전략 룰셋을 로드하고 메모리에 캐싱"""
    global _CACHED_RULES
    if _CACHED_RULES is not None and not force_reload:
        return _CACHED_RULES

    if os.path.exists(RULE_FILE_PATH):
        try:
            with open(RULE_FILE_PATH, 'r', encoding='utf-8') as f:
                _CACHED_RULES = yaml.safe_load(f)
                return _CACHED_RULES
        except Exception:
            pass

    # 기본 폴백 룰셋
    _CACHED_RULES = {
        'regimes': {},
        'mtf_narratives': {},
        'traps': {},
        'quant_validation': {
            'low_threshold': 30,
            'high_threshold': 70,
            'buy_positions': [],
            'sell_positions': [],
            'bottom_fishing_positions': []
        }
    }
    return _CACHED_RULES


# ==========================================
# 2. 시장 국면(Regime) 판정 엔진
# ==========================================
def evaluate_market_regime(df, latest):
    """차트 지표를 기반으로 시장 국면(Regime)을 자동 판정"""
    if len(df) < 30:
        return "데이터 수집/안정화 중"

    close = float(latest['Close'])
    ma20 = float(latest['MA20']) if 'MA20' in latest and not pd.isna(latest['MA20']) else None
    ma60 = float(latest['MA60']) if 'MA60' in latest and not pd.isna(latest['MA60']) else None
    adx = float(latest['ADX']) if 'ADX' in latest and not pd.isna(latest['ADX']) else None
    p_di = float(latest['+DI']) if '+DI' in latest and not pd.isna(latest['+DI']) else 0.0
    m_di = float(latest['-DI']) if '-DI' in latest and not pd.isna(latest['-DI']) else 0.0
    vol_ratio = float(latest['Vol_Ratio']) if 'Vol_Ratio' in latest and not pd.isna(latest['Vol_Ratio']) else 0.0
    bbw = float(latest['BBW']) if 'BBW' in latest and not pd.isna(latest['BBW']) else None

    # 볼린저 밴드 스퀴즈
    is_squeeze = (bbw <= df['BBW'].iloc[-120:].min() * 1.05) if (len(df) > 120 and bbw is not None) else False
    if is_squeeze:
        return "에너지 응축 (스퀴즈)"

    # 변동성 폭발
    has_valid_adx = adx is not None
    if vol_ratio >= 150 and has_valid_adx and adx > float(df['ADX'].iloc[-2]) and adx > 20:
        return "변동성 폭발"

    # 이평선 정배열/역배열 기반 추세
    has_valid_ma = (ma20 is not None) and (ma60 is not None)
    if has_valid_ma and ma20 >= ma60 and close >= ma60:
        return "강세 추세" if p_di > m_di else "상승 조정"
    elif has_valid_ma and ma20 < ma60 and close < ma60:
        return "약세 추세"

    return "횡보 박스"


# ==========================================
# 3. 보조 지표 및 다이버전스 분석
# ==========================================
def detect_bullish_divergence(df):
    """주가 신저가 형성 시 RSI 또는 OBV의 저점 상승(상승 다이버전스) 감지"""
    if len(df) < 15:
        return False

    recent_chunk = df.iloc[-4:]
    past_chunk = df.iloc[-30:-4] if len(df) >= 30 else df.iloc[:-4]
    if past_chunk.empty or recent_chunk.empty:
        return False

    p_min_idx = past_chunk['Low'].idxmin()
    r_min_idx = recent_chunk['Low'].idxmin()
    p_low = float(past_chunk.loc[p_min_idx, 'Low'])
    r_low = float(recent_chunk.loc[r_min_idx, 'Low'])

    # 최근 저가가 이전 저가 이하이거나 거의 유사(신저가 형성)
    if r_low <= p_low * 1.01:
        p_rsi = float(past_chunk.loc[p_min_idx, 'RSI']) if ('RSI' in past_chunk.columns and not pd.isna(past_chunk.loc[p_min_idx, 'RSI'])) else None
        r_rsi = float(recent_chunk.loc[r_min_idx, 'RSI']) if ('RSI' in recent_chunk.columns and not pd.isna(recent_chunk.loc[r_min_idx, 'RSI'])) else None
        p_obv = float(past_chunk.loc[p_min_idx, 'OBV']) if ('OBV' in past_chunk.columns and not pd.isna(past_chunk.loc[p_min_idx, 'OBV'])) else None
        r_obv = float(recent_chunk.loc[r_min_idx, 'OBV']) if ('OBV' in recent_chunk.columns and not pd.isna(recent_chunk.loc[r_min_idx, 'OBV'])) else None

        if (r_rsi is not None and p_rsi is not None and r_rsi > p_rsi + 2.0) or \
           (r_obv is not None and p_obv is not None and r_obv > p_obv):
            return True
    return False


# ==========================================
# 4. 전략 및 포지션 도출 엔진
# ==========================================
def evaluate_strategy(ctx):
    """현재 차트 상태 컨텍스트(ctx)를 바탕으로 포지션과 전략 해설을 도출"""
    is_short_term = ctx['is_short_term']
    is_falling_knife = ctx['is_falling_knife']
    res = ctx['res']
    close = ctx['close']
    prev_close = ctx['prev_close']
    prev_candle_close = ctx['prev_candle_close']
    regime = ctx['regime']
    bullish_div = ctx['bullish_div']
    ma20 = ctx['ma20']
    obv = ctx['obv']
    simple_prev_obv = ctx['simple_prev_obv']
    box_pos = ctx['box_pos']
    vol_ratio = ctx['vol_ratio']
    rsi = ctx['rsi']
    adx = ctx['adx']

    if is_short_term:
        if is_falling_knife:
            return "🔷 투매 진행 중 (절대 관망)", "대량 거래를 동반한 치명적 급락 발생. '떨어지는 칼날'이므로 하락 진정 시까지 절대 관망하십시오."
        elif res == 0 and close > prev_close:
            return "🔴 신고가 랠리 (강력 홀딩)", "과거 매물대를 모두 뚫어낸 신고가 영역입니다. 추세 훼손 전까지 수익을 극대화하십시오."
        elif regime == "에너지 응축 (스퀴즈)":
            if bullish_div or (close > ma20 and obv > simple_prev_obv):
                return "🔴 상방 분출 기대 선취매", "에너지 응축 구간이나, 주가가 중심선(20일선) 위에 있고 수급이 유입 중입니다. 상방 폭발에 대비한 선취매가 유효합니다."
            elif close < ma20 and obv < simple_prev_obv:
                return "🔷 하방 이탈 경계 (관망)", "에너지 응축 구간이며 주가가 중심선 아래에 있고 수급이 이탈 중입니다. 하방 폭락 위험이 있으니 관망하십시오."
            else:
                return "⚖️ 방향성 대기 (관망)", "볼린저 밴드 극도 수축 상태. 뚜렷한 방향성 분출 전까지 관망하십시오."
        elif regime == "횡보 박스":
            if box_pos <= 35 or bullish_div:
                return "🟠 박스권 하단 매수", "박스권 하단 지지 확인 및 반전 시그널 발생. 상단을 목표로 한 단기 스윙 전략이 유효합니다."
            elif box_pos >= 65:
                if obv > simple_prev_obv and vol_ratio >= 100:
                    return "🟠 돌파 기대 (보유)", "저항선 근접했으나 긍정적 수급과 거래량 유입 중. 돌파 여부를 예의주시하며 홀딩을 권장합니다."
                elif obv > simple_prev_obv and vol_ratio < 100:
                    return "⚖️ 저항 돌파 탐색 (관망)", "수급(OBV)은 양호하나 돌파를 확정짓기엔 거래량이 부족합니다. [신규] 돌파 확인 전까지 추격 매수를 자제하십시오. [보유자] 거래량 동반 돌파 시 홀딩하고, 저항 맞고 음봉 이탈 시에만 분할 익절로 대응하십시오."
                else:
                    return "🔵 단기 박스권 상단 매도", "저항선 부근이나 수급(OBV)마저 이탈 중입니다. 돌파 가능성이 낮으므로 리스크 관리를 위해 비중 축소를 권장합니다."
            elif close > ma20 and obv > simple_prev_obv:
                return "🟠 박스권 중심 반등 공략", "박스권 중간 지대이나 중심선(20일선)을 회복하며 수급이 유입되고 있습니다. 박스 상단을 목표로 한 짧은 스윙이 가능합니다."
            else:
                return "⚖️ 단기 관망", "박스권 중간 지대 위치. 뚜렷한 타점 도달 전까지 진입을 자제하십시오."
        elif regime in ["강세 추세", "상승 조정"]:
            if rsi <= 55 or bullish_div:
                return "🔴 추세 눌림목 적극 매수", "강한 상승 추세 속 건전한 눌림목 발생. 확률 높은 매수 타점으로 평가됩니다."
            elif rsi >= 70 and adx < 30:
                return "🔵 분할 익절", "단기 과열권 진입이며 추세 강도(ADX)도 약해지고 있습니다. 수익 보호를 위해 보유 비중 분할 실현을 권장합니다."
            elif rsi >= 70 and adx >= 30:
                return "🟠 추세 보유 (홀딩)", "단기 과열권이나 추세 강도(ADX)가 강력하여 추가 상승 여력이 있습니다. 추세 이탈 전까지 홀딩하십시오."
            else:
                return "🟠 추세 보유 (홀딩)", "우상향 흐름 진행 중. 상승 추세 이탈 전까지 지속 보유하여 수익을 극대화하십시오."
        elif regime == "약세 추세":
            if rsi >= 45 and close > prev_close:
                if obv > simple_prev_obv and vol_ratio > 100:
                    return "🟠 의미 있는 반등 시도", "하락장 속 유의미한 수급/거래량 동반 반등. 추세 전환의 단초가 될 수 있으나 신중하게 접근하십시오."
                elif obv > simple_prev_obv:
                    return "⚖️ 반등 관찰 (관망)", "수급은 개선되나 거래량 뒷받침이 미흡합니다. 진입보다 추가 확인이 필요한 시점입니다."
                else:
                    return "🔵 데드캣 바운스 경계 (매도)", "수급과 거래량 모두 뒷받침이 없는 단순 기술적 반등입니다. 보유자는 탈출 기회로 삼으십시오."
            elif rsi <= 30 or bullish_div:
                return "🟠 단기 기술적 반등 공략", "극단적 과매도 및 다이버전스 발생. 짧은 수익을 목표로 한 기술적 반등 매매만 권장합니다."
            else:
                return "🔷 적극 매도 및 관망", "하락 추세가 지배적입니다. 물타기를 자제하고 현금 비중을 높여 관망하십시오."
        elif regime == "변동성 폭발":
            if close > prev_candle_close:
                return "🔴 돌파 추세 추종", "평균을 상회하는 대량 거래와 함께 상방 돌파 분출. 단기 모멘텀 추종이 유리합니다."
            else:
                return "🔷 하방 변동성 폭발 (적극 관망)", "대량 거래를 동반한 강한 하방 이탈 발생. 추가 낙폭 위험이 크므로 절대 매수를 금지합니다."
        else:
            return "⚖️ 단기 관망", "뚜렷한 추세나 타점이 부재한 변곡점 구간입니다. 명확한 방향성 확인 후 대응하십시오."
    else:
        # 중장기 대세 모드
        if is_falling_knife:
            return "🔷 장기 투매 진행 중 (절대 매수금지)", "주봉 기준 대량 거래를 동반한 장대음봉 폭락이 포착되었습니다. 추가 연쇄 하락 위험이 극도로 큽니다."
        elif regime == "변동성 폭발":
            if close > prev_candle_close:
                return "🔴 장기 대시세 분출 (비중 확대)", "장기 박스권을 상방으로 막대한 거래량과 함께 뚫어내는 대형 우상향 시작 타점입니다."
            else:
                return "🔷 하방 변동성 폭발 (적극 관망)", "폭발적인 매도 자금 이탈과 함께 중장기 주요 구조선들을 연쇄적으로 이탈하는 초고위험 구간입니다."
        elif regime == "상승 조정" and (box_pos > 50 or obv < simple_prev_obv):
            return "⚖️ 장기 눌림목 대기", "장기 상승장 내 조정 구간이나, 하락세 진정 및 지지선 확인 전까지 보수적 관망을 권장합니다."
        elif regime in ["강세 추세", "상승 조정"]:
            return "🔴 비중 확대 (장기)", "대세 상승장에 진입했습니다. 장기적 시각에서 비중 확대 및 홀딩 전략이 유효합니다."
        elif regime == "약세 추세" and rsi < 30:
            return "🟠 저점 분할 매집", "역사적 저평가 구간 진입. 펀더멘털 확인 후 긴 호흡으로 1차 분할 매집을 고려할 수 있습니다."
        elif regime == "약세 추세":
            return "🔷 비중 축소 (장기)", "대세 하락장이 지속 중입니다. 포트폴리오 방어를 위해 주식 비중 축소를 권장합니다."
        else:
            return "⚖️ 장기 관망", "장기 추세의 변곡점이거나 방향성이 불분명한 구간입니다. 확실한 추세 형성 시까지 관망하십시오."


# ==========================================
# 5. 퀀트 점수와 전략 간의 상충 조정
# ==========================================
def reconcile_with_quant_score(pos, strategy, q_score, is_short_term, is_falling_knife, rules):
    """퀀트 스코어와 기술적 전략 간의 모순을 감지하여 보수적으로 튜닝"""
    q_val = rules.get('quant_validation', {})
    buy_list = set(q_val.get('buy_positions', []))
    sell_list = set(q_val.get('sell_positions', []))
    bottom_fishing_list = set(q_val.get('bottom_fishing_positions', []))
    low_th = q_val.get('low_threshold', 30)
    high_th = q_val.get('high_threshold', 70)

    if pos in buy_list and q_score < low_th:
        if pos in bottom_fishing_list:
            strategy += f" (참고: 퀀트 스코어는 {q_score}점으로 낮으나, 낙폭 과대에 따른 역발상 타점이므로 매수 관점을 유지합니다.)"
        else:
            pos = "⚖️ 단기 관망" if is_short_term else "⚖️ 장기 관망"
            strategy = f"매수/보유 신호가 포착되었으나 퀀트 스코어({q_score}점)가 다소 낮아 신뢰도가 떨어집니다. 관망을 권장합니다."
    elif pos in sell_list and q_score > high_th and not is_falling_knife:
        pos = "⚖️ 단기 관망" if is_short_term else "⚖️ 장기 관망"
        strategy = f"매도/비중축소 신호가 포착되었으나 퀀트 스코어({q_score}점)가 양호하여 상충이 발생합니다. 방향성 확인 후 대응하십시오."

    return pos, strategy


# ==========================================
# 6. 메인 진입점: 상세 의견 및 심층 진단 리포트 생성
# ==========================================
def generate_detailed_opinions(df, sup, res, currency, decimals, is_short_term, time_unit, q_score, patterns=None, weekly_bullish=None):
    """
    기존 app.py와 100% 호환되는 진입점 함수.
    YAML 룰셋과 규칙 엔진을 통해 시장 국면, 포지션, 전략 및 마크다운 리포트를 생성.
    """
    rules = load_rules()
    md_currency = currency.replace('$', r'\$')

    latest, prev = df.iloc[-1], df.iloc[-2]
    close = float(latest['Close'])
    rsi = float(latest['RSI']) if 'RSI' in latest and not pd.isna(latest['RSI']) else 50.0
    obv = float(latest['OBV']) if 'OBV' in latest and not pd.isna(latest['OBV']) else 0.0
    vol_ratio = float(latest['Vol_Ratio']) if 'Vol_Ratio' in latest and not pd.isna(latest['Vol_Ratio']) else 100.0
    atr = float(latest['ATR']) if 'ATR' in latest and not pd.isna(latest['ATR']) else 0.0
    ma20 = float(latest['MA20']) if 'MA20' in latest and not pd.isna(latest['MA20']) else close
    ma60 = float(latest['MA60']) if 'MA60' in latest and not pd.isna(latest['MA60']) else close
    adx = float(latest['ADX']) if 'ADX' in latest and not pd.isna(latest['ADX']) else np.nan

    prev_close = float(prev['Close'])
    prev_candle_close = prev_close
    prev_ma20 = float(prev['MA20']) if 'MA20' in prev and not pd.isna(prev['MA20']) else ma20

    simple_lookback = min(5, len(df) - 1) if len(df) > 1 else 1
    long_lookback = min(13, len(df) - 1) if len(df) > 1 else 1
    obv_lookback = simple_lookback if is_short_term else long_lookback
    simple_prev_obv = float(df['OBV'].iloc[-obv_lookback]) if 'OBV' in df.columns else 0.0

    # 1. 다이버전스 감지
    bullish_div = detect_bullish_divergence(df)

    # 2. 시장 국면 판정
    regime = evaluate_market_regime(df, latest)

    # 3. 박스권 위치 및 투매(Falling Knife) 여부
    box_pos = ((close - sup) / (res - sup) * 100) if (res > sup and sup > 0) else 100
    drop_pct = ((prev_close - close) / prev_close * 100) if prev_close > 0 else 0
    is_falling_knife = (drop_pct >= 7.0 and vol_ratio >= 120) or (drop_pct >= 10.0)

    macd_diff = float(latest['MACD'] - latest['Signal']) if ('MACD' in latest and 'Signal' in latest and not pd.isna(latest['MACD']) and not pd.isna(latest['Signal'])) else 0.0
    vol_pct = (atr / close) * 100 if close > 0 else 0

    has_valid_adx = not pd.isna(adx)
    adx_disp = f"**{adx:.1f}**" if has_valid_adx else "**산출 중**"

    # 4. 지표별 해설 생성 (YAML 룰셋 참조)
    comments = {}
    comments['ADX'] = f"현재 ADX 추세강도 지수는 {adx_disp}이며, 알고리즘은 현재 시장을 **[{regime}]** 국면으로 확정했습니다."

    rsi_disp = f"RSI({rsi:.1f})" if not pd.isna(latest.get('RSI', np.nan)) else "RSI(산출 중)"
    regimes_rule = rules.get('regimes', {}).get(regime, {})

    if regime == "에너지 응축 (스퀴즈)":
        comments['RSI'] = f"{rsi_disp}: {regimes_rule.get('rsi_comment') or '볼린저 밴드 수축 국면이므로 RSI의 움직임이 매우 둔화되어 있습니다. 방향성 탐색 중입니다.'}"
        comments['MACD'] = f"MACD({macd_diff:,.{decimals}f}): {regimes_rule.get('macd_comment') or '이동평균선이 밀집하며 MACD도 0선에 완전히 수렴했습니다. 폭풍 전야의 고요한 상태입니다.'}"
    elif regime == "횡보 박스":
        rsi_cfg = regimes_rule.get('rsi_comment', {})
        if isinstance(rsi_cfg, dict) and rsi_cfg:
            detail = (
                rsi_cfg.get('oversold') if (not pd.isna(rsi) and rsi <= 40) else (
                    rsi_cfg.get('overbought') if (not pd.isna(rsi) and rsi >= 60) else rsi_cfg.get('neutral')
                )
            ) or "박스권 내에서 방향성을 탐색 중입니다."
        else:
            detail = "박스권 내에서 방향성을 탐색 중입니다."
        comments['RSI'] = f"{rsi_disp}: {detail}"
        comments['MACD'] = f"MACD({macd_diff:,.{decimals}f}): {regimes_rule.get('macd_comment') or '뚜렷한 추세가 부재한 박스권이므로 MACD 크로스 신호의 신뢰도는 다소 떨어집니다.'}"
    elif regime == "강세 추세":
        rsi_cfg = regimes_rule.get('rsi_comment', {})
        if isinstance(rsi_cfg, dict) and rsi_cfg:
            detail = (
                rsi_cfg.get('overbought') if (not pd.isna(rsi) and rsi >= 70) else (
                    rsi_cfg.get('pullback') if (not pd.isna(rsi) and rsi <= 50) else rsi_cfg.get('neutral')
                )
            ) or "안정적인 상승 탄력을 유지하고 있습니다."
        else:
            detail = "안정적인 상승 탄력을 유지하고 있습니다."
        comments['RSI'] = f"{rsi_disp}: {detail}"
        comments['MACD'] = f"MACD({macd_diff:,.{decimals}f}): {regimes_rule.get('macd_comment') or '상승 모멘텀이 강하게 유지되며 이평선 정배열 확장을 지지하고 있습니다.'}"
    elif regime == "상승 조정":
        comments['RSI'] = f"{rsi_disp}: {regimes_rule.get('rsi_comment') or '상승 추세 속에서 조정을 받으며 지표가 식어가고 있습니다. 40~50 부근에서 지지받는지 확인이 필요합니다.'}"
        comments['MACD'] = f"MACD({macd_diff:,.{decimals}f}): {regimes_rule.get('macd_comment') or '단기적으로 데드크로스가 발생하거나 모멘텀이 둔화되었으나, 장기 상승 추세 베이스는 훼손되지 않았습니다.'}"
    elif regime == "약세 추세":
        rsi_cfg = regimes_rule.get('rsi_comment', {})
        if isinstance(rsi_cfg, dict) and rsi_cfg:
            detail = (
                rsi_cfg.get('rebound') if (not pd.isna(rsi) and rsi >= 55) else (
                    rsi_cfg.get('oversold') if (not pd.isna(rsi) and rsi <= 30) else rsi_cfg.get('neutral')
                )
            ) or "지속적인 하락 압력을 받고 있습니다."
        else:
            detail = "지속적인 하락 압력을 받고 있습니다."
        comments['RSI'] = f"{rsi_disp}: {detail}"
        comments['MACD'] = f"MACD({macd_diff:,.{decimals}f}): {regimes_rule.get('macd_comment') or '하락 모멘텀이 강하며, 추세 반전을 암시하는 뚜렷한 신호가 아직 없습니다.'}"
    elif regime == "변동성 폭발":
        comments['RSI'] = f"{rsi_disp}: {regimes_rule.get('rsi_comment') or '변동성 폭발로 인해 투심이 한쪽으로 극단적으로 쏠리는 오버슈팅 및 투매 국면입니다.'}"
        comments['MACD'] = f"MACD({macd_diff:,.{decimals}f}): {regimes_rule.get('macd_comment') or '단기 모멘텀이 평소의 범위를 벗어나 급격하게 방향성을 분출하고 있습니다.'}"
    else:
        comments['RSI'] = f"{rsi_disp}: {regimes_rule.get('rsi_comment') or '데이터 축적 중으로 지표 신뢰도를 검증 중입니다.'}"
        comments['MACD'] = f"MACD({macd_diff:,.{decimals}f}): {regimes_rule.get('macd_comment') or '추세 형성 초기 단계입니다.'}"

    comments['VOL'] = f"상대 거래량이 평균 대비 **{vol_ratio:.0f}%** 수준입니다. " + ("대량 거래가 터지며 시장의 강한 이목이 집중되었습니다." if vol_ratio > 150 else "평이한 수준의 거래가 이뤄지고 있습니다.")
    comments['OBV'] = f"최근 {obv_lookback}{time_unit}간 누적 수급(OBV)이 **{'상승(자금 유입)' if obv > simple_prev_obv else '하락(자금 이탈)'}** 중입니다."
    comments['ATR'] = f"예상되는 실질 변동폭(ATR)은 주당 평균 **{vol_pct:.1f}% ({atr:,.{decimals}f}{md_currency})** 수준입니다."

    # 5. 전략 컨텍스트 구성 및 평가
    ctx = {
        'is_short_term': is_short_term,
        'is_falling_knife': is_falling_knife,
        'res': res,
        'close': close,
        'prev_close': prev_close,
        'prev_candle_close': prev_candle_close,
        'regime': regime,
        'bullish_div': bullish_div,
        'ma20': ma20,
        'obv': obv,
        'simple_prev_obv': simple_prev_obv,
        'box_pos': box_pos,
        'vol_ratio': vol_ratio,
        'rsi': rsi,
        'adx': adx
    }
    pos, strategy = evaluate_strategy(ctx)

    # 6. 퀀트 스코어와 상충 조정
    pos, strategy = reconcile_with_quant_score(pos, strategy, q_score, is_short_term, is_falling_knife, rules)

    # 7. AI 리포트 마크다운 조립
    mode_str = "단기 스윙" if is_short_term else "장기 가치투자"
    ai_op = f"🤖 **StockMap AI {mode_str} 심층 진단 리포트**\n\n"
    ai_op += f"🔍 **[시장 국면 분류]**\n\n• 현재 해당 종목은 **[{regime}]** 국면에 위치해 있습니다.\n\n"

    # MTF (다중 시간대) 분석
    if is_short_term and weekly_bullish is not None:
        mtf_rules = rules.get('mtf_narratives', {})
        ai_op += "⏱️ **[MTF 다중 시간대 분석]**\n\n"
        if regime in ["강세 추세", "상승 조정"]:
            b_cfg = mtf_rules.get('bullish_trends', {})
            ai_op += (b_cfg.get('bull') if weekly_bullish else b_cfg.get('bear')) + "\n\n"
        elif regime == "약세 추세":
            b_cfg = mtf_rules.get('bear_trends', {})
            ai_op += (b_cfg.get('bull') if weekly_bullish else b_cfg.get('bear')) + "\n\n"
        elif regime == "횡보 박스":
            b_cfg = mtf_rules.get('range_trends', {})
            ai_op += (b_cfg.get('bull') if weekly_bullish else b_cfg.get('bear')) + "\n\n"
        else:
            ai_op += mtf_rules.get('default', "• **장기 흐름:** 장기 흐름에 동조화되어 에너지가 응축/분출되는 변곡점 구간입니다.") + "\n\n"

    ai_op += "💡 **[국면 맞춤형 통합 해석]**\n\n"
    if is_falling_knife:
        ai_op += "🚨 **[초고위험 투매 경보]** 현재 주가가 비정상적인 속도로 극심하게 급락 중인 '패닉셀' 구간입니다. 어떠한 기술적 반등 신호도 무시하고 철저히 관망할 것을 강력히 권고합니다.\n\n"
    elif res == 0:
        ai_op += "✨ **[신고가 랠리 분석]** 과거의 모든 악성 매물대를 소화하고 완벽한 신고가(상방 열림) 영역에 진입했습니다. 강력한 추세가 이어질 확률이 높습니다.\n\n"
    elif regime == "에너지 응축 (스퀴즈)":
        ai_op += "• 변동성이 극도로 응축된 상태입니다. 곧 강한 방향성 분출이 예상됩니다.\n\n"
    elif regime == "횡보 박스":
        if box_pos <= 35:
            ai_op += f"• 하단 지지선({sup:,.{decimals}f}{md_currency}) 부근으로 단기 매수 매력도가 높습니다.\n\n"
        elif box_pos >= 65:
            ai_op += f"• 상단 저항선({res:,.{decimals}f}{md_currency}) 부근으로 리스크 관리가 필요한 구간입니다.\n\n"
    elif regime == "강세 추세":
        ai_op += "• 매수세가 시장을 주도하는 강세장입니다. 추세 이탈 전까지 보유가 유리합니다.\n\n"
    elif regime == "상승 조정":
        ai_op += "• 상승 흐름 속 건전한 단기 조정(매물 소화)이 진행 중입니다.\n\n"
    elif regime == "약세 추세":
        ai_op += "• 하락 압력이 지배적이므로 철저한 현금 비중 관리와 보수적 접근이 필수입니다.\n\n"
    elif regime == "변동성 폭발":
        ai_op += f"• {'상방 대량 거래 폭발 확인. 새로운 대시세의 시작일 수 있으나 추격 매수는 신중하게 접근하십시오.' if close > prev_candle_close else '하방 대량 매도세 폭발 확인. 추가 연쇄 하락 위험이 있으므로 절대 역추세 매수를 자제하십시오.'}\n\n"

    ai_op += "📊 **[수급 및 주요 레벨]**\n\n"
    ai_op += f"• **세력 수급:** 누적 수급(OBV)이 꾸준히 {'유입되며 긍정적' if obv > simple_prev_obv else '이탈하며 부정적'}인 정황이 관찰됩니다.\n\n"

    # 불트랩 / 베어트랩 판정
    traps = rules.get('traps', {})
    latest_open, latest_high, latest_low = float(latest['Open']), float(latest['High']), float(latest['Low'])
    body = abs(latest_open - close)
    if close > prev_candle_close and close > ma20 and prev_candle_close <= prev_ma20 and not is_falling_knife:
        if vol_ratio < 80 or (latest_high - max(latest_open, close)) > body * 1.5:
            ai_op += traps.get('bull_trap', "🚨 **[가짜 상승(Bull Trap) 주의]** 저항을 돌파했으나 거래량이 부진하거나 윗꼬리가 깁니다. 섣부른 추격 매수를 자제하십시오.\n\n")
    elif close < prev_candle_close and close < ma20 and prev_candle_close >= prev_ma20:
        if vol_ratio < 70 or (min(latest_open, close) - latest_low) > body * 1.5:
            ai_op += traps.get('bear_trap', "🚨 **[가짜 하락(Bear Trap) 주의]** 지지를 이탈했으나 하락 물량 방어 흔적(아랫꼬리)이 보입니다. 일시적 충격일 수 있습니다.\n\n")

    # 역사적 백테스트 통계 자동 삽입
    _, matched_stats = match_current_setup({
        'regime': regime,
        'bullish_div': bullish_div,
        'is_falling_knife': is_falling_knife,
        'patterns': patterns or []
    })
    if matched_stats:
        ai_op += format_stats_for_report(matched_stats)

    ai_op += "📅 **[단기 실전 대응 시나리오 가이드]**\n\n"
    if res == 0:
        ai_op += "• **상방 추세 시나리오:** 저항 없는 신고가 상태입니다. 추세 꺾임 시까지 수익 극대화 관점.\n\n"
    else:
        ai_op += f"• **상방 돌파 시나리오:** 1차 저항선인 **{res:,.{decimals}f}{md_currency}** 강하게 돌파 시 새로운 상승 추세로 판단, 매수 관점 접근.\n\n"
    ai_op += f"• **하방 방어 시나리오:** 기계적 손절 라인은 **{max(0, close - atr):,.{decimals}f}{md_currency}** 부근, 핵심 지지선은 **{sup:,.{decimals}f}{md_currency}** 입니다. 이탈 시 즉각적 리스크 관리 우선.\n\n"

    if bullish_div and not is_falling_knife:
        if regime == "약세 추세":
            ai_op += "🔥 **[상승 다이버전스 포착]** 하락 추세 속에서 주가는 신저가를 기록했으나, 보조지표 저점이 상승하는 강력한 역발상 반전 시그널이 감지되었습니다!\n\n"
        else:
            ai_op += "🔥 **[상승 다이버전스 포착]** 보조지표의 저점이 상승하는 긍정적 반전 시그널이 확인되었습니다!\n\n"

    comments['AI'] = f"{ai_op}🎯 **최종 투자 전략 요약:** {strategy} (AI 권장 포지션: **{pos}**)"

    # 텍스트 파싱 의존을 원천 차단하기 위한 원본 불리언 및 레짐 데이터 제공
    comments['regime_raw'] = regime
    comments['bullish_div_raw'] = bullish_div
    comments['is_falling_knife_raw'] = is_falling_knife

    return pos, strategy, comments
