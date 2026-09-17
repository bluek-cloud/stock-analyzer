import os
import json
import pandas as pd
import numpy as np

STATS_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'backtest_stats.json')
_CACHED_STATS = None

def load_backtest_stats():
    """사전 백테스트 통계 데이터베이스 로드"""
    global _CACHED_STATS
    if _CACHED_STATS is not None:
        return _CACHED_STATS

    if os.path.exists(STATS_FILE_PATH):
        try:
            with open(STATS_FILE_PATH, 'r', encoding='utf-8') as f:
                _CACHED_STATS = json.load(f)
                return _CACHED_STATS
        except Exception:
            pass

    _CACHED_STATS = {}
    return _CACHED_STATS


def match_current_setup(market_ctx, patterns=None):
    """
    현재 종목의 기술적 지표 및 패턴을 기반으로 가장 적합한 6대 셋업 통계를 매칭
    """
    stats = load_backtest_stats()
    if not stats:
        return None, None

    patterns = patterns or market_ctx.get('patterns', [])
    regime = market_ctx.get('regime', '')
    bullish_div = market_ctx.get('bullish_div', False)

    # 1. 상승 다이버전스 셋업
    if bullish_div and "BULLISH_DIVERGENCE" in stats:
        return "BULLISH_DIVERGENCE", stats["BULLISH_DIVERGENCE"]

    # 2. 볼린저 스퀴즈 분출 셋업
    if ("스퀴즈" in regime or "폭발" in regime) and "BOLLINGER_SQUEEZE_BREAKOUT" in stats:
        return "BOLLINGER_SQUEEZE_BREAKOUT", stats["BOLLINGER_SQUEEZE_BREAKOUT"]

    # 3. 캔들 패턴 기반 매칭
    for p in patterns:
        if "상승 장악형" in p and "BULLISH_ENGULFING" in stats:
            return "BULLISH_ENGULFING", stats["BULLISH_ENGULFING"]
        if "망치형" in p and "HAMMER_BOTTOM" in stats:
            return "HAMMER_BOTTOM", stats["HAMMER_BOTTOM"]

    # 4. 강세 추세 / 200일선 지지 셋업
    if "강세" in regime and "MA200_PULLBACK" in stats:
        return "MA200_PULLBACK", stats["MA200_PULLBACK"]

    # 5. 박스권 돌파 / 횡보 셋업
    if "횡보" in regime and "BOX_BREAKOUT" in stats:
        return "BOX_BREAKOUT", stats["BOX_BREAKOUT"]

    # 기본 매칭 (200일선 눌림목)
    return "MA200_PULLBACK", stats.get("MA200_PULLBACK")


def run_stock_backtest(df, setup_type="AUTO", hold_days=20):
    """
    현재 종목의 과거 전체 데이터(DataFrame)에서 특정 셋업의 발생 시점과
    N거래일 경과 후의 실제 수익률, 승률, 손익비를 즉석 시뮬레이션
    """
    if len(df) < 60:
        return {
            'error': '시뮬레이션을 위한 데이터가 부족합니다 (최소 60거래일 이상 필요).'
        }

    df = df.copy()
    close = df['Close']
    open_p = df['Open']
    high = df['High']
    low = df['Low']

    # 보조 지표 계산 (없는 경우 보충)
    if 'MA20' not in df.columns:
        df['MA20'] = close.rolling(20).mean()
    if 'MA60' not in df.columns:
        df['MA60'] = close.rolling(60).mean()
    if 'MA200' not in df.columns:
        df['MA200'] = close.rolling(200).mean()
    if 'BB_Upper' not in df.columns:
        std = close.rolling(20).std()
        df['BB_Upper'] = df['MA20'] + (std * 2)
    if 'RSI' not in df.columns:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        rs = gain.ewm(alpha=1/14, adjust=False).mean() / (loss.ewm(alpha=1/14, adjust=False).mean() + 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))

    signals = []
    # 과거 전체 봉 순회 (hold_days 전까지만 진입 가능)
    for i in range(20, len(df) - hold_days):
        matched = False
        cur_close = close.iloc[i]
        cur_open = open_p.iloc[i]
        cur_high = high.iloc[i]
        cur_low = low.iloc[i]
        prev_close = close.iloc[i-1]
        prev_open = open_p.iloc[i-1]

        # 1. 200일선 눌림목
        if setup_type in ["MA200_PULLBACK", "AUTO"]:
            ma200 = df['MA200'].iloc[i]
            if not pd.isna(ma200) and i >= 210:
                ma200_prev10 = df['MA200'].iloc[i-10]
                if ma200 >= ma200_prev10 and (0.97 <= cur_low / ma200 <= 1.04) and cur_close > cur_open:
                    matched = True

        # 2. 상승 장악형
        if not matched and setup_type in ["BULLISH_ENGULFING", "AUTO"]:
            if prev_close < prev_open and cur_close > cur_open:
                if cur_open <= prev_open and cur_close > prev_open:
                    matched = True

        # 3. 망치형 캔들
        if not matched and setup_type in ["HAMMER_BOTTOM", "AUTO"]:
            c_range = cur_high - cur_low
            body = abs(cur_close - cur_open)
            lower_shadow = min(cur_open, cur_close) - cur_low
            if c_range > 0 and body <= c_range * 0.35 and lower_shadow >= c_range * 0.5:
                if cur_close <= df['MA20'].iloc[i]:
                    matched = True

        # 4. RSI 과매도 반등
        if not matched and setup_type in ["BULLISH_DIVERGENCE", "AUTO"]:
            rsi_val = df['RSI'].iloc[i]
            if not pd.isna(rsi_val) and rsi_val < 35 and cur_close > prev_close:
                matched = True

        # 5. 볼린저 밴드 상방 돌파
        if not matched and setup_type in ["BOLLINGER_SQUEEZE_BREAKOUT", "AUTO"]:
            bb_up = df['BB_Upper'].iloc[i]
            if not pd.isna(bb_up) and cur_close > bb_up and cur_close > cur_open:
                matched = True

        # 6. 박스권 상단 돌파
        if not matched and setup_type in ["BOX_BREAKOUT", "AUTO"]:
            if i >= 21:
                prev_20_high = high.iloc[i-20:i].max()
                if cur_close > prev_20_high and cur_close > cur_open:
                    matched = True

        if matched:
            # 진입 후 hold_days 경과 후 종가 확인
            exit_close = close.iloc[i + hold_days]
            ret_pct = ((exit_close - cur_close) / cur_close) * 100

            # 보유 기간 중 최대 상승폭(MFE), 최대 하락폭(MAE)
            future_highs = high.iloc[i+1 : i+hold_days+1]
            future_lows = low.iloc[i+1 : i+hold_days+1]
            mfe_pct = ((future_highs.max() - cur_close) / cur_close) * 100
            mae_pct = ((cur_close - future_lows.min()) / cur_close) * 100

            entry_date = df.index[i].strftime('%Y-%m-%d') if hasattr(df.index[i], 'strftime') else str(df.index[i])
            exit_date = df.index[i + hold_days].strftime('%Y-%m-%d') if hasattr(df.index[i + hold_days], 'strftime') else str(df.index[i + hold_days])

            entry_p = round(float(cur_close), 2 if cur_close < 1000 else 0)
            exit_p = round(float(exit_close), 2 if exit_close < 1000 else 0)
            if entry_p.is_integer() if hasattr(entry_p, 'is_integer') else False:
                entry_p = int(entry_p)
            if exit_p.is_integer() if hasattr(exit_p, 'is_integer') else False:
                exit_p = int(exit_p)

            signals.append({
                'entry_date': entry_date,
                'entry_price': entry_p,
                'exit_date': exit_date,
                'exit_price': exit_p,
                'return_pct': round(ret_pct, 2),
                'mfe_pct': round(mfe_pct, 2),
                'mae_pct': round(mae_pct, 2),
                'is_win': ret_pct > 0
            })

    if not signals:
        return {
            'total_trades': 0,
            'message': '과거 차트에서 일치하는 타점이 포착되지 않았습니다.'
        }

    trades_df = pd.DataFrame(signals)
    win_trades = trades_df[trades_df['is_win']]
    loss_trades = trades_df[~trades_df['is_win']]
    win_count = len(win_trades)
    total_count = len(trades_df)
    win_rate = (win_count / total_count) * 100

    sum_win = win_trades['return_pct'].sum()
    sum_loss = abs(loss_trades['return_pct'].sum())
    profit_factor = (sum_win / sum_loss) if sum_loss > 0 else (99.9 if sum_win > 0 else 1.0)

    return {
        'total_trades': total_count,
        'win_trades': win_count,
        'loss_trades': total_count - win_count,
        'win_rate': round(win_rate, 1),
        'avg_return': round(trades_df['return_pct'].mean(), 2),
        'profit_factor': round(profit_factor, 2),
        'avg_mfe': round(trades_df['mfe_pct'].mean(), 1),
        'avg_mae': round(trades_df['mae_pct'].mean(), 1),
        'max_profit': round(trades_df['return_pct'].max(), 2),
        'max_loss': round(trades_df['return_pct'].min(), 2),
        'recent_trades': signals[-5:]  # 최근 5회 매매 내역
    }


def format_stats_for_report(matched_stats):
    """Level 1 리포트에 삽입할 통계 요약 마크다운 텍스트"""
    if not matched_stats:
        return ""

    text = f"📊 **[역사적 백테스트 통계 & 기대 확률]**\n\n"
    text += f"• **매칭 셋업:** {matched_stats['name']} ({matched_stats['benchmark_note']})\n\n"
    text += f"• **20거래일 보유 승률:** **{matched_stats['win_rate_20d']}%** (5거래일 단기: {matched_stats['win_rate_5d']}%)\n\n"
    text += f"• **평균 기대 수익률:** **+{matched_stats['avg_return_20d']}%** (평균 최대 반등폭: +{matched_stats['avg_mfe']}%)\n\n"
    text += f"• **손익비 (Profit Factor):** **{matched_stats['profit_factor']} : 1** (권장 손절: -{matched_stats['recommended_sl_pct']}%, 목표가: +{matched_stats['recommended_tp_pct']}%)\n\n"
    return text


def format_stats_for_llm(matched_stats):
    """Gemini LLM 프롬프트에 주입할 통계 텍스트"""
    if not matched_stats:
        return "참조할 역사적 백테스트 통계 없음."

    return (
        f"- 매칭된 셋업: {matched_stats['name']}\n"
        f"- 5개년 누적 표본수: {matched_stats['sample_count']}건\n"
        f"- 20거래일 보유 승률: {matched_stats['win_rate_20d']}%\n"
        f"- 평균 기대 수익률: +{matched_stats['avg_return_20d']}%\n"
        f"- 손익비 (Profit Factor): {matched_stats['profit_factor']}\n"
        f"- 평균 최대 낙폭(MAE): -{matched_stats['avg_mae']}%\n"
        f"- 권장 손절 기준: -{matched_stats['recommended_sl_pct']}%\n"
        f"- 권장 익절 목표: +{matched_stats['recommended_tp_pct']}%\n"
    )
