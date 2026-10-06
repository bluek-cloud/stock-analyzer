import os
import json
import pandas as pd
import numpy as np
from setup_signals import build_setup_signals, current_setups, normalize_price_data, SETUP_NAMES, SIGNAL_VERSION

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


def verified_stats(stats):
    """Legacy constants are not evidence; only documented, versioned daily results qualify."""
    if not isinstance(stats, dict):
        return False
    provenance = stats.get('provenance', {})
    return (stats.get('verification_status') == 'verified'
            and stats.get('signal_version') == SIGNAL_VERSION
            and stats.get('timeframe') == 'daily'
            and isinstance(provenance, dict)
            and all(provenance.get(k) for k in ('source', 'period', 'universe', 'trade_log', 'generator')))


def match_current_setup(market_ctx, patterns=None, df=None):
    """Match shared OHLCV signals, never infer a setup from a regime label."""
    if market_ctx.get('is_falling_knife', False):
        return None, None
    setups = current_setups(df) if df is not None else market_ctx.get('matched_setups', [])
    key = next((k for k in SETUP_NAMES if k in setups), None)
    if key is None:
        return None, None
    stats = load_backtest_stats().get(key)
    if not market_ctx.get('is_short_term', True) or not verified_stats(stats):
        stats = None
    return key, stats


def run_stock_backtest(df, setup_type="AUTO", hold_days=20):
    """
    현재 종목의 과거 전체 데이터(DataFrame)에서 특정 셋업의 발생 시점과
    N거래일 경과 후의 실제 수익률, 승률, 손익비를 즉석 시뮬레이션
    """
    if len(df) < 60:
        return {
            'error': '시뮬레이션을 위한 데이터가 부족합니다 (최소 60개 봉 필요).'
        }

    if not isinstance(hold_days, int) or hold_days <= 0:
        return {'error': '보유 봉 수는 양의 정수여야 합니다.'}
    if setup_type != "AUTO" and setup_type not in SETUP_NAMES:
        return {'error': '지원하지 않는 셋업입니다.'}
    try:
        df = normalize_price_data(df)
        setup_signals = build_setup_signals(df)
    except ValueError as exc:
        return {'error': str(exc)}
    close, high, low = (pd.to_numeric(df[k]) for k in ('Close', 'High', 'Low'))
    signals = []
    # Signal-close event study. Overlapping events are retained, not a portfolio simulation.
    for i in range(20, len(df) - hold_days):
        row = setup_signals.iloc[i]
        matched = bool(row.any()) if setup_type == "AUTO" else bool(row[setup_type])
        cur_close = close.iloc[i]
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
                'setups': [k for k in SETUP_NAMES if row[k]],
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
    if not verified_stats(matched_stats):
        return ""

    text = f"📊 **[역사적 백테스트 통계 & 기대 확률]**\n\n"
    text += f"• **매칭 셋업:** {matched_stats['name']} ({matched_stats['benchmark_note']})\n\n"
    text += f"• **20거래일 보유 승률:** **{matched_stats['win_rate_20d']}%** (5거래일 단기: {matched_stats['win_rate_5d']}%)\n\n"
    text += f"• **평균 기대 수익률:** **+{matched_stats['avg_return_20d']}%** (평균 최대 반등폭: +{matched_stats['avg_mfe']}%)\n\n"
    text += f"• **손익비 (Profit Factor):** **{matched_stats['profit_factor']} : 1** (권장 손절: -{matched_stats['recommended_sl_pct']}%, 목표가: +{matched_stats['recommended_tp_pct']}%)\n\n"
    return text


def format_stats_for_llm(matched_stats):
    """Gemini LLM 프롬프트에 주입할 통계 텍스트"""
    if not verified_stats(matched_stats):
        return "검증된 역사적 백테스트 통계 없음. 승률·표본수·통계 기반 목표가를 추정하거나 인용하지 마세요."

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
