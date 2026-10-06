"""Shared, causal setup definitions for live analysis and historical scans."""
import numpy as np
import pandas as pd

SIGNAL_VERSION = "2"
SETUP_NAMES = {
    "BULLISH_DIVERGENCE": "RSI/OBV 상승 다이버전스",
    "BOLLINGER_SQUEEZE_BREAKOUT": "볼린저 스퀴즈 거래량 동반 상방 돌파",
    "BULLISH_ENGULFING": "상승 장악형",
    "HAMMER_BOTTOM": "바닥권 망치형",
    "MA200_PULLBACK": "200봉 이동평균 지지 및 5봉선 반등",
    "BOX_BREAKOUT": "20봉 박스권 거래량 동반 상방 돌파",
}


def normalize_price_data(df):
    """Normalize only bounded rounding artifacts; reject substantive bad bars.

    Integer adjusted prices may put a close one unit outside the reported
    high/low. Allow at most one unit (0.01 for fractional quotes), also capped
    at 0.01% of the price. Never change the open, close or volume.
    """
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not set(required).issubset(df.columns):
        raise ValueError("시가·고가·저가·종가·거래량 데이터가 필요합니다.")
    if not df.index.is_unique or not df.index.is_monotonic_increasing:
        raise ValueError("가격 데이터 날짜는 중복 없이 오름차순이어야 합니다.")
    out = df.copy()
    out[required] = out[required].apply(pd.to_numeric, errors="coerce")
    # Some feeds encode suspended sessions as zero O/H/L with a valid close.
    suspended = (out['Volume'] == 0) & (out['Close'] > 0) & out[['Open', 'High', 'Low']].eq(0).all(axis=1)
    for column in ('Open', 'High', 'Low'):
        out.loc[suspended, column] = out.loc[suspended, 'Close']
    prices = out[["Open", "High", "Low", "Close"]]
    if (not np.isfinite(out[required].to_numpy(dtype=float)).all()
            or (prices <= 0).any().any() or (out["Volume"] < 0).any()):
        raise ValueError("가격 또는 거래량에 결측값·비정상 값이 있습니다.")
    integral = prices.eq(np.floor(prices)).all(axis=1)
    tolerance = np.minimum(np.where(integral, 1., .01), prices.min(axis=1) * .0001)
    body_high = out[['Open', 'Close']].max(axis=1)
    body_low = out[['Open', 'Close']].min(axis=1)
    high_gap = body_high - out['High']
    low_gap = out['Low'] - body_low
    if ((high_gap > tolerance + 1e-9).any() or (low_gap > tolerance + 1e-9).any()
            or (out['High'] < out['Low']).any()):
        raise ValueError("고가·저가와 시가·종가의 차이가 허용 반올림 범위를 초과했습니다.")
    adjusted = (high_gap > 0) | (low_gap > 0)
    out['High'] = np.maximum(out['High'], body_high)
    out['Low'] = np.minimum(out['Low'], body_low)
    out.attrs['rounding_adjusted_bars'] = int(df.attrs.get('rounding_adjusted_bars', 0)) + int(adjusted.sum())
    out.attrs['suspended_bars'] = int(df.attrs.get('suspended_bars', 0)) + int(suspended.sum())
    return out


def prepare_setup_data(df):
    out = normalize_price_data(df)
    close = out["Close"]
    for period in (5, 20, 60, 200):
        out[f"MA{period}"] = close.rolling(period).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).fillna(0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).fillna(0).ewm(alpha=1/14, adjust=False).mean()
    out["RSI"] = (100 - 100 / (1 + gain / (loss + 1e-10))).mask((gain == 0) & (loss == 0), 50.)
    out["OBV"] = (np.sign(delta).fillna(0) * out["Volume"]).cumsum()
    std = close.rolling(20).std()
    out["BB_Upper"] = out["MA20"] + 2 * std
    out["BBW"] = 4 * std / out["MA20"] * 100
    out["Prior_Vol20"] = out["Volume"].shift(1).rolling(20).mean()
    return out


def bullish_divergence_at(df, i):
    """Compare lows in the last four bars with the preceding 26 bars."""
    if i < 29:
        return False
    past = df.iloc[i-29:i-3]
    recent = df.iloc[i-3:i+1]
    p = past.iloc[int(np.argmin(past["Low"].to_numpy()))]
    r = recent.iloc[int(np.argmin(recent["Low"].to_numpy()))]
    return bool(r["Low"] <= p["Low"] and
                (r["RSI"] > p["RSI"] + 2 or r["OBV"] > p["OBV"]))


def detect_bullish_divergence(df):
    if len(df) < 30:
        return False
    data = prepare_setup_data(df)
    return bullish_divergence_at(data, len(data)-1)


def build_setup_signals(df):
    d = prepare_setup_data(df)
    o, h, l, c, v = (d[k] for k in ("Open", "High", "Low", "Close", "Volume"))
    bullish = c > o
    body, span = (c-o).abs(), h-l
    volume_breakout = (d["Prior_Vol20"] > 0) & (v >= 1.5*d["Prior_Vol20"])
    # Require a squeeze in the previous five bars, never the future.
    squeeze = d["BBW"] <= d["BBW"].rolling(120, min_periods=120).min()*1.05
    recent_squeeze = squeeze.shift(1).rolling(5, min_periods=1).max().eq(1)
    prior_high = h.shift(1).rolling(20).max()
    prior_low = l.shift(1).rolling(20).min()
    box = (prior_high/prior_low - 1) <= 0.10
    near200 = l.div(d["MA200"]).between(.97, 1.04)
    cross5 = ((c.shift(1) <= d["MA5"].shift(1)) & (c > d["MA5"])) | ((l <= d["MA5"]) & (c > d["MA5"]))
    signals = pd.DataFrame(index=d.index)
    signals["BULLISH_DIVERGENCE"] = [bullish_divergence_at(d, i) for i in range(len(d))]
    signals["BOLLINGER_SQUEEZE_BREAKOUT"] = recent_squeeze & volume_breakout & bullish & (c > d["BB_Upper"]) & (c.shift(1) <= d["BB_Upper"].shift(1))
    signals["BULLISH_ENGULFING"] = (c.shift(1) < o.shift(1)) & bullish & (o <= c.shift(1)) & (c >= o.shift(1)) & (body >= body.shift(1))
    pullback = (c <= d["MA20"]) | (c < c.shift(5))
    signals["HAMMER_BOTTOM"] = (span > 0) & (body <= span*.35) & (pd.concat([o,c],axis=1).min(axis=1)-l >= span*.5) & (h-pd.concat([o,c],axis=1).max(axis=1) <= span*.15) & pullback
    signals["MA200_PULLBACK"] = (d["MA200"] >= d["MA200"].shift(10)) & (near200 | near200.shift(1).eq(True)) & bullish & cross5
    signals["BOX_BREAKOUT"] = box & volume_breakout & bullish & (c > prior_high)
    prior_vol5 = v.shift(1).rolling(5).mean()
    drop = (1-c/c.shift(1))*100
    falling_knife = (drop >= 10) | ((drop >= 7) & (v >= prior_vol5*1.2))
    signals.loc[falling_knife | (v <= 0), :] = False
    signals.iloc[:20, :] = False
    return signals.fillna(False).astype(bool)


def current_setups(df):
    if len(df) < 21:
        return []
    last = build_setup_signals(df).iloc[-1]
    return [key for key in SETUP_NAMES if last[key]]
