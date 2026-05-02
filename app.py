# =========================================================================
# 기존 `app.py`의 detect_patterns_and_levels 함수를 아래와 같이 교체해 주세요.
# =========================================================================

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

    # 지지선 판별 로직 (기존과 동일)
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

    # 🌟 핵심 수정: 저항선 판별 로직 (신고가 예외 처리)
    # 현재가보다 높은 과거 데이터 필터링
    above = closes[closes > latest['Close']] 
    
    if above.empty:
        # 현재가보다 높은 과거 고점이 없다면 = 완전한 돌파(신고가) 상태!
        resistance = 0  # 저항이 없음을 의미하는 플래그 값(0) 할당
    else:
        # 현재가보다 높은 저항선 후보가 있는 경우 (정상적인 저항선 탐색)
        if len(resistance_candidates) >= 2:
            res_clusters = cluster_levels(resistance_candidates)
            valid_res = [c for c in res_clusters if c['center'] > latest['Close']]
            if valid_res:
                resistance = valid_res[0]['center']  
            else:
                resistance = above.max()
        else:
            resistance = above.max()

    return patterns, support, resistance
