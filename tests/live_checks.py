"""Opt-in public-data integration check: python tests/live_checks.py --scope 600."""
import argparse
import ast
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import requests
from accumulation_scanner import scan_smart_money_stocks, get_investor_net_buys
from setup_signals import normalize_price_data, current_setups
from rule_engine import generate_detailed_opinions
from backtest_engine import run_stock_backtest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', type=int, default=600)
    args = parser.parse_args()
    # Bound provider calls only in this diagnostic process.
    original = requests.sessions.Session.request
    def bounded(self, method, url, **kwargs):
        kwargs.setdefault('timeout', 15)
        return original(self, method, url, **kwargs)
    requests.sessions.Session.request = bounded
    reader = fdr.DataReader
    price_cache = {}
    def cached_reader(symbol, start=None, end=None):
        key = (symbol, start, end)
        if key not in price_cache:
            price_cache[key] = reader(symbol, start=start, end=end)
        return price_cache[key].copy()
    fdr.DataReader = cached_reader
    print('Fetching Doosan prices...', flush=True)
    raw = fdr.DataReader('034020', start=(datetime.now()-timedelta(days=1825)).strftime('%Y-%m-%d'))
    clean = normalize_price_data(raw)
    # Execute the app's actual pure functions without launching its top-level UI.
    tree = ast.parse((ROOT/'app.py').read_text(encoding='utf-8'))
    wanted = {'calculate_indicators', 'calculate_quant_score', 'detect_patterns_and_levels', 'parse_query'}
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    universe = pd.read_csv(ROOT/'krx_cache.csv', dtype={'Code': str})
    ns = {'pd': pd, 'np': np, '_get_krx_data_safe': lambda: universe}
    exec(compile(ast.Module(body=functions, type_ignores=[]), 'app.py', 'exec'), ns)
    assert ns['parse_query']('두산에너빌리티')[1] == '034020'
    analysis = {}
    for short in (True, False):
        chart = clean if short else clean.resample('W').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
        chart = ns['calculate_indicators'](chart)
        patterns, support, resistance = ns['detect_patterns_and_levels'](chart)
        score = ns['calculate_quant_score'](chart, short)
        position, strategy, comments = generate_detailed_opinions(chart, support, resistance, '원', 0, short, '일' if short else '주', score, patterns)
        backtest = run_stock_backtest(chart)
        assert 'error' not in backtest, backtest
        analysis['daily' if short else 'weekly'] = {'rows': len(chart), 'report_created': bool(comments['AI']), 'backtest_trades': backtest['total_trades']}
    print('Doosan daily/weekly analysis and backtest passed.', flush=True)
    print(f'Starting live scanner: {args.scope} candidates...', flush=True)
    class Status:
        completed = 0
        def progress(self, value):
            bucket = int(value*10)
            if bucket > self.completed:
                self.completed = bucket
                print(f'Scan {bucket*10}%', flush=True)
    result = scan_smart_money_stocks(universe, scan_scope=args.scope, min_accum_candles=1, progress_bar=Status())
    print('Checking strict mode on the same downloaded prices...', flush=True)
    strict = scan_smart_money_stocks(universe, scan_scope=args.scope, min_accum_candles=2)
    assert set(strict['code'] if not strict.empty else []).issubset(set(result['code'] if not result.empty else []))
    report = {'checked_at': datetime.now().isoformat(),
              'universe_source': 'local krx_cache.csv snapshot',
              'doosan': {'rows': len(clean), 'latest_date': str(clean.index[-1].date()),
                         'rounding_adjusted_bars': clean.attrs['rounding_adjusted_bars'], 'analysis': analysis},
              'scanner': result.attrs['scan_summary'], 'matches': len(result),
              'strict_scanner': strict.attrs['scan_summary'], 'strict_matches': len(strict),
              'investor_data_available': int(result['investor_data_available'].sum()) if not result.empty else 0,
              'match_codes': result['code'].tolist() if not result.empty else []}
    (ROOT/'live-check-results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
