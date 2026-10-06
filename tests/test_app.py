"""UI integration checks with deterministic prices and no external API calls."""
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

import pandas as pd
from streamlit.testing.v1 import AppTest


class AppTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({"Open": 100., "High": 101., "Low": 99.,
                                  "Close": 100., "Volume": 1000.},
                                 index=pd.bdate_range("2021-01-01", periods=1300))
        self.data.loc[self.data.index[-1], ["High", "Close", "Volume"]] = [111., 110., 2000.]
        for target, kwargs in (
            ("FinanceDataReader.DataReader", {"side_effect": lambda *a, **k: self.data.copy()}),
            ("requests.head", {"return_value": Mock(status_code=503)}),
            ("requests.get", {"return_value": Mock(status_code=503)}),
        ):
            p = patch(target, **kwargs)
            p.start()
            self.addCleanup(p.stop)

    def test_daily_weekly_backtest_and_rag_context(self):
        at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()
        self.assertFalse(at.exception)
        at.session_state["target_query"] = "005930"
        at.run()
        self.assertFalse(at.exception)
        at.button(key="btn_run_sim").click().run()
        self.assertFalse(at.exception)
        # A matching current breakout has no historical equivalent: keep zero,
        # do not substitute the AUTO strategy behind the user's selection.
        self.assertTrue(any("과거 차트에서 일치하는 타점" in item.value for item in at.info))
        with patch("llm_analyst.generate_rag_analyst_report", return_value="검증된 통계 없음") as llm:
            at.button(key="btn_rag_report").click().run()
            self.assertFalse(at.exception)
            self.assertIn("matched_setups", llm.call_args.args[1])
        at.radio[1].set_value("중장기 대세 (2년 차트/주봉)").run()
        self.assertFalse(at.exception)
        at.button(key="btn_run_sim").click().run()
        self.assertFalse(at.exception)

    def test_scan_table_displays_unknown_investor_values(self):
        at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()
        at.radio(key="main_menu").set_value("💎 세력 매집 급등전야 포착").run()
        at.session_state["smart_money_scan_results_v2"] = pd.DataFrame([{
            "code": "005930", "name": "Example", "current_price": 100,
            "pct_from_low": 1., "ma_disp": 1., "cum_ret": 1., "vol_growth": 100.,
            "accum_dates_str": "2025-01-01", "frgn_sum": None, "inst_sum": None,
            "total_smart_money": None, "has_smart_money": False, "badge": "⚪ 수급 미확인",
        }])
        at.run()
        self.assertFalse(at.exception)
        table = at.dataframe[0].value
        self.assertEqual(table.iloc[0]["합산순매수"], "미확인")

    def test_invalid_price_data_shows_error_instead_of_crashing(self):
        self.data.loc[self.data.index[100], "Close"] = float("nan")
        at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()
        at.session_state["target_query"] = "999990"
        at.run()
        self.assertFalse(at.exception)
        self.assertTrue(at.error)


if __name__ == "__main__":
    unittest.main()
