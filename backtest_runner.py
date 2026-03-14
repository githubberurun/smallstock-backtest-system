import pandas as pd
import numpy as np
import os
import json
import requests
import yfinance as yf
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta

# ==========================================
# 2025-2026年 最新公式ドキュメント準拠
# Pandas: https://pandas.pydata.org/docs/
# yfinance: https://yfinance.readthedocs.io/en/latest/
# J-Quants API v2: https://jpx-jquants.com/api/
# ==========================================

def debug_log(msg: str) -> None:
    """内部デバッグ用のロギング関数。引数の型チェックを含む。"""
    if not isinstance(msg, str): 
        msg = str(msg)
    print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ==========================================
# 1. J-Quants API v2 クライアント (リフレッシュトークン方式修正版)
# ==========================================
class JQuantsV2Client:
    """JPX公式 J-Quants API v2 クライアント"""
    def __init__(self, api_key: str):
        if not isinstance(api_key, str): raise TypeError("api_key must be string")
        if not api_key: raise ValueError("API Key (Refresh Token) is required")
        self.api_key: str = api_key
        self.id_token: str = ""
        self.base_url: str = "https://api.jquants.com/v1" 
        self._refresh_id_token()

    def _refresh_id_token(self) -> None:
        """リフレッシュトークンを使用してIDトークンを取得・更新する"""
        # 修正: auth_refreshエンドポイントにクエリパラメータとしてトークンを渡す
        url = f"{self.base_url}/token/auth_refresh?refreshtoken={self.api_key}"
        try:
            res = requests.post(url, timeout=15)
            res.raise_for_status()
            self.id_token = res.json().get("idToken", "")
            if self.id_token:
                debug_log("Successfully authenticated with J-Quants API.")
            else:
                debug_log("Warning: Authentication succeeded but idToken is empty.")
        except Exception as e:
            debug_log(f"Failed J-Quants v2 Auth: {e}")
            self.id_token = ""

    def get_statements(self, ticker: str) -> Dict[str, Any]:
        """最新の財務諸表（短信）を取得する"""
        if not isinstance(ticker, str): raise TypeError("ticker must be string")
        if not self.id_token: 
            self._refresh_id_token()
            if not self.id_token: return {}
        
        # 修正: 財務情報の公式エンドポイントは /fins/statements
        url = f"{self.base_url}/fins/statements?code={ticker}"
        headers = {"Authorization": f"Bearer {self.id_token}"}
        
        try:
            res = requests.get(url, headers=headers, timeout=15)
            if res.status_code == 401: # トークン切れの自動再試行
                self._refresh_id_token()
                headers["Authorization"] = f"Bearer {self.id_token}"
                res = requests.get(url, headers=headers, timeout=15)
            
            res.raise_for_status()
            data = res.json().get("statements", [])
            if not data:
                return {}
            
            # 複数ある場合は DisclosedDate でソートして最新（配列の最後）を取得
            data.sort(key=lambda x: x.get("DisclosedDate", ""))
            return data[-1]
            
        except Exception as e:
            debug_log(f"J-Quants API Error for {ticker}: {e}")
            return {}

# ==========================================
# ファンダメンタルズ・キャッシュマネージャー（J-Quants統合・自動修復版）
# ==========================================
class FundamentalCache:
    """J-Quantsからの情報取得負荷を下げるためのローカルキャッシュ"""
    def __init__(self, data_dir: str, api_key: str):
        if not isinstance(data_dir, str): raise TypeError("data_dir must be string")
        if not isinstance(api_key, str): raise TypeError("api_key must be string")
        
        self.filepath = os.path.join(data_dir, "fundamentals_cache.json")
        self.jq_client = JQuantsV2Client(api_key)
        self.data: Dict[str, Dict[str, float]] = self._load()

    def _load(self) -> Dict[str, Dict[str, float]]:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                debug_log(f"Failed to load fundamental cache: {e}")
        return {}

    def get_fundamentals(self, ticker: str) -> Dict[str, float]:
        if not isinstance(ticker, str): raise TypeError("ticker must be string")
        
        # 【自動修復ロジック】キャッシュが存在しない、または前回エラー時の 0.0 データの場合は再取得
        needs_fetch = False
        if ticker not in self.data:
            needs_fetch = True
        elif self.data[ticker].get('roe', 0.0) == 0.0 and self.data[ticker].get('equity_ratio', 0.0) == 0.0:
            needs_fetch = True
            
        if needs_fetch:
            debug_log(f"Fetching fundamentals for {ticker} from J-Quants...")
            try:
                stmt = self.jq_client.get_statements(ticker)
                
                # 修正: J-Quantsのキーに合わせて取得 (Profit: 純利益, Equity: 自己資本, TotalAssets: 総資産)
                equity_raw = stmt.get("Equity")
                assets_raw = stmt.get("TotalAssets")
                income_raw = stmt.get("Profit")
                
                # Noneや空文字に対する堅牢なパース
                equity = float(equity_raw) if equity_raw else 0.0
                total_assets = float(assets_raw) if assets_raw else 0.0
                net_income = float(income_raw) if income_raw else 0.0
                
                # 自己資本比率 (%) = 自己資本 / 総資産 * 100
                equity_ratio = (equity / total_assets * 100.0) if total_assets > 0 else 0.0
                
                # ROE (%) = 当期純利益 / 自己資本 * 100
                roe = (net_income / equity * 100.0) if equity > 0 else 0.0
                
                self.data[ticker] = {'roe': roe, 'equity_ratio': equity_ratio}
            except Exception as e:
                debug_log(f"Error calculating fundamental for {ticker}: {e}")
                self.data[ticker] = {'roe': 0.0, 'equity_ratio': 0.0} 
            
            # API制限・消失回避のため、1件ごとに保存
            try:
                with open(self.filepath, 'w', encoding='utf-8') as f:
                    json.dump(self.data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                debug_log(f"Failed to save cache: {e}")
                
        return self.data[ticker]

# ==========================================
# 1. 小型株専用・統合分析エンジン (Q-Mo Strategy)
# ==========================================
class SmallCapStrategyAnalyzer:
    @staticmethod
    def _to_float(val: Any, default: float = 0.0) -> float:
        try:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return default
            f = float(val)
            return f if np.isfinite(f) else default
        except (ValueError, TypeError):
            return default

    @staticmethod
    def calculate_indicators(df: pd.DataFrame, benchmark_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if df.empty or len(df) < 200: 
            return df
            
        df.columns = [str(c).lower() for c in df.columns]
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset(set(df.columns)):
            raise KeyError(f"DataFrameに必須列が不足しています: {required_cols - set(df.columns)}")

        df['prev_close'] = df['close'].shift(1)
        
        # 過去指標の完全継承
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma25'] = df['close'].rolling(window=25).mean()
        df['ma50'] = df['close'].rolling(window=50).mean()
        df['ma75'] = df['close'].rolling(window=75).mean()
        df['ma200'] = df['close'].rolling(window=200).mean()
        df['dev25'] = (df['close'] - df['ma25']) / df['ma25'] * 100
        
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['std20'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['ma20'] + (df['std20'] * 2)
        df['bb_lower'] = df['ma20'] - (df['std20'] * 2)
        df['bb_m3'] = df['ma20'] - (df['std20'] * 3)
        df['bb_m1'] = df['ma20'] - (df['std20'] * 1)
        
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['sig'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['sig']
        df['macd_improving'] = df['macd_hist'] > df['macd_hist'].shift(1)
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, np.nan)).fillna(0)))
        
        tr = pd.concat([
            (df['high'] - df['low']), 
            (df['high'] - df['close'].shift()).abs(), 
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        df['tr'] = tr
        df['atr'] = df['tr'].rolling(window=14).mean() 
        df['atr_pct'] = (df['atr'] / df['close']) * 100
        
        df['vol_ma25'] = df['volume'].rolling(25).mean().replace(0, np.nan)
        df['vol_ratio'] = (df['volume'] / df['vol_ma25']).fillna(0)
        
        df['is_bullish'] = df['close'] > df['open']
        day_range = (df['high'] - df['low']).replace(0, np.nan)
        df['close_position'] = ((df['close'] - df['low']) / day_range).fillna(0)
        df['lowest_5'] = df['low'].rolling(window=5).min()
        
        if benchmark_df is not None and not benchmark_df.empty:
            benchmark_df.columns = [str(c).lower() for c in benchmark_df.columns]
            benchmark_df['bm_ma50'] = benchmark_df['close'].rolling(window=50).mean()
            benchmark_df['bm_ma200'] = benchmark_df['close'].rolling(window=200).mean()
            benchmark_df['market_healthy'] = (benchmark_df['close'] > benchmark_df['bm_ma50']) & (benchmark_df['bm_ma50'] > benchmark_df['bm_ma200'])
            df = df.merge(benchmark_df[['date', 'market_healthy']], on='date', how='left')
            df['market_healthy'] = df['market_healthy'].ffill().fillna(False)
            
            df = df.merge(benchmark_df[['date', 'close']], on='date', how='left', suffixes=('', '_bm'))
            df['close_bm'] = df['close_bm'].ffill()
            df['rs_21'] = (df['close'].pct_change(21) - df['close_bm'].pct_change(21)) * 100
        else:
            df['market_healthy'] = True
            df['rs_21'] = 0.0

        return df

    @staticmethod
    def evaluate_entry(row_dict: Dict[str, Any], n_chg: float, vix: float, fund_data: Dict[str, float]) -> Tuple[bool, float, bool]:
        if not isinstance(row_dict, dict): raise TypeError("row_dict must be a dictionary")
        if not isinstance(fund_data, dict): raise TypeError("fund_data must be a dictionary")
        
        market_healthy = bool(row_dict.get('market_healthy', True))
        if not market_healthy or vix >= 25.0:
            return False, 0.0, False
            
        roe = SmallCapStrategyAnalyzer._to_float(fund_data.get('roe', 0.0))
        eq_ratio = SmallCapStrategyAnalyzer._to_float(fund_data.get('equity_ratio', 0.0))
        
        if roe < 10.0 or eq_ratio < 50.0:
            return False, 0.0, False 
            
        curr_c = SmallCapStrategyAnalyzer._to_float(row_dict.get('close', 0.0))
        ma25_val = SmallCapStrategyAnalyzer._to_float(row_dict.get('ma25', 0.0))
        ma50_val = SmallCapStrategyAnalyzer._to_float(row_dict.get('ma50', 0.0))
        ma200_val = SmallCapStrategyAnalyzer._to_float(row_dict.get('ma200', 0.0))
        
        vol_ratio = SmallCapStrategyAnalyzer._to_float(row_dict.get('vol_ratio', 1.0))
        is_bullish = bool(row_dict.get('is_bullish', False))
        close_pos = SmallCapStrategyAnalyzer._to_float(row_dict.get('close_position', 0.0))

        score = 0.0
        
        if curr_c > ma25_val and ma25_val > ma50_val and ma50_val > ma200_val:
            if vol_ratio >= 2.0 and is_bullish and close_pos >= 0.70:
                score += 100.0
                
        is_entry = (score >= 100.0) 
        return is_entry, float(score), False

# ==========================================
# 2. 米国市場キャッシュ & ポートフォリオバックテスター
# ==========================================
class USMarketCache:
    def __init__(self) -> None:
        debug_log("Caching US market data...")
        try:
            ndx_data = yf.Ticker("^IXIC").history(period="10y")
            vix_data = yf.Ticker("^VIX").history(period="10y")
            if not ndx_data.empty and not vix_data.empty:
                self.ndx = ndx_data['Close'].pct_change() * 100
                self.vix = vix_data['Close']
                self.ndx.index = self.ndx.index.tz_localize(None).strftime('%Y-%m-%d')
                self.vix.index = self.vix.index.tz_localize(None).strftime('%Y-%m-%d')
            else:
                self.ndx, self.vix = pd.Series(dtype=float), pd.Series(dtype=float)
        except Exception:
            self.ndx, self.vix = pd.Series(dtype=float), pd.Series(dtype=float)

    def get_state(self, date_str: str) -> Tuple[float, float]:
        if not isinstance(date_str, str): raise TypeError("date_str must be string")
        if self.ndx.empty or self.vix.empty: return 0.0, 15.0
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        for i in range(1, 6):
            prev = (dt - timedelta(days=i)).strftime('%Y-%m-%d')
            if prev in self.ndx.index and prev in self.vix.index: 
                return float(self.ndx[prev]), float(self.vix[prev])
        return 0.0, 15.0

class SmallCapPortfolioBacktester:
    def __init__(self, data_dir: str, api_key: str, initial_cash: float = 1000000.0, max_positions: int = 5) -> None:
        if not isinstance(data_dir, str): raise TypeError("data_dir must be string")
        if not isinstance(api_key, str): raise TypeError("api_key must be string")
        
        self.cash: float = initial_cash
        self.initial_cash: float = initial_cash
        self.max_positions: int = max_positions
        self.us_market = USMarketCache()
        self.fund_cache = FundamentalCache(data_dir, api_key)
        
        self.stats = {
            'orders_placed': 0, 'orders_exec': 0,
            'time_stops': 0, 'hard_stops': 0, 'trailing_stops': 0,
            'breakeven_stops': 0, 'climax_exits': 0, 'gap_cancels': 0
        }
        
        debug_log("Loading and calculating indicators for SMALL CAP tickers...")
        self.timeline: Dict[str, Dict[str, Dict[str, Any]]] = {}
        dates_set = set()
        
        bm_path = f"{data_dir}/13060.parquet"
        bm_df = pd.read_parquet(bm_path) if os.path.exists(bm_path) else None
        
        files = [f for f in os.listdir(data_dir) if f.endswith(".parquet") and f != "13060.parquet"]
        
        for file in files:
            ticker = file.replace(".parquet", "")
            df = pd.read_parquet(f"{data_dir}/{file}")
            df = SmallCapStrategyAnalyzer.calculate_indicators(df, bm_df)
            if df.empty: continue
            
            self.fund_cache.get_fundamentals(ticker)
            
            df['date_str'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            records = df.to_dict(orient='records')
            
            for row in records:
                d_str = row['date_str']
                dates_set.add(d_str)
                if d_str not in self.timeline: self.timeline[d_str] = {}
                self.timeline[d_str][ticker] = row
                
        self.sorted_dates = sorted(list(dates_set))
        debug_log(f"Timeline built. Total trading days: {len(self.sorted_dates)}")
        
        passed_tickers = [t for t, data in self.fund_cache.data.items() if data.get('roe', 0.0) >= 10.0 and data.get('equity_ratio', 0.0) >= 50.0]
        debug_log(f"★ Quality Filter Passed Tickers (ROE>=10%, Eq>=50%): {len(passed_tickers)} / {len(self.fund_cache.data)}")

    def run(self) -> Dict[str, Any]:
        cash = self.cash
        positions: Dict[str, Dict[str, Any]] = {} 
        pending_orders: List[Dict[str, Any]] = [] 
        equity_curve: List[float] = []
        total_trades = 0

        for date_str in self.sorted_dates:
            today_market = self.timeline[date_str]
            n_chg, vix = self.us_market.get_state(date_str)
            
            new_pending = []
            for order in pending_orders:
                ticker = order['ticker']
                if ticker in today_market and len(positions) < self.max_positions:
                    row = today_market[ticker]
                    open_p = SmallCapStrategyAnalyzer._to_float(row.get('open', 0.0))
                    prev_close = SmallCapStrategyAnalyzer._to_float(row.get('prev_close', open_p))
                    lowest_5 = SmallCapStrategyAnalyzer._to_float(row.get('lowest_5', open_p))
                    
                    self.stats['orders_placed'] += 1
                    
                    if open_p > (prev_close * 1.02) or open_p < (prev_close * 0.98):
                        self.stats['gap_cancels'] += 1
                        continue 
                        
                    exec_price = open_p * 1.002 
                    alloc_cash = order['allocated_cash']
                    qty = alloc_cash // exec_price
                    
                    if qty > 0 and cash >= (qty * exec_price):
                        cash -= qty * exec_price
                        positions[ticker] = {
                            'qty': qty, 'entry_p': exec_price, 'high_p': exec_price, 
                            'days_held': 0, 'breakeven_active': False,
                            'swing_low': lowest_5
                        }
                        self.stats['orders_exec'] += 1
            pending_orders = new_pending

            closed_tickers = []
            for ticker, pos in positions.items():
                if ticker not in today_market: continue
                row = today_market[ticker]
                
                curr_c = SmallCapStrategyAnalyzer._to_float(row.get('close', 0.0))
                current_atr = SmallCapStrategyAnalyzer._to_float(row.get('atr', 0.0))
                rsi = SmallCapStrategyAnalyzer._to_float(row.get('rsi', 0.0))
                
                pos['days_held'] += 1
                pos['high_p'] = max(pos['high_p'], curr_c)
                exit_score = 0
                
                if curr_c >= pos['entry_p'] + (current_atr * 1.5):
                    pos['breakeven_active'] = True

                atr_stop = pos['entry_p'] - (current_atr * 2.0)
                hard_stop_price = min(atr_stop, pos['swing_low'] * 0.99)
                
                if pos['breakeven_active']:
                    hard_stop_price = max(hard_stop_price, pos['entry_p'] * 1.005)
                
                if curr_c <= hard_stop_price:
                    exit_score += 100
                    if pos['breakeven_active']:
                        self.stats['breakeven_stops'] += 1
                    else:
                        self.stats['hard_stops'] += 1 
                
                if rsi >= 85.0 and exit_score == 0:
                    exit_score += 100
                    self.stats['climax_exits'] += 1

                trailing_stop_price = pos['high_p'] - (current_atr * 3.0)
                if curr_c <= trailing_stop_price and exit_score == 0:
                    exit_score += 100
                    self.stats['trailing_stops'] += 1
                
                if pos['days_held'] >= 15 and curr_c < (pos['entry_p'] * 1.03) and exit_score == 0: 
                    exit_score += 100
                    self.stats['time_stops'] += 1

                if exit_score >= 80:
                    cash += pos['qty'] * (curr_c * 0.998)
                    total_trades += 1
                    closed_tickers.append(ticker)

            for ct in closed_tickers:
                del positions[ct]

            open_slots = self.max_positions - len(positions)
            if open_slots > 0 and cash > 0:
                candidates = []
                for ticker, row in today_market.items():
                    if ticker in positions: continue 
                    
                    fund_data = self.fund_cache.get_fundamentals(ticker)
                    is_entry, score, is_high_risk = SmallCapStrategyAnalyzer.evaluate_entry(row, n_chg, vix, fund_data)
                    
                    if is_entry:
                        candidates.append((score, ticker))
                
                candidates.sort(key=lambda x: x[0], reverse=True)
                allowed_slots_today = min(open_slots, self.max_positions)
                
                for score, ticker in candidates[:allowed_slots_today]:
                    target_alloc = cash / open_slots 
                    pending_orders.append({
                        'ticker': ticker,
                        'allocated_cash': target_alloc
                    })
                    open_slots -= 1

            daily_equity = cash
            for ticker, pos in positions.items():
                if ticker in today_market:
                    curr_c = SmallCapStrategyAnalyzer._to_float(today_market[ticker].get('close', pos['entry_p']))
                    daily_equity += pos['qty'] * (curr_c * 0.998)
            equity_curve.append(daily_equity)

        final_equity = equity_curve[-1] if equity_curve else self.initial_cash
        
        if equity_curve:
            eq_series = pd.Series(equity_curve)
            cummax = eq_series.cummax()
            mdd_series = (eq_series - cummax) / cummax
            mdd = float(mdd_series.min()) if not pd.isna(mdd_series.min()) else 0.0
        else:
            mdd = 0.0
            
        ret_val = (final_equity - self.initial_cash) / self.initial_cash
        
        return {
            "Initial_Cash": self.initial_cash,
            "Final_Cash": final_equity,
            "Net_Profit": final_equity - self.initial_cash,
            "Return": f"{ret_val:.2%}",
            "MDD": f"{mdd:.2%}",
            "Total_Trades": total_trades,
            "Stats": self.stats
        }

# ==========================================
# 3. 空データ・異常値に対する堅牢性証明テスト
# ==========================================
def run_integrity_tests() -> None:
    debug_log("Running integrity tests for Q-Mo Logic...")
    empty_df = pd.DataFrame()
    res_df = SmallCapStrategyAnalyzer.calculate_indicators(empty_df)
    assert res_df.empty, "Empty DataFrame should return empty DataFrame"
    
    dummy_row_ok = {
        'close': 1050.0, 'ma25': 1000.0, 'ma50': 950.0, 'ma200': 800.0,
        'vol_ratio': 2.5, 'is_bullish': True, 'close_position': 0.8,
        'market_healthy': True
    }
    dummy_fund_ok = {'roe': 12.0, 'equity_ratio': 55.0}
    try:
        is_entry, score, is_risk = SmallCapStrategyAnalyzer.evaluate_entry(dummy_row_ok, 0.0, 15.0, dummy_fund_ok)
        assert isinstance(score, float)
        assert is_entry is True, "Valid Q-Mo row should return True"
    except Exception as e:
        raise AssertionError(f"Failed handling valid data: {e}")

    dummy_fund_bad = {'roe': 5.0, 'equity_ratio': 60.0}
    try:
        is_entry, score, is_risk = SmallCapStrategyAnalyzer.evaluate_entry(dummy_row_ok, 0.0, 15.0, dummy_fund_bad)
        assert is_entry is False, "Zombie company must be rejected"
    except Exception as e:
        raise AssertionError(f"Failed handling zombie data: {e}")

    dummy_row_err = {'ma200': np.nan, 'vol_ratio': 'invalid', 'market_healthy': 'error'}
    dummy_fund_err = {'roe': None, 'equity_ratio': 'error'}
    try:
        is_entry, score, is_risk = SmallCapStrategyAnalyzer.evaluate_entry(dummy_row_err, 0.0, 15.0, dummy_fund_err)
        assert isinstance(score, float)
        assert is_entry is False
    except Exception as e:
        raise AssertionError(f"Failed to handle corrupted data safely: {e}")
        
    debug_log("All integrity tests passed.")

if __name__ == "__main__":
    # GitHub Actions の環境変数(Secrets)から動的にAPIキーを取得する
    JQ_API_KEY = os.environ.get("JQUANTS_API_KEY", "")
    
    run_integrity_tests()
    
    if not JQ_API_KEY:
        print("\n[ERROR] J-Quants v2 APIキーが環境変数 'JQUANTS_API_KEY' にセットされていません。")
        print("GitHub Secretsの設定、またはローカルの環境変数を確認して再試行してください。")
        exit(1)
        
    try:
        data_dir = "Colog_github"
        if not os.path.exists(data_dir):
            print(f"[ERROR] Directory '{data_dir}' not found. Run data fetcher first.")
            exit(1)
            
        print("\n==================================================")
        print(" 🚀 STARTING SMALL CAP PORTFOLIO BACKTEST")
        print("==================================================")
        
        STARTING_CAPITAL = 1000000.0
        MAX_CONCURRENT_POSITIONS = 5
        
        tester = SmallCapPortfolioBacktester(data_dir=data_dir, api_key=JQ_API_KEY, initial_cash=STARTING_CAPITAL, max_positions=MAX_CONCURRENT_POSITIONS)
        res = tester.run()
        
        print(f"\n==================================================")
        print(f" 📊 SMALL CAP SIMULATION RESULTS (Q-Mo FUNDAMENTAL STRATEGY)")
        print(f"==================================================")
        print(f" ▶ 初期資金 (Initial Cash) : ¥{int(res['Initial_Cash']):,}")
        print(f" ▶ 最終資産 (Final Cash)   : ¥{int(res['Final_Cash']):,}")
        print(f" ▶ 純利益 (Net Profit)     : ¥{int(res['Net_Profit']):,}")
        print(f" ▶ 総利回り (Return)       : {res['Return']}")
        print(f" ▶ 最大下落率 (MDD)        : {res['MDD']}")
        print(f" ▶ 総取引回数 (Trades)     : {res['Total_Trades']} 回")
        
        st = res['Stats']
        exec_rate = (st['orders_exec'] / st['orders_placed']) * 100 if st['orders_placed'] > 0 else 0
        
        print(f"==================================================")
        print(f" 🔬 小型株ロジック 分析レポート")
        print(f" [1] 成行の約定状況: {st['orders_exec']}/{st['orders_placed']} ({exec_rate:.1f}%)")
        print(f" [2] ギャップ（窓開け）回避: {st['gap_cancels']} 回")
        print(f" [3] タイムストップ(15日)撤退: {st['time_stops']} 回")
        print(f" [4] 建値ストップ(負け回避): {st['breakeven_stops']} 回")
        print(f" [5] ハードストップ(安値割れ): {st['hard_stops']} 回")
        print(f" [6] トレイリングストップ(3.0ATR): {st['trailing_stops']} 回")
        print(f" [7] クライマックス売り(過熱極致): {st['climax_exits']} 回")
        print(f"==================================================", flush=True)
        
    except Exception as e:
        print(f"[FATAL] {e}")
