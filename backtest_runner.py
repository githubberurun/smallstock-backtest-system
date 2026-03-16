import pandas as pd
import numpy as np
import os
import json
import requests
import yfinance as yf
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta

# ==========================================
# 2025-2026年 最新公式ドキュメント準拠
# Pandas: https://pandas.pydata.org/docs/
# yfinance: https://yfinance.readthedocs.io/en/latest/
# J-Quants V2 財務情報仕様: https://jpx-jquants.com/ja/spec/fin-summary
# J-Quants V1->V2 移行仕様: https://jpx-jquants.com/spec/migration-v1-v2
# ==========================================

def debug_log(msg: str) -> None:
    """内部デバッグ用のロギング関数。引数の型チェックを含む。"""
    if not isinstance(msg, str): 
        msg = str(msg)
    print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ==========================================
# 1. J-Quants API v2 クライアント (V2公式仕様完全対応)
# ==========================================
class JQuantsV2Client:
    """JPX公式 J-Quants API v2 クライアント"""
    def __init__(self, api_key: str):
        if not isinstance(api_key, str): raise TypeError("api_key must be string")
        if not api_key: raise ValueError("API Key is required")
        
        self.api_key: str = api_key
        self.base_url: str = "https://api.jquants.com/v2" 
        
        self.is_api_broken: bool = False
        self.consecutive_403_count: int = 0

    def get_statements(self, ticker: str) -> Dict[str, Any]:
        """最新の財務情報を取得する"""
        if not isinstance(ticker, str): raise TypeError("ticker must be string")
        if self.is_api_broken: return {}
            
        headers = {
            "x-api-key": self.api_key,
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        url = f"{self.base_url}/fins/summary?code={ticker}"
        
        for attempt in range(2):
            try:
                time.sleep(0.6) 
                
                res = requests.get(url, headers=headers, timeout=10)
                
                if res.status_code == 404:
                    return {} 
                    
                if res.status_code in (401, 403):
                    debug_log(f"HTTP {res.status_code} for {ticker}. Response: {res.text}")
                    self.consecutive_403_count += 1
                    
                    if self.consecutive_403_count >= 5:
                        debug_log("CRITICAL: API Key rejected 5 times. Tripping circuit breaker. Bypassing J-Quants.")
                        self.is_api_broken = True
                        return {}
                        
                    time.sleep(2.0)
                    if attempt == 0: continue
                    else: res.raise_for_status()
                        
                elif res.status_code == 429:
                    debug_log(f"HTTP 429 Rate Limit. Sleeping 5s...")
                    time.sleep(5.0)
                    if attempt == 0: continue
                    else: res.raise_for_status()

                res.raise_for_status()
                self.consecutive_403_count = 0
                
                resp_json = res.json()
                data = resp_json.get("summary") or resp_json.get("data") or resp_json.get("statements") or []
                
                if not data:
                    return {}
                
                data.sort(key=lambda x: str(x.get("DiscDate") or x.get("DisclosedDate") or x.get("Date") or ""))
                
                for stmt in reversed(data):
                    if ("TA" in stmt or "TotalAssets" in stmt) and ("Eq" in stmt or "Equity" in stmt):
                        return stmt
                        
                return data[-1]
                
            except Exception as e:
                if "401" not in str(e) and "403" not in str(e) and "429" not in str(e):
                    debug_log(f"J-Quants URL({url}) Error: {e}")
                    break
                elif attempt == 1:
                    debug_log(f"J-Quants URL({url}) Error after retry: {e}")
                
        return {}

# ==========================================
# 2. ファンダメンタルズ・キャッシュマネージャー
# ==========================================
class FundamentalCache:
    """J-Quantsからの情報取得負荷を下げるためのローカルキャッシュ"""
    def __init__(self, data_dir: str, api_key: str):
        if not isinstance(data_dir, str): raise TypeError("data_dir must be string")
        if not isinstance(api_key, str): raise TypeError("api_key must be string")
        
        self.filepath = os.path.join(data_dir, "fundamentals_cache.json")
        self.jq_client = JQuantsV2Client(api_key)
        self.data: Dict[str, Dict[str, Any]] = self._load()

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                debug_log(f"Failed to load fundamental cache: {e}")
        return {}

    def get_fundamentals(self, ticker: str) -> Dict[str, float]:
        if not isinstance(ticker, str): raise TypeError("ticker must be string")
        
        if ticker in self.data:
            return {
                'roe': float(self.data[ticker].get('roe', 0.0)),
                'equity_ratio': float(self.data[ticker].get('equity_ratio', 0.0))
            }

        debug_log(f"Fetching fundamentals for {ticker}...")
        roe, equity_ratio = 0.0, 0.0
        success = False
        
        try:
            stmt = self.jq_client.get_statements(ticker)
            if not stmt:
                raise ValueError("No data returned from J-Quants")
            
            assets_raw = stmt.get("TA") or stmt.get("TotalAssets")
            equity_raw = stmt.get("Eq") or stmt.get("Equity")
            income_raw = stmt.get("NP") or stmt.get("Profit") or stmt.get("OP")
            
            if assets_raw is None or equity_raw is None:
                raise ValueError(f"Target keys missing in J-Quants response.")
                
            equity = float(equity_raw)
            total_assets = float(assets_raw)
            net_income = float(income_raw) if income_raw else 0.0
            
            equity_ratio = (equity / total_assets * 100.0) if total_assets > 0 else 0.0
            roe = (net_income / equity * 100.0) if equity > 0 else 0.0
            success = True
            
        except Exception as e:
            debug_log(f"J-Quants failed for {ticker} ({e}). Falling back to yfinance...")
            try:
                yf_ticker = f"{ticker}.T"
                info = yf.Ticker(yf_ticker).info
                
                r_raw = info.get('returnOnEquity', 0.0)
                roe = float(r_raw) * 100 if r_raw is not None else 0.0
                
                d_raw = info.get('debtToEquity', 0.0)
                dte = float(d_raw) if d_raw is not None else 0.0
                equity_ratio = 100.0 / (1.0 + (dte / 100.0)) if dte else 0.0
                success = True if (roe != 0.0 or equity_ratio != 0.0) else False
            except Exception as ye:
                debug_log(f"yfinance fallback also failed for {ticker}. Handled safely.")
                roe, equity_ratio = 0.0, 0.0
                
        self.data[ticker] = {
            'roe': roe, 
            'equity_ratio': equity_ratio,
            'status': 'success' if success else 'failed'
        }
        
        try:
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
            
        return {'roe': roe, 'equity_ratio': equity_ratio}

# ==========================================
# 3. 小型株専用・統合分析エンジン (138% Base Logic)
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
        
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['ma25'] = df['close'].rolling(window=25).mean()
        df['ma50'] = df['close'].rolling(window=50).mean()
        df['ma200'] = df['close'].rolling(window=200).mean()
        
        df['std20'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['ma20'] + (df['std20'] * 2)
        df['bb_lower'] = df['ma20'] - (df['std20'] * 2)
        
        df['bb_width'] = ((df['bb_upper'] - df['bb_lower']) / df['ma20']).replace([np.inf, -np.inf], np.nan).fillna(0)
        df['bb_width_min'] = df['bb_width'].rolling(window=20).min().fillna(0)
        
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
            benchmark_df['bm_ma25'] = benchmark_df['close'].rolling(window=25).mean()
            benchmark_df['bm_ma50'] = benchmark_df['close'].rolling(window=50).mean()
            
            # TOPIXが50日線を上回っているかを判定用データとして保持（フィルタリングは資金管理側で行う）
            benchmark_df['market_healthy'] = (benchmark_df['close'] > benchmark_df['bm_ma50'])
            
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
    def evaluate_entry(row_dict: Dict[str, Any], fund_data: Dict[str, float]) -> Tuple[bool, float, bool]:
        if not isinstance(row_dict, dict): raise TypeError("row_dict must be a dictionary")
        if not isinstance(fund_data, dict): raise TypeError("fund_data must be a dictionary")
        
        roe = SmallCapStrategyAnalyzer._to_float(fund_data.get('roe', 0.0))
        eq_ratio = SmallCapStrategyAnalyzer._to_float(fund_data.get('equity_ratio', 0.0))
        
        if roe < 10.0 or eq_ratio < 50.0:
            return False, 0.0, False 
            
        curr_c = SmallCapStrategyAnalyzer._to_float(row_dict.get('close', 0.0))
        ma50_val = SmallCapStrategyAnalyzer._to_float(row_dict.get('ma50', 0.0))
        ma200_val = SmallCapStrategyAnalyzer._to_float(row_dict.get('ma200', 0.0))
        
        vol_ratio = SmallCapStrategyAnalyzer._to_float(row_dict.get('vol_ratio', 1.0))
        is_bullish = bool(row_dict.get('is_bullish', False))
        close_pos = SmallCapStrategyAnalyzer._to_float(row_dict.get('close_position', 0.0))
        
        bb_width = SmallCapStrategyAnalyzer._to_float(row_dict.get('bb_width', 1.0))
        rs_21 = SmallCapStrategyAnalyzer._to_float(row_dict.get('rs_21', 0.0))
        rsi = SmallCapStrategyAnalyzer._to_float(row_dict.get('rsi', 0.0))

        score = 0.0
        
        # 【138%爆発エンジン】エントリー基準は最高効率（取引回数480回水準）のものに固定
        if curr_c > ma50_val and ma50_val > ma200_val:
            if bb_width <= 0.25:
                if vol_ratio >= 2.0 and is_bullish and close_pos >= 0.70 and rs_21 > 0.0 and rsi < 80.0:
                    score += 100.0
                
        is_entry = (score >= 100.0) 
        return is_entry, float(score), False

# ==========================================
# 4. 米国市場キャッシュ & ポートフォリオバックテスター
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
            'take_profit': 0, 'trailing_stops': 0, 'hard_stops': 0,
            'breakeven_stops': 0, 'time_stops': 0, 'gap_cancels': 0,
            'half_size_entries': 0 # 新規: 動的資金管理による半量エントリー回数
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
        
    def run(self) -> Dict[str, Any]:
        cash = self.cash
        positions: Dict[str, Dict[str, Any]] = {} 
        pending_orders: List[Dict[str, Any]] = [] 
        equity_curve: List[float] = []
        total_trades = 0

        for date_str in self.sorted_dates:
            today_market = self.timeline[date_str]
            n_chg, vix = self.us_market.get_state(date_str)
            
            current_total_equity = cash + sum([p['qty'] * SmallCapStrategyAnalyzer._to_float(today_market.get(t, {}).get('close', p['entry_p'])) for t, p in positions.items()])
            
            # その日の相場の健康状態をチェック（どの銘柄でも同じ値が入っている）
            is_market_healthy = True
            if today_market:
                first_ticker = list(today_market.keys())[0]
                is_market_healthy = bool(today_market[first_ticker].get('market_healthy', True))

            new_pending = []
            for order in pending_orders:
                ticker = order['ticker']
                if ticker in today_market and len(positions) < self.max_positions:
                    row = today_market[ticker]
                    open_p = SmallCapStrategyAnalyzer._to_float(row.get('open', 0.0))
                    prev_close = SmallCapStrategyAnalyzer._to_float(row.get('prev_close', open_p))
                    lowest_5 = SmallCapStrategyAnalyzer._to_float(row.get('lowest_5', open_p))
                    
                    self.stats['orders_placed'] += 1
                    
                    if open_p > (prev_close * 1.05) or open_p < (prev_close * 0.98):
                        self.stats['gap_cancels'] += 1
                        continue 
                        
                    exec_price = open_p * 1.002 
                    alloc_cash = order['allocated_cash']
                    qty = alloc_cash // exec_price
                    
                    if qty > 0 and cash >= (qty * exec_price):
                        cash -= qty * exec_price
                        positions[ticker] = {
                            'qty': qty, 'entry_p': exec_price, 'high_p': exec_price, 
                            'days_held': 0, 'swing_low': lowest_5,
                            'breakeven_active': False
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

                # 138%エンジンのエグジット（完全維持）
                if (curr_c >= pos['entry_p'] * 1.25 or rsi >= 85.0) and exit_score == 0:
                    exit_score += 100
                    self.stats['take_profit'] += 1

                if pos['high_p'] >= pos['entry_p'] + (current_atr * 2.0):
                    pos['breakeven_active'] = True

                hard_stop_price = pos['entry_p'] - (current_atr * 2.0)
                if pos['swing_low'] > 0:
                    hard_stop_price = min(hard_stop_price, pos['swing_low'] * 0.98) 
                
                if pos.get('breakeven_active', False):
                    hard_stop_price = max(hard_stop_price, pos['entry_p'] * 1.005) 
                
                if curr_c <= hard_stop_price and exit_score == 0:
                    exit_score += 100
                    if pos.get('breakeven_active', False):
                        self.stats['breakeven_stops'] += 1
                    else:
                        self.stats['hard_stops'] += 1
                        
                trailing_stop_price = pos['high_p'] - (current_atr * 2.5)
                if curr_c <= trailing_stop_price and exit_score == 0:
                    exit_score += 100
                    self.stats['trailing_stops'] += 1
                
                if pos['days_held'] >= 8 and curr_c <= (pos['entry_p'] * 1.02) and exit_score == 0: 
                    exit_score += 100
                    self.stats['time_stops'] += 1

                if exit_score >= 80:
                    cash += pos['qty'] * (curr_c * 0.998)
                    total_trades += 1
                    closed_tickers.append(ticker)

            for ct in closed_tickers:
                del positions[ct]

            # 【真のMDDキラー：動的ポジションサイジング（可変レバレッジ）】
            open_slots = self.max_positions - len(positions)
            
            # VIXが25以上のパニック相場では新規エントリーを見送る（暴落回避）
            if open_slots > 0 and cash > 0 and vix < 25.0:
                
                # リスク係数の算出：地合いが悪い、またはVIXが高い時は投資額を「半分（0.5）」にする
                risk_multiplier = 1.0
                if not is_market_healthy or vix >= 20.0:
                    risk_multiplier = 0.5
                
                candidates = []
                for ticker, row in today_market.items():
                    if ticker in positions: continue 
                    
                    fund_data = self.fund_cache.get_fundamentals(ticker)
                    # 評価関数からはVIXと地合いフィルターを外し、ロジック純度のみで判定
                    is_entry, score, is_high_risk = SmallCapStrategyAnalyzer.evaluate_entry(row, fund_data)
                    
                    if is_entry:
                        candidates.append((score, ticker))
                
                candidates.sort(key=lambda x: x[0], reverse=True)
                allowed_slots_today = min(open_slots, self.max_positions)
                
                for score, ticker in candidates[:allowed_slots_today]:
                    # 基本の投資枠（総資産の20%）にリスク係数を掛ける
                    max_alloc_per_trade = (current_total_equity * (1.0 / self.max_positions)) * risk_multiplier
                    target_alloc = min(cash / open_slots, max_alloc_per_trade)
                    
                    pending_orders.append({
                        'ticker': ticker,
                        'allocated_cash': target_alloc
                    })
                    if risk_multiplier == 0.5:
                        self.stats['half_size_entries'] += 1
                    
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
# 5. 空データ・異常値に対する堅牢性証明テスト
# ==========================================
def run_integrity_tests() -> None:
    debug_log("Running integrity tests for Q-Mo Logic...")
    empty_df = pd.DataFrame()
    res_df = SmallCapStrategyAnalyzer.calculate_indicators(empty_df)
    assert res_df.empty, "Empty DataFrame should return empty DataFrame"
    
    dummy_row_ok = {
        'close': 1050.0, 'ma50': 1000.0, 'ma200': 900.0,
        'bb_width': 0.20, 'bb_width_min': 0.18, 
        'vol_ratio': 2.5, 'is_bullish': True, 'close_position': 0.8, 
        'market_healthy': True, 'rs_21': 5.0, 'rsi': 65.0
    }
    dummy_fund_ok = {'roe': 12.0, 'equity_ratio': 55.0}
    try:
        is_entry, score, is_risk = SmallCapStrategyAnalyzer.evaluate_entry(dummy_row_ok, dummy_fund_ok)
        assert isinstance(score, float)
        assert is_entry is True, "Valid Scaled VCP Breakout row should return True"
    except Exception as e:
        raise AssertionError(f"Failed handling valid data: {e}")

    dummy_fund_bad = {'roe': 5.0, 'equity_ratio': 60.0}
    try:
        is_entry, score, is_risk = SmallCapStrategyAnalyzer.evaluate_entry(dummy_row_ok, dummy_fund_bad)
        assert is_entry is False, "Zombie company must be rejected"
    except Exception as e:
        raise AssertionError(f"Failed handling zombie data: {e}")

    dummy_row_err = {'ma200': np.nan, 'vol_ratio': 'invalid', 'market_healthy': 'error'}
    dummy_fund_err = {'roe': None, 'equity_ratio': 'error'}
    try:
        is_entry, score, is_risk = SmallCapStrategyAnalyzer.evaluate_entry(dummy_row_err, dummy_fund_err)
        assert isinstance(score, float)
        assert is_entry is False
    except Exception as e:
        raise AssertionError(f"Failed to handle corrupted data safely: {e}")
        
    debug_log("All integrity tests passed.")

if __name__ == "__main__":
    JQ_API_KEY = os.environ.get("JQUANTS_API_KEY", "")
    
    run_integrity_tests()
    
    if not JQ_API_KEY:
        print("\n[ERROR] J-Quants APIキーが環境変数 'JQUANTS_API_KEY' にセットされていません。")
        print("GitHub Secretsの設定を確認してください。")
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
        print(f" 📊 SMALL CAP SIMULATION RESULTS (DYNAMIC SIZING VCP)")
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
        print(f" 🔬 動的資金管理・分析レポート")
        print(f" [1] 成行の約定状況: {st['orders_exec']}/{st['orders_placed']} ({exec_rate:.1f}%)")
        print(f" [2] ギャップ回避: {st['gap_cancels']} 回")
        print(f" [3] 防御モード(半量)発動: {st['half_size_entries']} 回")
        print(f" [4] クライマックス利確: {st['take_profit']} 回")
        print(f" [5] トレイリングストップ: {st['trailing_stops']} 回")
        print(f" [6] 建値ストップ: {st['breakeven_stops']} 回")
        print(f" [7] ハードストップ: {st['hard_stops']} 回")
        print(f" [8] タイムストップ撤退: {st['time_stops']} 回")
        print(f"==================================================", flush=True)
        
    except Exception as e:
        print(f"[FATAL] {e}")
