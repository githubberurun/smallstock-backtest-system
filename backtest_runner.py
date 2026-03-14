import pandas as pd
import numpy as np
import os
import yfinance as yf
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

# ==========================================
# 2025-2026年 最新公式ドキュメント準拠
# Pandas: https://pandas.pydata.org/docs/
# yfinance: https://yfinance.readthedocs.io/en/latest/
# ==========================================

def debug_log(msg: str) -> None:
    """内部デバッグ用のロギング関数"""
    if not isinstance(msg, str): raise TypeError("msg must be a string")
    print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ==========================================
# 1. 小型株専用・統合分析エンジン (Stealth Trend Accumulation)
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
        
        # 過去の分析指標をすべて完全継承（他分析やダマシ判定との互換性維持のため）
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
        
        # ATR (ボラティリティ指標)
        tr = pd.concat([
            (df['high'] - df['low']), 
            (df['high'] - df['close'].shift()).abs(), 
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        df['tr'] = tr
        df['atr'] = df['tr'].rolling(window=14).mean() 
        
        # 【新指標：ボラティリティ比率（静かなる蓄積の検知）】
        df['atr_pct'] = (df['atr'] / df['close']) * 100
        
        # 出来高分析
        df['vol_ma25'] = df['volume'].rolling(25).mean().replace(0, np.nan)
        df['vol_ratio'] = (df['volume'] / df['vol_ma25']).fillna(0)
        
        df['is_bullish'] = df['close'] > df['open']
        day_range = (df['high'] - df['low']).replace(0, np.nan)
        df['close_position'] = ((df['close'] - df['low']) / day_range).fillna(0)
        df['lowest_5'] = df['low'].rolling(window=5).min()
        
        # TOPIX地合いフィルター（国内専用の防壁として機能）
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
    def evaluate_entry(row_dict: Dict[str, Any], n_chg: float, vix: float) -> Tuple[bool, float, bool]:
        if not isinstance(row_dict, dict): raise TypeError("row_dict must be a dictionary")
        
        # 日本固有の相場崩壊を防ぐため、TOPIXがダウントレンドの時は完全停止
        market_healthy = bool(row_dict.get('market_healthy', True))
        if not market_healthy or vix >= 30.0:
            return False, 0.0, False
            
        curr_c = SmallCapStrategyAnalyzer._to_float(row_dict.get('close', 0.0))
        ma50_val = SmallCapStrategyAnalyzer._to_float(row_dict.get('ma50', 0.0))
        ma200_val = SmallCapStrategyAnalyzer._to_float(row_dict.get('ma200', 0.0))
        
        atr_pct = SmallCapStrategyAnalyzer._to_float(row_dict.get('atr_pct', 10.0))
        vol_ratio = SmallCapStrategyAnalyzer._to_float(row_dict.get('vol_ratio', 1.0))
        rs_21 = SmallCapStrategyAnalyzer._to_float(row_dict.get('rs_21', 0.0))
        d25 = SmallCapStrategyAnalyzer._to_float(row_dict.get('dev25', 0.0))

        score = 0.0
        
        # 【ステルス・トレンド（静かなる蓄積）ロジック】
        # 1. 退屈で強固な上昇トレンド（MA50がMA200より上にあり、株価も上）
        if curr_c > ma50_val and ma50_val > ma200_val:
            
            # 2. TOPIXよりパフォーマンスが良い（相対強度プラス）
            if rs_21 > 0.0:
                
                # 3. ボラティリティが極端に低い（ATRが株価の4.0%未満 ＝ 乱高下しない退屈な銘柄）
                if atr_pct < 4.0:
                    
                    # 4. 誰も注目していない（出来高が25日平均の1.2倍未満 ＝ 煽りやイナゴがいない）
                    if vol_ratio < 1.2:
                        
                        # 5. MA25から極端に乖離していない（高値掴み防止）
                        if -2.0 <= d25 <= 5.0:
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
    def __init__(self, data_dir: str, initial_cash: float = 1000000.0, max_positions: int = 5) -> None:
        if not isinstance(data_dir, str): raise TypeError("data_dir must be string")
        self.cash: float = initial_cash
        self.initial_cash: float = initial_cash
        self.max_positions: int = 5 
        self.us_market = USMarketCache()
        
        self.stats = {
            'orders_placed': 0, 'orders_exec': 0,
            'time_stops': 0, 'hard_stops': 0, 'trailing_stops': 0,
            'breakeven_stops': 0, 'inago_exits': 0, 'gap_cancels': 0 
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
            
            new_pending = []
            for order in pending_orders:
                ticker = order['ticker']
                if ticker in today_market and len(positions) < self.max_positions:
                    row = today_market[ticker]
                    open_p = SmallCapStrategyAnalyzer._to_float(row.get('open', 0.0))
                    prev_close = SmallCapStrategyAnalyzer._to_float(row.get('prev_close', open_p))
                    lowest_5 = SmallCapStrategyAnalyzer._to_float(row.get('lowest_5', open_p))
                    
                    self.stats['orders_placed'] += 1
                    
                    # ギャップアップ・ダウン回避
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
                vol_ratio = SmallCapStrategyAnalyzer._to_float(row.get('vol_ratio', 1.0))
                
                pos['days_held'] += 1
                pos['high_p'] = max(pos['high_p'], curr_c)
                exit_score = 0
                
                # ブレークイーブン (1.5 ATR)
                if curr_c >= pos['entry_p'] + (current_atr * 1.5):
                    pos['breakeven_active'] = True

                # ハードストップ (退屈な銘柄なのでボラが少ない。2.0 ATRで十分)
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
                
                # 【イナゴ押し付けエグジット（真の利確）】
                # 退屈な銘柄に突然「出来高3倍以上」の火柱が立ち、RSIが75を超えて過熱したら、
                # イナゴ（一般個人）が群がってきた証拠。そこで大口と一緒に全株を売り抜ける！
                if vol_ratio >= 3.0 and rsi >= 75.0 and exit_score == 0:
                    exit_score += 100
                    self.stats['inago_exits'] += 1

                # トレイリングストップ (トレンドを伸ばすため少し広めの 3.0 ATR)
                trailing_stop_price = pos['high_p'] - (current_atr * 3.0)
                if curr_c <= trailing_stop_price and exit_score == 0:
                    exit_score += 100
                    self.stats['trailing_stops'] += 1
                
                # タイムストップ（20日）
                # 退屈な銘柄をジワジワ伸ばすため、ホールド期間は長めに許容する。1ヶ月経って含み益3%未満なら撤退。
                if pos['days_held'] >= 20 and curr_c < (pos['entry_p'] * 1.03) and exit_score == 0: 
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
                    
                    is_entry, score, is_high_risk = SmallCapStrategyAnalyzer.evaluate_entry(row, n_chg, vix)
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
    debug_log("Running integrity tests for Small Cap Logic...")
    empty_df = pd.DataFrame()
    res_df = SmallCapStrategyAnalyzer.calculate_indicators(empty_df)
    assert res_df.empty, "Empty DataFrame should return empty DataFrame"
    
    # ステルストレンド・テスト
    dummy_row_ok = {
        'close': 1050.0, 'ma50': 1000.0, 'ma200': 800.0,
        'atr_pct': 3.5, 'vol_ratio': 0.8, 'rs_21': 5.0, 'dev25': 2.0,
        'market_healthy': True
    }
    try:
        is_entry, score, is_risk = SmallCapStrategyAnalyzer.evaluate_entry(dummy_row_ok, 0.0, 15.0)
        assert isinstance(score, float)
        assert is_entry is True, "Valid stealth trend row should return True"
    except Exception as e:
        raise AssertionError(f"Failed handling valid data: {e}")

    dummy_row_err = {'atr_pct': np.nan, 'vol_ratio': 'invalid', 'market_healthy': 'error'}
    try:
        is_entry, score, is_risk = SmallCapStrategyAnalyzer.evaluate_entry(dummy_row_err, 0.0, 15.0)
        assert isinstance(score, float)
        assert is_entry is False
    except Exception as e:
        raise AssertionError(f"Failed to handle corrupted data safely: {e}")
        
    debug_log("All integrity tests passed.")

if __name__ == "__main__":
    run_integrity_tests()
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
        
        tester = SmallCapPortfolioBacktester(data_dir=data_dir, initial_cash=STARTING_CAPITAL, max_positions=MAX_CONCURRENT_POSITIONS)
        res = tester.run()
        
        print(f"\n==================================================")
        print(f" 📊 SMALL CAP SIMULATION RESULTS (STEALTH TREND STRATEGY)")
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
        print(f" [3] イナゴ売り抜け(真の利確): {st['inago_exits']} 回 ★NEW")
        print(f" [4] タイムストップ(20日)撤退: {st['time_stops']} 回")
        print(f" [5] 建値ストップ(負け回避): {st['breakeven_stops']} 回")
        print(f" [6] ハードストップ(安値割れ): {st['hard_stops']} 回")
        print(f" [7] トレイリングストップ(3.0ATR): {st['trailing_stops']} 回")
        print(f"==================================================", flush=True)
        
    except Exception as e:
        print(f"[FATAL] {e}")
