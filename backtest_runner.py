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
# 1. 小型株専用・統合分析エンジン (Trend Follow & Breakout対応)
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
        df['ma25'] = df['close'].rolling(window=25).mean()
        # 乖離率（トレンド判定用）
        df['dev25'] = (df['close'] - df['ma25']) / df['ma25'] * 100
        
        # ボリンジャーバンド（ブレイクアウト判定用）
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['std20'] = df['close'].rolling(window=20).std()
        df['bb_p1'] = df['ma20'] + df['std20'] # +1σ
        df['bb_p2'] = df['ma20'] + (df['std20'] * 2) # +2σ
        
        # MACD
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['sig'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['sig']
        
        # RSI
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
        
        # 出来高急増（モメンタム判定用）
        df['vol_ma25'] = df['volume'].rolling(25).mean().replace(0, np.nan)
        df['vol_ratio'] = (df['volume'] / df['vol_ma25']).fillna(0)
        
        df['is_bullish'] = df['close'] > df['open']
        
        # 直近高値（ブレイクアウト判定用）
        df['high_20'] = df['high'].rolling(window=20).max().shift(1)

        if benchmark_df is not None and not benchmark_df.empty:
            benchmark_df.columns = [str(c).lower() for c in benchmark_df.columns]
            df = df.merge(benchmark_df[['date', 'close']], on='date', how='left', suffixes=('', '_bm'))
            df['close_bm'] = df['close_bm'].ffill()
            df['rs_21'] = (df['close'].pct_change(21) - df['close_bm'].pct_change(21)) * 100
        else:
            df['rs_21'] = 0.0

        return df

    @staticmethod
    def evaluate_entry(row_dict: Dict[str, Any], n_chg: float, vix: float) -> Tuple[bool, float, bool]:
        if not isinstance(row_dict, dict): raise TypeError("row_dict must be a dictionary")
        
        # VIXが極端に高い（市場パニック）時は手を出さない
        if vix >= 35.0:
            return False, 0.0, False
            
        curr_c = SmallCapStrategyAnalyzer._to_float(row_dict.get('close', 0.0))
        rsi_val = SmallCapStrategyAnalyzer._to_float(row_dict.get('rsi', 50.0), 50.0)
        vol_ratio = SmallCapStrategyAnalyzer._to_float(row_dict.get('vol_ratio', 1.0), 1.0)
        is_bullish = bool(row_dict.get('is_bullish', False))
        d25 = SmallCapStrategyAnalyzer._to_float(row_dict.get('dev25', 0.0))
        bb_p1 = SmallCapStrategyAnalyzer._to_float(row_dict.get('bb_p1', float('inf')))
        high_20 = SmallCapStrategyAnalyzer._to_float(row_dict.get('high_20', float('inf')))

        score = 0.0
        
        # 【大改修：完全順張り（ブレイクアウト＆モメンタム）ロジック】
        # 1. 25日線乖離率がプラス（上昇トレンドにあること）
        # 2. ボリンジャーバンド+1σを上抜け（強いモメンタムの発生）
        # 3. 出来高が過去25日平均の2倍以上（大口資金の流入）
        # 4. 陽線で引けていること（買い意欲の強さ）
        if d25 > 0.0 and curr_c > bb_p1 and vol_ratio >= 2.0 and is_bullish:
            score += 50.0
            
            # さらに直近20日高値をブレイクしていれば加点（新高値ブレイク）
            if curr_c > high_20:
                score += 50.0
                
            # RSIが強気圏（50〜85）なら加点
            if 50.0 <= rsi_val <= 85.0:
                score += 20.0
                
        is_entry = (score >= 100.0) # 厳格なブレイクアウト条件を満たしたものだけエントリー
        return is_entry, float(score), (vix >= 25.0)

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
        self.max_positions: int = max_positions
        self.us_market = USMarketCache()
        
        self.stats = {
            'orders_placed': 0, 'orders_exec': 0,
            'time_stops': 0, 'hard_stops': 0, 'trailing_stops': 0,
            'climax_exits': 0, 'gap_down_cancels': 0 
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
                    
                    self.stats['orders_placed'] += 1
                    
                    # 【大改修：成行エントリー】
                    # シグナルが出た翌日の「始値」で確実に約定させる。
                    # ただし、前日終値から-3%以上の特大ギャップダウンで始まった場合は危険なので見送る
                    if open_p < (prev_close * 0.97):
                        self.stats['gap_down_cancels'] += 1
                        continue 
                        
                    exec_price = open_p * 1.002 # スリッページを考慮
                    alloc_cash = order['allocated_cash']
                    qty = alloc_cash // exec_price
                    
                    if qty > 0 and cash >= (qty * exec_price):
                        cash -= qty * exec_price
                        positions[ticker] = {
                            'qty': qty, 'entry_p': exec_price, 'high_p': exec_price, 
                            'days_held': 0
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
                dev25 = SmallCapStrategyAnalyzer._to_float(row.get('dev25', 0.0))
                vol_ratio = SmallCapStrategyAnalyzer._to_float(row.get('vol_ratio', 0.0))
                
                pos['days_held'] += 1
                pos['high_p'] = max(pos['high_p'], curr_c)
                exit_score = 0
                
                # 順張り用のハードストップ（エントリーからの絶対防衛ライン）
                hard_stop_price = pos['entry_p'] - (current_atr * 3.0)
                
                if curr_c <= hard_stop_price:
                    exit_score += 100
                    self.stats['hard_stops'] += 1
                
                # クライマックス売り（極端な過熱感での利確）
                if dev25 > 40.0 and rsi > 80.0 and vol_ratio >= 3.0 and exit_score == 0:
                    exit_score += 100
                    self.stats['climax_exits'] += 1

                # 【改修点】順張り用のトレイリングストップ（広く取る）
                # 強いトレンドに乗るため、小さなノイズで狩られないように 4.0 ATR まで幅を広げる
                trailing_stop_price = pos['high_p'] - (current_atr * 4.0)
                if curr_c <= trailing_stop_price and exit_score == 0:
                    exit_score += 100
                    self.stats['trailing_stops'] += 1
                
                # タイムストップ（勢いが持続しなかった場合）
                # ブレイクアウトは短期間で結果が出るはずなので、15日で含み益が+3%未満なら切る
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
                    
                    is_entry, score, is_high_risk = SmallCapStrategyAnalyzer.evaluate_entry(row, n_chg, vix)
                    if is_entry:
                        candidates.append((score, ticker))
                
                candidates.sort(key=lambda x: x[0], reverse=True)
                
                is_high_risk = vix >= 25.0
                max_daily_new_orders = 2 if is_high_risk else self.max_positions
                allowed_slots_today = min(open_slots, max_daily_new_orders)
                
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
    dummy_row_err = {'rsi': np.nan, 'dev25': 'invalid', 'vol_ratio': None}
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
        print(f" 📊 SMALL CAP SIMULATION RESULTS (BREAKOUT STRATEGY)")
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
        print(f"     ┗ 特大ギャップダウン回避: {st['gap_down_cancels']} 回")
        print(f" [2] タイムストップ(15日)撤退: {st['time_stops']} 回")
        print(f" [3] ハードストップ(-3.0ATR): {st['hard_stops']} 回")
        print(f" [4] トレイリングストップ(4.0ATR): {st['trailing_stops']} 回")
        print(f" [5] クライマックス売り(過熱極致): {st['climax_exits']} 回")
        print(f"==================================================", flush=True)
        
    except Exception as e:
        print(f"[FATAL] {e}")
