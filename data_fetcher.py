import os
import requests
import pandas as pd
import time
from typing import Dict, List, Optional, Final, Any
from datetime import datetime, timedelta

# ==========================================
# 2025-2026年 最新公式ドキュメント準拠
# Pandas: https://pandas.pydata.org/docs/
# Requests: https://requests.readthedocs.io/en/latest/
# J-Quants API Reference: https://jpx.gitbook.io/j-quants-ja/api-reference
# ==========================================

BASE_URL: Final[str] = "https://api.jquants.com/v2"
PRICES_ENDPOINT: Final[str] = "/prices/daily_quotes"
FINS_ENDPOINT: Final[str] = "/fins/statements"

class JQuantsV2Fetcher:
    """J-Quants API v2準拠のデータ取得クラス（小型株対応）"""
    def __init__(self, api_key: str) -> None:
        if not isinstance(api_key, str):
            raise TypeError("API key must be a string")
        self.api_key: str = api_key.strip()
        self.headers: Dict[str, str] = {"x-api-key": self.api_key}

    def get_safe_start_date(self) -> str:
        """プラン制限(10年)の境界値を考慮した開始日を算出"""
        safe_date = datetime.now() - timedelta(days=365 * 10 - 1)
        return safe_date.strftime("%Y-%m-%d")

    def get_top_small_cap_tickers(self, limit: int = 300, max_market_cap: float = 50_000_000_000.0) -> List[str]:
        """売買代金上位銘柄の中から、時価総額が基準未満の小型株を抽出する"""
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        if not isinstance(max_market_cap, float) or max_market_cap <= 0:
            raise ValueError("max_market_cap must be a positive float")
            
        print("[INFO] Fetching daily quotes to sort by TurnoverValue...")
        target_date = datetime.now().date()
        daily_quotes = []
        
        # 1. 直近の有効な取引日の全銘柄株価を取得
        for _ in range(10):
            params = {"date": target_date.strftime("%Y%m%d")}
            try:
                response = requests.get(f"{BASE_URL}{PRICES_ENDPOINT}", headers=self.headers, params=params, timeout=30)
                if response.status_code == 200:
                    data = response.json().get("daily_quotes", [])
                    if len(data) > 500:
                        daily_quotes = data
                        break
            except Exception as e:
                print(f"[WARN] Failed to fetch daily data for {target_date}: {e}")
            target_date -= timedelta(days=1)
            time.sleep(1)
            
        if not daily_quotes:
            print("[ERROR] Could not fetch recent market data.")
            return []

        # 2. 売買代金で降順ソート
        df = pd.DataFrame(daily_quotes)
        df['TurnoverValue'] = pd.to_numeric(df.get('TurnoverValue', 0), errors='coerce')
        df['Close'] = pd.to_numeric(df.get('Close', 0), errors='coerce')
        df = df.sort_values('TurnoverValue', ascending=False)
        
        target_tickers: List[str] = []
        print(f"[INFO] Scanning for small caps (Market Cap < {max_market_cap/1_000_000_000:.1f}B JPY)...")
        
        # 3. 上位から順に時価総額を計算し、小型株を抽出
        for _, row in df.iterrows():
            code = str(row['Code'])
            close_price = float(row['Close'])
            if close_price <= 0: 
                continue
                
            f_params = {"code": code}
            try:
                f_resp = requests.get(f"{BASE_URL}{FINS_ENDPOINT}", headers=self.headers, params=f_params, timeout=10)
                if f_resp.status_code == 200:
                    f_data = f_resp.json().get("statements", [])
                    if f_data:
                        # 最新の財務諸表から期末発行済株式数を取得
                        latest_statement = f_data[-1]
                        shares_str = latest_statement.get("NumberOfIssuedAndOutstandingSharesAtTheEndOfPeriod", "0")
                        shares = float(shares_str) if shares_str and str(shares_str).strip() != "" else 0.0
                        
                        market_cap = close_price * shares
                        
                        if 0 < market_cap < max_market_cap:
                            target_tickers.append(code[:4])
                            print(f"  [+] Small Cap Found: {code[:4]} (Market Cap: {market_cap/1_000_000_000:.1f}B JPY)")
                            if len(target_tickers) >= limit:
                                break
            except Exception as e:
                print(f"[WARN] Failed to fetch financials for {code}: {e}")
                
            time.sleep(0.2) # APIのレートリミット保護
            
        return target_tickers

    def fetch(self, ticker: str) -> pd.DataFrame:
        if not isinstance(ticker, str):
            raise TypeError("ticker must be a string")
            
        code: str = f"{ticker}0" if len(ticker) == 4 else ticker
        start_date: str = self.get_safe_start_date()
        all_data: List[Dict[str, Any]] = []
        pagination_key: Optional[str] = None

        while True:
            params: Dict[str, Any] = {"code": code, "from": start_date}
            if pagination_key:
                params["pagination_key"] = pagination_key

            try:
                response = requests.get(f"{BASE_URL}{PRICES_ENDPOINT}", headers=self.headers, params=params, timeout=30)
            except requests.exceptions.RequestException as e:
                print(f"[ERROR] Network error during fetch: {e}")
                return pd.DataFrame()

            if response.status_code != 200:
                print(f"[ERROR] API {response.status_code}: {response.text}")
                return pd.DataFrame()

            res_json = response.json()
            all_data.extend(res_json.get("daily_quotes", []))

            pagination_key = res_json.get("pagination_key")
            if not pagination_key:
                break
            time.sleep(0.3)

        return self._clean(pd.DataFrame(all_data))

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if df.empty: 
            return df
            
        col_map = {
            'Date': 'date', 
            'AdjustmentClose': 'close', 'AdjC': 'close', 'Close': 'close_raw',
            'AdjustmentHigh': 'high', 'AdjH': 'high', 'High': 'high_raw',
            'AdjustmentLow': 'low', 'AdjL': 'low', 'Low': 'low_raw',
            'AdjustmentOpen': 'open', 'AdjO': 'open', 'Open': 'open_raw',
            'AdjustmentVolume': 'volume', 'AdjVo': 'volume', 'Volume': 'volume_raw',
            'TurnoverValue': 'turnover', 'Va': 'turnover'
        }
        df = df.rename(columns=col_map)
        
        if 'close' not in df.columns and 'close_raw' in df.columns:
            df = df.rename(columns={
                'close_raw': 'close', 
                'high_raw': 'high', 
                'low_raw': 'low', 
                'open_raw': 'open', 
                'volume_raw': 'volume'
            })

        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'turnover']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'turnover' not in df.columns or df['turnover'].isnull().all():
            df['turnover'] = df['close'] * df['volume']
                
        if 'date' in df.columns:
            df = df.dropna(subset=['close']).sort_values("date").reset_index(drop=True)
            
        return df

# ==========================================
# 空データ・異常値に対する堅牢性証明テスト
# ==========================================
def test_integrity() -> None:
    print("[TEST] Running integrity tests for data_fetcher.py...")
    dummy_fetcher = JQuantsV2Fetcher("dummy_key")
    
    df_mock = pd.DataFrame({
        'Date': ['2026-01-01'], 'AdjustmentClose': [150.5], 'AdjustmentHigh': [155.0], 
        'AdjustmentLow': [149.0], 'AdjustmentOpen': [150.0], 'AdjustmentVolume': [5000], 'TurnoverValue': [752500]
    })
    cleaned_df = dummy_fetcher._clean(df_mock)
    assert 'close' in cleaned_df.columns, "AdjustmentClose should be mapped to 'close'"
    assert 'turnover' in cleaned_df.columns, "TurnoverValue should be mapped to 'turnover'"
    assert cleaned_df['close'].iloc[0] == 150.5, "Value matching failed for close"

    df_empty = pd.DataFrame()
    cleaned_empty = dummy_fetcher._clean(df_empty)
    assert cleaned_empty.empty, "Empty DataFrame should return empty DataFrame"
    
    try:
        dummy_fetcher.fetch(1234) # type: ignore
        assert False, "fetch() should raise TypeError for non-string input"
    except TypeError:
        pass

    print("[TEST] All integrity tests passed.")

if __name__ == "__main__":
    test_integrity()
    
    key = os.getenv("JQUANTS_API_KEY")
    if not key:
        print("[WARN] JQUANTS_API_KEY is not set. Exiting fetcher execution.")
        exit(0)
        
    fetcher = JQuantsV2Fetcher(key)
    os.makedirs("Colog_github", exist_ok=True)
    
    # 時価総額500億円未満、売買代金上位300銘柄を取得
    target_tickers = fetcher.get_top_small_cap_tickers(limit=300, max_market_cap=50_000_000_000.0)
    if "13060" not in target_tickers:
        target_tickers.append("13060") # TOPIX ETF (ベンチマーク用)
        
    print(f"[INFO] Starting data fetch for {len(target_tickers)} tickers...")
    
    for i, target_ticker in enumerate(target_tickers):
        print(f"[{i+1}/{len(target_tickers)}] Fetching {target_ticker}...", end=" ")
        fetched_data = fetcher.fetch(target_ticker)
        if not fetched_data.empty:
            fetched_data.to_parquet(f"Colog_github/{target_ticker}.parquet", index=False)
            print(f"OK ({len(fetched_data)} rows)")
        else:
            print("FAILED")
