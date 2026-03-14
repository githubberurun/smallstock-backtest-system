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
PRICES_ENDPOINT: Final[str] = "/equities/bars/daily"
FINS_ENDPOINT: Final[str] = "/fins/summary"
INFO_ENDPOINT: Final[str] = "/listed/info"

class JQuantsV2Fetcher:
    """J-Quants API v2準拠のキャッシュ対応・小型株データ取得クラス"""
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
        """売買代金上位銘柄の中から、小型株（500億未満）を効率的に抽出"""
        print("[INFO] Step 1: Fetching master listed info to filter candidates...", flush=True)
        try:
            info_resp = requests.get(f"{BASE_URL}{INFO_ENDPOINT}", headers=self.headers, timeout=30)
            if info_resp.status_code != 200:
                print(f"[ERROR] Failed to fetch listed info: {info_resp.text}")
                return []
            info_df = pd.DataFrame(info_resp.json().get("data", []))
        except Exception as e:
            print(f"[ERROR] Exception in listed info fetch: {e}")
            return []

        # 1. 候補の絞り込み (プライム市場の大型株やREITなどはスキャンから除外して高速化)
        small_cap_segments = ["Growth", "Standard", "グロース", "スタンダード"]
        candidates_info = info_df[
            (info_df["MarketCodeName"].str.contains("|".join(small_cap_segments), na=True)) &
            (~info_df["SectorName"].str.contains("ETF|ETN|REIT", na=False))
        ]
        candidate_codes = set(candidates_info["Code"].tolist())
        print(f"[INFO] Candidates filtered: {len(candidate_codes)} tickers from Growth/Standard segments.", flush=True)

        print("[INFO] Step 2: Fetching daily quotes for all tickers...", flush=True)
        target_date = datetime.now().date()
        daily_quotes = []
        for _ in range(10):
            params = {"date": target_date.strftime("%Y%m%d")}
            response = requests.get(f"{BASE_URL}{PRICES_ENDPOINT}", headers=self.headers, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json().get("data", [])
                if len(data) > 500:
                    daily_quotes = data
                    break
            target_date -= timedelta(days=1)
            time.sleep(1)

        if not daily_quotes:
            print("[ERROR] Could not fetch recent market data.")
            return []

        # 2. 売買代金でソート
        df_quotes = pd.DataFrame(daily_quotes)
        df_quotes["Va_n"] = pd.to_numeric(df_quotes.get("Va", 0), errors="coerce")
        df_quotes["C_n"] = pd.to_numeric(df_quotes.get("C", 0), errors="coerce")
        df_quotes = df_quotes.sort_values("Va_n", ascending=False)

        target_tickers: List[str] = []
        scan_limit = 1000
        processed_count = 0
        
        print(f"[INFO] Step 3: Verifying Market Cap for top {scan_limit} turnover candidates...", flush=True)
        for _, row in df_quotes.iterrows():
            code = str(row.get("Code", ""))
            if code not in candidate_codes:
                continue
            
            processed_count += 1
            if processed_count > scan_limit:
                break

            # ★ キャッシュ機能: すでにファイルが存在する銘柄は時価総額チェックAPIをスキップ
            if os.path.exists(f"Colog_github/{code[:4]}.parquet"):
                target_tickers.append(code[:4])
                print(f"  [{len(target_tickers)}/{limit}] Cached Small Cap Loaded: {code[:4]}", flush=True)
                if len(target_tickers) >= limit:
                    break
                continue

            close_p = float(row["C_n"])
            if close_p <= 0: 
                continue

            # 財務情報を取得して時価総額を確認
            try:
                f_resp = requests.get(f"{BASE_URL}{FINS_ENDPOINT}", headers=self.headers, params={"code": code}, timeout=10)
                if f_resp.status_code == 429:
                    print("[WARN] Rate limit hit. Sleeping 10s...", flush=True)
                    time.sleep(10)
                    f_resp = requests.get(f"{BASE_URL}{FINS_ENDPOINT}", headers=self.headers, params={"code": code}, timeout=10)

                if f_resp.status_code == 200:
                    f_data = f_resp.json().get("data", [])
                    if f_data:
                        shares = float(f_data[-1].get("NumberOfIssuedAndOutstandingSharesAtTheEndOfPeriod", 0))
                        mkt_cap = close_p * shares
                        if 0 < mkt_cap < max_market_cap:
                            target_tickers.append(code[:4])
                            print(f"  [{len(target_tickers)}/{limit}] Found: {code[:4]} | Cap: {mkt_cap/1e8:.1f}億 | Va: {row['Va_n']/1e6:.1f}M", flush=True)
                            if len(target_tickers) >= limit:
                                break
            except Exception as e:
                pass
            
            time.sleep(0.3)

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
            quotes = res_json.get("data", res_json.get("daily_quotes", []))
            all_data.extend(quotes)

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
            'AdjClose': 'close', 'AdjC': 'close', 'AdjustmentClose': 'close',
            'C': 'close_raw', 'Close': 'close_raw',
            'AdjHigh': 'high', 'AdjH': 'high', 'AdjustmentHigh': 'high',
            'H': 'high_raw', 'High': 'high_raw',
            'AdjLow': 'low', 'AdjL': 'low', 'AdjustmentLow': 'low',
            'L': 'low_raw', 'Low': 'low_raw',
            'AdjOpen': 'open', 'AdjO': 'open', 'AdjustmentOpen': 'open',
            'O': 'open_raw', 'Open': 'open_raw',
            'AdjVolume': 'volume', 'AdjVo': 'volume', 'AdjustmentVolume': 'volume',
            'Vo': 'volume_raw', 'Volume': 'volume_raw',
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
        'Date': ['2026-01-01'], 'AdjC': [150.5], 'AdjH': [155.0], 
        'AdjL': [149.0], 'AdjO': [150.0], 'AdjVo': [5000], 'Va': [752500]
    })
    cleaned_df = dummy_fetcher._clean(df_mock)
    assert 'close' in cleaned_df.columns, "AdjC should be mapped to 'close'"
    assert 'turnover' in cleaned_df.columns, "Va should be mapped to 'turnover'"
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
    
    target_tickers = fetcher.get_top_small_cap_tickers(limit=300, max_market_cap=50_000_000_000.0)
    if "13060" not in target_tickers and "1306" not in target_tickers:
        target_tickers.append("13060") # TOPIX ETF (ベンチマーク用)
        
    print(f"[INFO] Starting data fetch for {len(target_tickers)} tickers...", flush=True)
    
    for i, target_ticker in enumerate(target_tickers):
        path = f"Colog_github/{target_ticker}.parquet"
        
        # ★ キャッシュ機能: すでにファイルが存在する場合は取得APIをスキップ
        if os.path.exists(path):
            print(f"[{i+1}/{len(target_tickers)}] Fetching {target_ticker}... CACHED (SKIP)", flush=True)
            continue
            
        print(f"[{i+1}/{len(target_tickers)}] Fetching {target_ticker}...", end=" ", flush=True)
        fetched_data = fetcher.fetch(target_ticker)
        if not fetched_data.empty:
            fetched_data.to_parquet(path, index=False)
            print(f"OK ({len(fetched_data)} rows)", flush=True)
        else:
            print("FAILED", flush=True)
