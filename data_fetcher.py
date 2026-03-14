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
# J-Quants API Reference: https://jpx-jquants.com/ja/spec/eq-master
# ==========================================

BASE_URL: Final[str] = "https://api.jquants.com/v2"
PRICES_ENDPOINT: Final[str] = "/equities/bars/daily"
FINS_ENDPOINT: Final[str] = "/fins/summary"
INFO_ENDPOINT: Final[str] = "/equities/master" 

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
        candidate_codes = None
        
        try:
            info_resp = requests.get(f"{BASE_URL}{INFO_ENDPOINT}", headers=self.headers, timeout=30)
            if info_resp.status_code == 200:
                info_data = info_resp.json().get("data", [])
                if info_data:
                    info_df = pd.DataFrame(info_data)
                    
                    # 最優先で「名前（Nm/Name）」が入るカラムを指定検索する
                    market_col = next((c for c in ["MarketCodeName", "MktNm", "Section", "Segment"] if c in info_df.columns), None)
                    sector_col = next((c for c in ["SectorName", "S33Nm", "S17Nm"] if c in info_df.columns), None)
                    code_col = next((c for c in ["Code", "code"] if c in info_df.columns), "Code")
                    
                    if market_col and code_col in info_df.columns:
                        sample_mkt = info_df[market_col].dropna().unique()[:5]
                        print(f"[DEBUG] Using '{market_col}' for Market Filter. Sample values: {sample_mkt}", flush=True)
                        
                        small_cap_segments = ["Growth", "Standard", "グロース", "スタンダード", "G", "S"]
                        candidates_info = info_df[info_df[market_col].astype(str).str.contains("|".join(small_cap_segments), na=False, case=False)]
                        
                        if sector_col:
                            sample_sec = info_df[sector_col].dropna().unique()[:3]
                            print(f"[DEBUG] Using '{sector_col}' for Sector Filter. Sample values: {sample_sec}", flush=True)
                            candidates_info = candidates_info[~candidates_info[sector_col].astype(str).str.contains("ETF|ETN|REIT", na=False, case=False)]
                            
                        extracted_codes = set(candidates_info[code_col].astype(str).tolist())
                        
                        # 安全装置: 0件になった場合はフィルター失敗とみなし、全件スキャンを許可する
                        if len(extracted_codes) == 0:
                            print("[WARN] Filtering resulted in 0 tickers. Ignoring filter to prevent total failure.", flush=True)
                        else:
                            candidate_codes = extracted_codes
                            print(f"[INFO] Candidates filtered: {len(candidate_codes)} tickers from Growth/Standard segments.", flush=True)
                    else:
                        print(f"[WARN] Valid Market column not found. Available cols: {info_df.columns.tolist()}", flush=True)
            else:
                print(f"[WARN] Master info fetch failed ({info_resp.status_code}). Proceeding without pre-filtering.", flush=True)
        except Exception as e:
            print(f"[WARN] Exception in master info fetch: {e}. Proceeding without pre-filtering.", flush=True)

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
        
        va_col = "Va" if "Va" in df_quotes.columns else "TurnoverValue" if "TurnoverValue" in df_quotes.columns else None
        c_col = "C" if "C" in df_quotes.columns else "Close" if "Close" in df_quotes.columns else None
        code_col_q = "Code" if "Code" in df_quotes.columns else "code" if "code" in df_quotes.columns else None

        if va_col is None or c_col is None or code_col_q is None:
            print(f"[ERROR] Required columns missing in daily quotes. Available: {df_quotes.columns.tolist()}", flush=True)
            return []

        df_quotes["Va_n"] = pd.to_numeric(df_quotes[va_col], errors="coerce")
        df_quotes["C_n"] = pd.to_numeric(df_quotes[c_col], errors="coerce")
        df_quotes = df_quotes.sort_values("Va_n", ascending=False)

        target_tickers: List[str] = []
        scan_limit = 1000
        processed_count = 0
        
        print(f"[INFO] Step 3: Verifying Market Cap for top {scan_limit} turnover candidates...", flush=True)
        for _, row in df_quotes.iterrows():
            code = str(row[code_col_q])
            if not code or code == "nan":
                continue
            
            # 事前フィルターが有効な場合のみチェック
            if candidate_codes is not None:
                if code not in candidate_codes and code[:4] not in candidate_codes:
                    continue
            
            processed_count += 1
            if processed_count > scan_limit:
                break

            ticker_base = code[:4]
            # キャッシュ機能
            if os.path.exists(f"Colog_github/{ticker_base}.parquet"):
                if ticker_base not in target_tickers:
                    target_tickers.append(ticker_base)
                    print(f"  [{len(target_tickers)}/{limit}] Cached Small Cap Loaded: {ticker_base}", flush=True)
                if len(target_tickers) >= limit:
                    break
                continue

            close_p = float(row["C_n"])
            if close_p <= 0 or pd.isna(close_p): 
                continue

            # 財務情報を取得して時価総額を確認 (確実に5桁コードにする)
            query_code = f"{ticker_base}0"
            try:
                f_resp = requests.get(f"{BASE_URL}{FINS_ENDPOINT}", headers=self.headers, params={"code": query_code}, timeout=10)
                if f_resp.status_code == 429:
                    print("[WARN] Rate limit hit. Sleeping 10s...", flush=True)
                    time.sleep(10)
                    f_resp = requests.get(f"{BASE_URL}{FINS_ENDPOINT}", headers=self.headers, params={"code": query_code}, timeout=10)

                if f_resp.status_code == 200:
                    f_data = f_resp.json().get("data", [])
                    if f_data:
                        latest = f_data[-1]
                        shares = 0.0
                        
                        # V2キー名揺れ対応（EPS等を除外し、株式数を動的に探索）
                        for k, v in latest.items():
                            if v is None or str(v).strip() == "": continue
                            k_lower = str(k).lower()
                            if "share" in k_lower or "outstanding" in k_lower or "発行済" in str(k):
                                if "per" not in k_lower and "eps" not in k_lower and "dividend" not in k_lower:
                                    try:
                                        val = float(v)
                                        if val > 100000: # 発行済株式数は通常10万以上
                                            shares = val
                                            break
                                    except ValueError: 
                                        pass
                        
                        if shares > 0:
                            mkt_cap = close_p * shares
                            if 0 < mkt_cap < max_market_cap:
                                target_tickers.append(ticker_base)
                                print(f"  [{len(target_tickers)}/{limit}] Found: {ticker_base} | Cap: {mkt_cap/1e8:.1f}億 | Va: {row['Va_n']/1e6:.1f}M", flush=True)
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
        
        # キャッシュ機能: すでにファイルが存在する場合は取得APIをスキップ
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
