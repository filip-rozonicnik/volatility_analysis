import yfinance as yf
import pandas as pd
from pathlib import Path


#gledamo futurse in gledali bomo NG- natural gas in CL-crude oil
#funkcija bo vracala pandas dataframe
def download_commodity_data(
    ticker: str,
    start_date: str,
    end_date: str | None = None,
    interval: str = "1d",
    save_path: str | None = None
):
    data = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=False,
        progress=False
        )
    
    if data.empty:
        raise ValueError(f'ni podatkov za ticker:{ticker}')
    
    #ce je slucajno posredovalo pandas MutiIndex stolpce
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path, index=False)

    return data
    