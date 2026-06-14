import numpy as np
import pandas as pd


#prejema in vraca pandas DataFrame
def clean_data(data: pd.DataFrame):
    data = data.copy()

    #imejmo neko standardno obliko
    data.columns = [col.strip().replace(' ', '_') for col in data.columns]

    #preverimo da je vse kar rabimo
    required = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

    for col in required:
        if col not in data.columns:
            raise ValueError(f'manjka {col}')
        
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date')

    #podvojene damo stran
    data = data.drop_duplicates(subset='Date', keep='last')
    #samo kar rabimo
    data = data[["Date", "Open", "High", "Low", "Close", "Volume"]]
    data.dropna()

    #rabimo pozitivne vrednosti
    data = data[
            (data['Open'] > 0)
            &(data['Close'] > 0)
            &(data['Low'] > 0)
            &(data['High'] > 0)
        ]

    data = data.reset_index(drop=True)

    return data

#vrne nam torej pandas DataFrame object, notri imamo
#po stolpcih date, open, close, low, high in volume 


#dodamo return featurse s katerimi bomo volatilnost racunali
#funckcije spet sprejema in vraca pandas DataFrame
def add_returns(data: pd.DataFrame):
    data = data.copy()

    data['simple_return'] = data['Close'].pct_change()
    data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
    data['abs_log_return'] = abs(data['log_return'])
    data['squared_log_return'] = data['log_return'] ** 2

    data.dropna()
    data = data.reset_index(drop=True)

    return data

#rolling volatilnost pove kaksna je bila volatilnost v nekem prejsnem obdobju
#racunamo standardni odklon

#spet sprejme in vraca pandas DataFrame
def rolling_volatility(data: pd.DataFrame):
    data = data.copy()

    data['vol_5d'] = data['log_return'].rolling(window=5).std()
    data['vol_10d'] = data['log_return'].rolling(window=10).std()
    data['vol_20d'] = data['log_return'].rolling(window=20).std()
    data['vol_60d'] = data['log_return'].rolling(window=60).std()


    #to hocemo izracunati, zato je ta projekt
    data['future_5d'] = (
        data['log_return'].rolling(window=5).std().shift(-5)
        )
    
    data.dropna()
    data.reset_index(drop=True)

    return data