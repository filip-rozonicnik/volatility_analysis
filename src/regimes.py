#tu bo koda v kateri naredimo razrede za volatilnost iz columna future_5d
#razredi bodo low < .33, medium <.66 in high, kjer je zgornja meja 1 az volatilnost
#klasifikacije niso hudo znanstvene

import pandas as pd

#vrne tuple = (float, float)
def fit_regimes(train_data: pd.DataFrame,
                target_col: str = 'future_5d'):
    
    q33 = train_data[target_col].quantile(0.33)
    q66 = train_data[target_col].quantile(0.66)

    return q33, q66

def assign_regime(
        volatility: float,
        q33: float,
        q66: float):
    
    if volatility <= q33:
        return 'LOW'
    elif volatility <= q66:
        return 'MEDIUM'
    else:
        return 'HIGH'
    


#vrne pandas DataFrame
def add_volatility_regimes(
        data: pd.DataFrame,
        q33,
        q66,
        target_col: str='future_5d'):
    
    data = data.copy()

    data['future_regime'] = data[target_col].apply(
        lambda x: assign_regime(x, q33, q66)
        )
    
    return data
    


