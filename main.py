from src.data import download_commodity_data
from src.data_processing import (
    clean_data,
    add_returns,
    rolling_volatility
)
from src.regimes import (
    fit_regimes,
    add_volatility_regimes
)
from src.prediction_model import (
    data_split,
    train_random_forest,
    evaluate_model,
    predict_latest,
    print_feature_importance,
    tune_random_forest
)


def main(ticker: str,
        path: str, 
        start_date = "2010-01-01",
        end_date = None):

    raw_path = f"data/raw/{path}_raw.csv"
    processed_path = f"data/processed/{path}_processed.csv"

    raw_data = download_commodity_data(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        interval="1d",
        save_path=raw_path
    )

    data_cleaned = clean_data(raw_data)
    data_with_returns = add_returns(data_cleaned)
    final_data = rolling_volatility(data_with_returns)
    
   
    train_data, test_data = data_split(final_data, train_size= 0.8)
    q33, q66 = fit_regimes(train_data)
    
    print(f"LOW/MEDIUM meja: {q33:.5f}")
    print(f"MEDIUM/HIGH meja: {q66:.5f}")

    train_data = add_volatility_regimes(train_data, q33, q66)
    test_data = add_volatility_regimes(test_data, q33, q66)

    full_data = add_volatility_regimes(final_data, q33, q66)
    full_data.to_csv(processed_path, index= False)

    model, best_params, best_score = tune_random_forest(train_data)

    print("\nUporabljeni najboljši parametri v končnem modelu:")
    for key, value in best_params.items():
        print(f"{key}: {value}")

    print(f"\nNajboljši validacijski macro F1 score: {best_score:.4f}")

    evaluate_model(model, test_data)

    predicted_regime, probabilities = predict_latest(model, full_data)

    latest_date = full_data.iloc[-1]['Date']

    print(f"\nZadnji datum v podatkih: {latest_date}")
    print(f"Napovedan režim naslednjega tedna: {predicted_regime}")

    print("\nVerjetnosti:")
    print(probabilities)

    print(f"\nProcesirani podatki shranjeni v: {processed_path}")


#kar se tice NG=F tickerja
#ko pozenemo vidimo da v supportu da je v testnem setu low volatility
#bilo zelo malo, zato tudi ocitno model nikoli ne napoveduje
#ali to dela zelo redko low volatility, zato je tudi precision za
#low volaitlity 0
#iz visokega recalla high vidimo da zelo rad cilja na high volatility
#macroavg je slab ker ignorira low
#zakljucek je da model ni prevec dober

#poglejmo se en druge ticker CL=F, torej crude oil futures
#testni podatki so veliko bolj uravnotezeni,
#iz precisiona sledi da malo preveckrat rece low ko nebi smel
#spet najbolje zaznava high volatility, malo je boljsi kot naiven baselie
#torej ce bi vedno predictali tisto kar je po testnih podatkih najpogosteje
#bi dobili rezultat slabsi kot ce bi zaupali temu modelu

#importance pove, da bi se mogoce splacalo odstraniti log_return in volume,
#mogoce pa bi se splacalo spremeniti parametre modela pri treningu

#originalno so bili parametri nastavljeni na         
# n_estimators=300,
#       max_depth=5,
#       min_samples_leaf=20,
#       random_state=42,

#splacalo bi se te parametre spreminjati, 
#class_weight pustimo balanced, da ne bo ignoriralo 
#razredov (low, medium, high) ce se redkeje pojavijo
#v ta namen je dodana funkcija tune_random_forest ki bo ciljala
#na cim vecji f1_macro avg, ker to pomeni da vse tri razrede dobro napoveduje, ce je 
#macro avg dober

#zal ker predolgo traja ne moremo preizkusiti vec kombinacij,
#torej best params niso zares best params ker jih ne preverimo dovolj
#tudi z best params ni model prevec dober


if __name__ == "__main__":
    main('NG=F', 'natural_gas')