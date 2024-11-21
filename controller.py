import joblib
import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

def predict_bitcoin_prices(open_price): 
    high_model = joblib.load("models/high_model.pkl")
    low_model = joblib.load("models/low_model.pkl")
    close_model = joblib.load("models/close_model.pkl")

    high_pred = high_model.predict(open_price)
    low_pred = low_model.predict(open_price)
    close_pred = close_model.predict(open_price)

    data = {
        "Abertura(Hoje)": [open_price],
        "Maior Valor": [high_pred[0]],
        "Menor Valor": [low_pred[0]],
        "Fechamento": [close_pred[0]],
    }

    return pd.DataFrame(data)

def today_open_price():
    bitcoin = yf.Ticker("BTC-USD")
    historical_data = bitcoin.history(period="1d")

    price_open = historical_data['Open'].iloc[0] if not historical_data.empty else None
    return [[price_open]]

def predict_next_thirty_days(df, scaler):
    time_step = 15
    model = joblib.load("models/LSTM_bitcoin_predict.pkl")
    test_data = get_data(df, scaler)

    # Pegando os últimos `time_step` dados para o input inicial
    x_input = test_data[-time_step:].reshape(1, -1)

    # Preparando a lista temporária
    temp_input = x_input.flatten().tolist()

    lst_output = []
    pred_days = 30

    for _ in range(pred_days):
        if len(temp_input) > time_step:
            x_input = np.array(temp_input[-time_step:]).reshape((1, time_step, 1))
        else:
            x_input = np.array(temp_input).reshape((1, time_step, 1))
        
        # Realizando a predição
        yhat = model.predict(x_input, verbose=0)
        temp_input.append(yhat[0][0])
        lst_output.append(yhat[0][0])

    print(f"Output of predicted next days: {len(lst_output)}")
    print(lst_output)

def get_data(df, scaler):
    df = get_formated_data(df, scaler)
    # Seleciona apenas os dados de teste (40% finais dos dados)
    test_data = df[int(len(df) * 0.60):]
    return test_data

def get_formated_data(df, scaler):   
    # Filtra colunas e datas
    df = df[["date", "close"]]
    df = df[df["date"] > "2020-12-31"]
    df.drop(columns=["date"], inplace=True)
    
    # Normaliza os dados
    df = scaler.fit_transform(np.array(df).reshape(-1, 1))
    return df
