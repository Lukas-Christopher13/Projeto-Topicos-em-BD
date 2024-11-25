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
    time_stpe = 15
    model = joblib.load("models/LSTM_bitcoin_predict.pkl")
    test_data = get_data(df, scaler)

    # Pegando os últimos `time_step` dados para o input inicial
    x_input = test_data[len(test_data) - time_stpe:].reshape(1, -1)

    #converte para lista para podermos adicinar novos valores a medida que o loop avança
    temp_imput = list(x_input)
    temp_imput = temp_imput[0].tolist()

    #Aramazena os valores preveistos
    lst_output = []
    n_steps = time_stpe 
    i = 0
    pred_days = 30

    while(i < pred_days):
        if(len(temp_imput) > time_stpe):
            x_input = np.array(temp_imput[1:])
            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape((1, n_steps, 1))

            yhat = model.predict(x_input, verbose=0)

            temp_imput.extend(yhat[0].tolist())
            temp_imput = temp_imput[1:]
            
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_imput.extend(yhat[0].tolist())

            lst_output.extend(yhat.tolist())
            i=i+1

    return lst_output

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


def creat_yfinance_file():
    ticker = "BTC-USD"
    btc_data = yf.Ticker(ticker).history(period="max")

    btc_data = btc_data.reset_index()

    btc_data["unix"] = btc_data["Date"].astype(int) // 10**9  
    btc_data["date"] = btc_data["Date"].dt.strftime("%Y-%m-%d %H:%M:%S")  
    btc_data["symbol"] = "BTC/USD" 
    btc_data["Volume BTC"] = btc_data["Volume"] / btc_data["Close"]  
    btc_data["Volume USD"] = btc_data["Volume"] 

    btc_data = btc_data[["unix", "date", "symbol", "Open", "High", "Low", "Close", "Volume BTC", "Volume USD"]]
    btc_data.columns = ["unix", "date", "symbol", "open", "high", "low", "close", "Volume BTC", "Volume USD"]

    btc_data.to_csv("archive/yfinance_BTC.csv", index=False)
