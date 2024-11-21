import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from controller import predict_bitcoin_prices, today_open_price, predict_next_thirty_days, get_formated_data

today_price = today_open_price()
df = predict_bitcoin_prices(today_price)
df = df.applymap(lambda x: f"{x:.2f}" if isinstance(x, (float, np.float64)) else x)

# Estilo CSS para aumentar a fonte
st.markdown(
    """
    <style>
    .dataframe tbody tr td {
        font-size: 16px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Exibindo tabela
st.title("Previsão do Bitcoin para Hoje")
st.dataframe(df)

#=================================================================
st.subheader("Preveja com um valor de abertura especifico")

today_price = st.number_input("Digite o valor de abertura do BTC:", min_value=1, step=1)
today_price = [[today_price]]

df = predict_bitcoin_prices(today_price)
df = df.applymap(lambda x: f"{x:.2f}" if isinstance(x, (float, np.float64)) else x)
df = df.rename(columns={"Abertura(Hoje)": "Abertura"})

st.markdown(
    """
    <style>
    .dataframe tbody tr td {
        font-size: 16px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.dataframe(df)

#=================================================================

# Simulando os dados (substitua por seus dados reais)
df_BTC_daily = pd.read_csv("archive/BTC-Daily.csv")

scaler = MinMaxScaler(feature_range=(0, 1))
closed_date_df = get_formated_data(df_BTC_daily, scaler)
lst_output = predict_next_thirty_days(df_BTC_daily, scaler)

# Processando os dados
lstmdf = closed_date_df.tolist()
lstmdf.extend((np.array(lst_output).reshape(-1, 1)).tolist())
lstmdf = scaler.inverse_transform(lstmdf).reshape(1, -1).tolist()[0]

historical_data = lstmdf[:-30]
forecast_data = lstmdf[-30:]

# Criando o gráfico com Plotly
fig = go.Figure()

# Dados históricos
fig.add_trace(go.Scatter(
    x=list(range(len(historical_data))),
    y=historical_data,
    mode='lines',
    name='Dados Históricos',
    line=dict(color='blue')
))

# Previsão
fig.add_trace(go.Scatter(
    x=list(range(len(historical_data), len(lstmdf))),
    y=forecast_data,
    mode='lines',
    name='Previsão (30 dias)',
    line=dict(color='orange')
))

# Atualizando layout
fig.update_layout(
    title_text="Gráfico com a Previsão",
    plot_bgcolor="white",
    font=dict(size=15, color="black"),
    legend_title_text="Legenda",
    xaxis=dict(showgrid=False, title="Timestamp"),
    yaxis=dict(showgrid=False, title="Stock"),
)

# Exibindo no Streamlit
st.title("Gráfico com Dados Históricos e Previsão Para os Proximos 30 Dias")
st.plotly_chart(fig)

