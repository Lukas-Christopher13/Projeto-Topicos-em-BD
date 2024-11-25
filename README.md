# Bitcoin Price Prediction

Este projeto utiliza aprendizado de máquina para prever preços de Bitcoin com base em dados históricos.

Acesse o projeto ao vivo em: [Bitcoin Price Prediction - Hugging Face](https://huggingface.co/spaces/LukasChristopher13/Bitcoin_Price_Predict_UEPB)

---

## Pré-requisitos

Antes de começar, certifique-se de ter os seguintes itens instalados:

- **Python 3.8.10** (ou configure com [pyenv](https://github.com/pyenv/pyenv))
- **pip** (gerenciador de pacotes do Python)
- **virtualenv** (opcional, mas recomendado)

---

## Dataset

O data set que utilizei inicialmente foi o: https://www.kaggle.com/datasets/prasoonkottarathil/btcinusd
Contudo os dados fornecidos pela biblioteca "yfinance" são mais atuais, então optei por usar esses dados.  

## Configuração do ambiente

Siga os passos abaixo para configurar o ambiente do projeto:

### 1. Clone o repositório

Clone este repositório para sua máquina local:

### 2. Criem um .venv

Use: 'python3 -m venv .venv'. Para criar um ambinete virtual 
Em seguida use: 'pip install -r requirements.txt' para instalar as dependencias

### 3. rode o streamlit

execute o comando: 'streamlit run app.py'


### OBS 1:

Por algum motivo tive problema na instalação das dependencias via
'pip install -r requirements.txt'. Caso isso ocorra instale manualmente com o seguinte comando:

pip install joblib
pip install nbformat
pip install pandas
pip install numpy
pip install seaborn
pip install yfinance
pip install plotly
pip install sklearn
pip install tensorflow
pip install matplotlib
pip install streamlit

### OBS 2: 

O arquivo 'Bitcoin_historical_data.ipynb' contem os passos necessários para criar os modelos!

