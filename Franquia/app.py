import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.title("Previsão Inicial de Custo")

dados = pd.read_csv("3.Franquia/slr12.csv", sep=";")

X = dados[['FrqAnual']] # 2 colchetes: modelo espera um dataframe
y = dados['CusInic'] # serie do pandas

modelo = LinearRegression().fit(X,y)

#layout com colunas
#col1, col2 = st.columns(2)
# with col1:
#     st.header("Dados")
#     st.table(dados.head(10))

# with col2:
#     st.header("Gráfico de Dispersão")
#     fig, ax = plt.subplots()
#     ax.scatter(X,y, color = 'blue')
#     ax.plot(X, modelo.predict(X), color = 'red')
#     st.pyplot(fig)

st.header("Dados")
st.table(dados.head(10))
st.header("Gráfico de Dispersão")
fig, ax = plt.subplots()
ax.scatter(X,y, color = 'blue')
ax.plot(X, modelo.predict(X), color = 'red')
st.pyplot(fig)

st.header("Valor Anual da Franquia:")
novo_valor = st.number_input("Insira Novo Valor", min_value=1.0, max_value=999999.0, value=1500.0,step=100.0)
processar = st.button("Processar")

if processar:
    dados_novo_valor = pd.DataFrame([novo_valor], columns=['FrqAnual'])
    prev = modelo.predict(dados_novo_valor)
    st.header(f"Previsão de Custo Inicial R$:{prev[0]:.2f}")