import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# -----------------------------------
# Configuração da página
# -----------------------------------

st.set_page_config(page_title="Análise Estatística - Iris Dataset",
layout="wide")


# -----------------------------------
# Função da média (implementação manual)
# -----------------------------------

def calcular_media(lista):
    soma = 0
    for valor in lista:
        soma += valor
    return soma / len(lista)


# -----------------------------------
# Carregar dataset
# -----------------------------------

@st.cache_data
def load_data():

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

    colunas = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species"
    ]

    df = pd.read_csv(url, names=colunas)

    # remove possíveis linhas vazias
    df = df.dropna()

    return df


df = load_data()


# -----------------------------------
# Interface
# -----------------------------------

st.title("📊 Analisador Estatístico - Iris Dataset")
st.markdown("---")

# -----------------------------------
# Sidebar (filtros)
# -----------------------------------

st.sidebar.header("Filtros")

atributo = st.sidebar.selectbox(
    "Escolha o atributo",
    [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]
)

especies = ["Todas"] + sorted(df["species"].unique())

especie_selecionada = st.sidebar.selectbox(
    "Escolha a espécie",
    especies
)

# -----------------------------------
# Filtrar dados
# -----------------------------------

if especie_selecionada == "Todas":
    dados_filtrados = df[atributo]
else:
    dados_filtrados = df[df["species"] == especie_selecionada][atributo]

dados = dados_filtrados.tolist()

n = len(dados)

st.write(f"Número de observações: {n}")

# -----------------------------------
# FASE 1 - Estatísticas simples
# -----------------------------------

st.header("📍 Estatísticas de Dados Avulsos")

col1, col2, col3, col4, col5 = st.columns(5)

# Média

media = calcular_media(dados)

# Mediana

mediana = dados_filtrados.median()

# Moda (tratamento seguro)

moda_series = dados_filtrados.mode()

if moda_series.empty:
    moda = np.nan
else:
    moda = moda_series.iloc[0]

# Variância (amostral)

variancia = dados_filtrados.var()

# Desvio padrão

desvio = dados_filtrados.std()

with col1:
    st.metric("Média", f"{media:.3f}")

with col2:
    st.metric("Mediana", f"{mediana:.3f}")

with col3:
    st.metric("Moda", f"{moda:.3f}")

with col4:
    st.metric("Variância", f"{variancia:.3f}")

with col5:
    st.metric("Desvio Padrão", f"{desvio:.3f}")


# -----------------------------------
# Histograma
# -----------------------------------

st.subheader("Distribuição dos Dados")

fig, ax = plt.subplots()

ax.hist(dados_filtrados, bins=10)

ax.set_xlabel(atributo)
ax.set_ylabel("Frequência")

st.pyplot(fig)

# -----------------------------------
# FASE 2 - Distribuição de frequência
# -----------------------------------

st.markdown("---")
st.header("📈 Distribuição de Frequência")

# Regra de Sturges

k = math.ceil(1 + 3.322 * math.log10(n))

# Amplitudes

valor_min = dados_filtrados.min()
valor_max = dados_filtrados.max()

amplitude_total = valor_max - valor_min
amplitude_classe = amplitude_total / k

st.write(f"Número de classes (Sturges): {k}")
st.write(f"Amplitude total: {amplitude_total:.3f}")
st.write(f"Amplitude da classe: {amplitude_classe:.3f}")

# -----------------------------------
# Construção das classes
# -----------------------------------

classes = []

limite_inferior = valor_min

for i in range(k):

    limite_superior = limite_inferior + amplitude_classe

    classes.append(
        (
            round(limite_inferior, 3),
            round(limite_superior, 3)
        )
    )

    limite_inferior = limite_superior


# -----------------------------------
# Construção da tabela
# -----------------------------------

tabela = []

freq_acumulada = 0

for i, (inf, sup) in enumerate(classes):

    if i == k - 1:
        fi = dados_filtrados[
            (dados_filtrados >= inf) &
            (dados_filtrados <= sup)
        ].count()

    else:
        fi = dados_filtrados[
            (dados_filtrados >= inf) &
            (dados_filtrados < sup)
        ].count()

    freq_acumulada += fi

    fr = (fi / n) * 100

    xi = (inf + sup) / 2

    tabela.append({

        "Classe": f"{inf} ├ {sup}",
        "Ponto Médio (xi)": xi,
        "f_i": fi,
        "F_ac": freq_acumulada,
        "f_r %": round(fr, 2)

    })


df_freq = pd.DataFrame(tabela)

st.table(df_freq)

# -----------------------------------
# FASE 3 - Estatísticas agrupadas
# -----------------------------------

st.header("🗂️ Estatísticas de Dados Agrupados")

# Média agrupada

soma = 0

for linha in tabela:
    soma += linha["f_i"] * linha["Ponto Médio (xi)"]

media_agrupada = soma / n

# Variância agrupada

soma_var = 0

for linha in tabela:
    soma_var += linha["f_i"] * (
        (linha["Ponto Médio (xi)"] - media_agrupada) ** 2
    )

variancia_agrupada = soma_var / (n - 1)

desvio_agrupado = math.sqrt(variancia_agrupada)

# -----------------------------------
# Mediana agrupada
# -----------------------------------

classe_mediana = None

for linha in tabela:
    if linha["F_ac"] >= n / 2:
        classe_mediana = linha
        break

L = float(classe_mediana["Classe"].split(" ├ ")[0])

F_anterior = 0

for linha in tabela:
    if linha == classe_mediana:
        break
    F_anterior = linha["F_ac"]

f_mediana = classe_mediana["f_i"]

h = amplitude_classe

mediana_agrupada = L + (
    ((n / 2) - F_anterior) / f_mediana
) * h


# -----------------------------------
# Exibir resultados
# -----------------------------------

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Média Agrupada",
         f"{media_agrupada:.3f}")

with c2:
    st.metric("Mediana Agrupada", 
      f"{mediana_agrupada:.3f}")

with c3:
    st.metric("Variância Agrupada", 
        
    f"{variancia_agrupada:.3f}")

with c4:
    st.metric("Desvio Padrão Agrupado", 
         f"{desvio_agrupado:.3f}")

st.success("Cálculos concluídos com sucesso.")