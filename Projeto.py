import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# -----------------------------------
# Configuração da página
# -----------------------------------
st.set_page_config(page_title="Análise Estatística - Iris Dataset", layout="wide")

# -----------------------------------
# Função da média (implementação manual)
# -----------------------------------
def calcular_media(lista):
    if not lista: return 0
    return sum(lista) / len(lista)

# -----------------------------------
# Carregar dataset
# -----------------------------------
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    # Nomes originais do CSV para mapeamento correto
    colunas_originais = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    df = pd.read_csv(url, names=colunas_originais)
    
    # Mapeamento para nomes amigáveis em português
    mapeamento = {
        "sepal_length": "comprimento_sepala",
        "sepal_width": "largura_sepala",
        "petal_length": "comprimento_petala",
        "petal_width": "largura_petala"
    }
    df = df.rename(columns=mapeamento)
    return df.dropna()

df = load_data()

# -----------------------------------
# Interface e Sidebar
# -----------------------------------
st.title("📊 Analisador Estatístico - Iris Dataset")
st.markdown("---")

st.sidebar.header("Filtros")

# Corrigido: As chaves do selectbox agora batem com as colunas do DF
atributo = st.sidebar.selectbox(
    "Escolha o atributo",
    ["comprimento_sepala", "largura_sepala", "comprimento_petala", "largura_petala"]
)

especies = ["Todas"] + sorted(df["species"].unique().tolist())
especie_selecionada = st.sidebar.selectbox("Escolha a espécie", especies)

# -----------------------------------
# Filtrar dados
# -----------------------------------
if especie_selecionada == "Todas":
    dados_filtrados = df[atributo]
else:
    dados_filtrados = df[df["species"] == especie_selecionada][atributo]

dados = dados_filtrados.tolist()
n = len(dados)

# -----------------------------------
# FASE 1 - Estatísticas simples
# -----------------------------------
st.header("📍 Estatísticas de Dados Avulsos")

col1, col2, col3, col4, col5 = st.columns(5)

media = calcular_media(dados)
mediana = dados_filtrados.median()
moda_series = dados_filtrados.mode()
moda = moda_series.iloc[0] if not moda_series.empty else np.nan
variancia = dados_filtrados.var()
desvio = dados_filtrados.std()

col1.metric("Média", f"{media:.3f}")
col2.metric("Mediana", f"{mediana:.3f}")
col3.metric("Moda", f"{moda:.3f}")
col4.metric("Variância", f"{variancia:.3f}")
col5.metric("Desvio Padrão", f"{desvio:.3f}")

# -----------------------------------
# TABELA DE DADOS NÃO AGRUPADOS
# -----------------------------------
st.subheader("📋 Tabela de Dados Não Agrupados")

tabela_nao_agrupados = []
soma_x2 = sum([x**2 for x in dados])
soma_d2 = sum([(x - media)**2 for x in dados])

# Criar DataFrame para exibição
df_lista = []
for i, xi in enumerate(dados, 1):
    df_lista.append({
        "i": i,
        "x_i": round(xi, 3),
        "x_i²": round(xi**2, 3),
        "d_i (x_i - x̄)": round(xi - media, 3),
        "d_i²": round((xi - media)**2, 3)
    })

df_visualizacao = pd.DataFrame(df_lista)

# Corrigido: .append() foi removido do Pandas. Usamos pd.concat ou lógica de exibição.
if len(df_visualizacao) > 20:
    st.dataframe(df_visualizacao.head(20), use_container_width=True)
    st.caption(f"*(Mostrando primeiras 20 observações de {n})*")
else:
    st.dataframe(df_visualizacao, use_container_width=True)

st.info(f"**Totais:** Σx = {sum(dados):.3f} | Σx² = {soma_x2:.3f} | Σd² = {soma_d2:.3f}")

# -----------------------------------
# Histograma
# -----------------------------------
st.subheader("Distribuição dos Dados")
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(dados, bins=10, color='skyblue', edgecolor='black')
ax.set_xlabel(atributo)
ax.set_ylabel("Frequência")
st.pyplot(fig)

# -----------------------------------
# FASE 2 - Distribuição de frequência
# -----------------------------------
st.markdown("---")
st.header("📈 Distribuição de Frequência")

k = math.ceil(1 + 3.322 * math.log10(n))
valor_min, valor_max = min(dados), max(dados)
amplitude_total = valor_max - valor_min
amplitude_classe = amplitude_total / k

col_a, col_b, col_c = st.columns(3)
col_a.write(f"**Classes (Sturges):** {k}")
col_b.write(f"**Amplitude Total:** {amplitude_total:.3f}")
col_c.write(f"**Amplitude da Classe:** {amplitude_classe:.3f}")

# Construção das classes e tabela
tabela_freq = []
freq_acumulada = 0
limite_inf = valor_min

for i in range(k):
    limite_sup = limite_inf + amplitude_classe
    # Lógica de intervalo fechado no último
    if i == k - 1:
        fi = len([x for x in dados if limite_inf <= x <= limite_sup])
    else:
        fi = len([x for x in dados if limite_inf <= x < limite_sup])
    
    freq_acumulada += fi
    xi = (limite_inf + limite_sup) / 2
    
    tabela_freq.append({
        "Classe": f"{limite_inf:.3f} ├ {limite_sup:.3f}",
        "Ponto Médio (xi)": round(xi, 3),
        "f_i": fi,
        "F_ac": freq_acumulada,
        "f_r %": round((fi / n) * 100, 2)
    })
    limite_inf = limite_sup

df_freq = pd.DataFrame(tabela_freq)
st.table(df_freq)

# -----------------------------------
# FASE 3 - Estatísticas agrupadas
# -----------------------------------
st.header("🗂️ Estatísticas de Dados Agrupados")

# Média agrupada
media_agrupada = sum([row["f_i"] * row["Ponto Médio (xi)"] for row in tabela_freq]) / n

# Variância agrupada
soma_var_agrup = sum([row["f_i"] * ((row["Ponto Médio (xi)"] - media_agrupada)**2) for row in tabela_freq])
variancia_agrupada = soma_var_agrup / (n - 1)
desvio_agrupado = math.sqrt(variancia_agrupada)

# Mediana agrupada
classe_mediana = next(row for row in tabela_freq if row["F_ac"] >= n / 2)
L_mediana = float(classe_mediana["Classe"].split(" ├ ")[0])
idx_mediana = tabela_freq.index(classe_mediana)
F_anterior = tabela_freq[idx_mediana - 1]["F_ac"] if idx_mediana > 0 else 0
f_mediana = classe_mediana["f_i"]

mediana_agrupada = L_mediana + (((n / 2) - F_anterior) / f_mediana) * amplitude_classe

c1, c2, c3, c4 = st.columns(4)
c1.metric("Média Agrupada", f"{media_agrupada:.3f}")
c2.metric("Mediana Agrupada", f"{mediana_agrupada:.3f}")
c3.metric("Variância Agrupada", f"{variancia_agrupada:.3f}")
c4.metric("Desvio Padrão Agrupado", f"{desvio_agrupado:.3f}")

st.success("Cálculos concluídos com sucesso! ✅")
