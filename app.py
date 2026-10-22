import streamlit as st
import spacy
import sqlite3
import pandas as pd
from datetime import datetime
from transformers import pipeline

# 1. Carregar os modelos em cache (evita lentidão no Streamlit)
@st.cache_resource
def load_nlp_models():
    nlp = spacy.load("pt_core_news_sm")
    # Carrega um modelo de rede neural real treinado para sentimentos (PT/ES/EN)
    classificador = pipeline("sentiment-analysis", model="pysentimiento/robertuito-sentiment-analysis")
    return nlp, classificador

nlp, classificador = load_nlp_models()

# 2. Configuração do Banco de Dados SQLite3
def init_db():
    conn = sqlite3.connect("banco_sentimentos_ia.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS historico_analise (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data_hora TEXT,
            texto_input TEXT,
            sentimento TEXT,
            compound REAL
        )
    ''')
    conn.commit()
    return conn

def salvar_no_banco(texto, sentimento, compound):
    conn = init_db()
    cursor = conn.cursor()
    data_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO historico_analise (data_hora, texto_input, sentimento, compound)
        VALUES (?, ?, ?, ?)
    ''', (data_hora, texto, sentimento, compound))
    conn.commit()
    conn.close()

def ler_banco():
    conn = init_db()
    df = pd.read_sql_query("SELECT * FROM historico_analise ORDER BY id DESC", conn)
    conn.close()
    return df

# 3. Interface Streamlit
st.title("🧠 Analisador de Sentimentos (IA Nativa PT-BR)")
st.markdown("Fatiamento com **spaCy** e análise de sentimento com **Hugging Face Transformers**.")

texto_usuario = st.text_area("Digite o texto em português para análise:", height=150)

if st.button("Analisar Texto"):
    if texto_usuario.strip() == "":
        st.warning("Por favor, digite um texto para analisar.")
    else:
        # A. Fatiamento e remoção de Stop Words (spaCy)
        doc = nlp(texto_usuario)
        tokens_limpos = [token.text for token in doc if not token.is_stop and not token.is_punct]
        
        # B. Inferência direta no modelo de IA
        resultado = classificador(texto_usuario)[0]
        label_ia = resultado['label']   # Retorna 'POS', 'NEG' ou 'NEU'
        confianca = resultado['score']  # Retorna a probabilidade (0.0 a 1.0)
        
        # C. Conversão para a lógica do "Compound"
        if label_ia == 'POS':
            sentimento = "Positivo 😃"
            compound = confianca       # Valor positivo
        elif label_ia == 'NEG':
            sentimento = "Negativo 😔"
            compound = -confianca      # Valor negativo
        else:
            sentimento = "Neutro 😐"
            compound = 0.0             # Neutro
            
        salvar_no_banco(texto_usuario, sentimento, compound)
        
        st.subheader("Resultados da Análise")
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"**Sentimento:** {sentimento}")
        with col2:
            st.info(f"**Compound (Confiança):** {compound:.4f}")
            
        st.write("**Tokens em PT-BR (Texto fatiado sem stop words):**")
        st.code(tokens_limpos)

st.divider()

# 4. Histórico
st.subheader("🗄️ Histórico salvo no SQLite3")
historico_df = ler_banco()
if not historico_df.empty:
    st.dataframe(historico_df, use_container_width=True)
else:
    st.write("Nenhum histórico encontrado ainda.")










# import streamlit as st
# import spacy
# import sqlite3
# import pandas as pd
# from datetime import datetime
# from LeIA import SentimentIntensityAnalyzer  # Substitui o NLTK pelo LeIA (PT-BR)

# # 1. Carregar o modelo NLP e o Analisador de Sentimentos em cache
# @st.cache_resource
# def load_nlp_models():
#     # Modelo do spaCy para português
#     nlp = spacy.load("pt_core_news_sm")
#     # Analisador focado em Português Brasileiro
#     sia = SentimentIntensityAnalyzer()
#     return nlp, sia

# nlp, sia = load_nlp_models()

# # 2. Configuração do Banco de Dados SQLite3
# def init_db():
#     conn = sqlite3.connect("banco_sentimentos_ptbr.db")
#     cursor = conn.cursor()
#     cursor.execute('''
#         CREATE TABLE IF NOT EXISTS historico_analise (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             data_hora TEXT,
#             texto_input TEXT,
#             sentimento TEXT,
#             compound REAL
#         )
#     ''')
#     conn.commit()
#     return conn

# def salvar_no_banco(texto, sentimento, compound):
#     conn = init_db()
#     cursor = conn.cursor()
#     data_hora_atual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
#     cursor.execute('''
#         INSERT INTO historico_analise (data_hora, texto_input, sentimento, compound)
#         VALUES (?, ?, ?, ?)
#     ''', (data_hora_atual, texto, sentimento, compound))
    
#     conn.commit()
#     conn.close()

# def ler_banco():
#     conn = init_db()
#     df = pd.read_sql_query("SELECT * FROM historico_analise ORDER BY id DESC", conn)
#     conn.close()
#     return df

# # 3. Interface da Aplicação Streamlit
# st.title("🧠 Analisador de Sentimentos (PT-BR)")
# st.markdown("Fatiamento com **spaCy** e análise de sentimento nativa com **LeIA**.")

# texto_usuario = st.text_area("Digite o texto em português para análise:", height=150)

# if st.button("Analisar Texto"):
#     if texto_usuario.strip() == "":
#         st.warning("Por favor, digite um texto para analisar.")
#     else:
#         # A. Fatiamento e Remoção de Stop Words/Pontuação com spaCy
#         doc = nlp(texto_usuario)
#         tokens_limpos = [token.text for token in doc if not token.is_stop and not token.is_punct]
        
#         # B. Análise de Sentimento e Compound nativa em PT-BR
#         scores = sia.polarity_scores(texto_usuario)
#         compound = scores['compound']
        
#         if compound >= 0.05:
#             sentimento = "Positivo 😃"
#         elif compound <= -0.05:
#             sentimento = "Negativo 😔"
#         else:
#             sentimento = "Neutro 😐"
            
#         # C. Salvar no Banco de Dados
#         salvar_no_banco(texto_usuario, sentimento, compound)
        
#         # D. Exibir Resultados
#         st.subheader("Resultados da Análise")
        
#         col1, col2 = st.columns(2)
#         with col1:
#             st.success(f"**Sentimento:** {sentimento}")
#         with col2:
#             st.info(f"**Compound:** {compound}")
            
#         st.write("**Tokens (Texto fatiado sem stop words):**")
#         st.code(tokens_limpos)

# st.divider()

# # 4. Visualização do Banco de Dados
# st.subheader("🗄️ Histórico salvo no SQLite3")
# historico_df = ler_banco()

# if not historico_df.empty:
#     st.dataframe(historico_df, use_container_width=True)
# else:
#     st.write("Nenhum histórico encontrado ainda.")












# import streamlit as st
# import spacy
# import sqlite3
# import pandas as pd
# from datetime import datetime
# import nltk
# from nltk.sentiment.vader import SentimentIntensityAnalyzer

# # 1. Configuração inicial e download de dependências do NLTK
# try:
#     nltk.data.find('sentiment/vader_lexicon.zip')
# except LookupError:
#     nltk.download('vader_lexicon')

# # 2. Carregar o modelo NLP em cache para não recarregar a cada interação no Streamlit
# @st.cache_resource
# def load_nlp_models():
#     # Certifique-se de ter rodado no terminal: python -m spacy download pt_core_news_sm
#     nlp = spacy.load("pt_core_news_sm")
#     sia = SentimentIntensityAnalyzer()
#     return nlp, sia

# nlp, sia = load_nlp_models()

# # 3. Configuração do Banco de Dados SQLite3
# def init_db():
#     conn = sqlite3.connect("banco_sentimentos.db")
#     cursor = conn.cursor()
#     cursor.execute('''
#         CREATE TABLE IF NOT EXISTS historico_analise (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             data_hora TEXT,
#             texto_input TEXT,
#             sentimento TEXT,
#             compound REAL
#         )
#     ''')
#     conn.commit()
#     return conn

# def salvar_no_banco(texto, sentimento, compound):
#     conn = init_db()
#     cursor = conn.cursor()
#     data_hora_atual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
#     cursor.execute('''
#         INSERT INTO historico_analise (data_hora, texto_input, sentimento, compound)
#         VALUES (?, ?, ?, ?)
#     ''', (data_hora_atual, texto, sentimento, compound))
    
#     conn.commit()
#     conn.close()

# def ler_banco():
#     conn = init_db()
#     df = pd.read_sql_query("SELECT * FROM historico_analise ORDER BY id DESC", conn)
#     conn.close()
#     return df

# # 4. Interface da Aplicação Streamlit
# st.title("🧠 Analisador de Sentimentos com spaCy")
# st.markdown("Fatiamento de texto, remoção de stop words e análise de sentimento.")

# texto_usuario = st.text_area("Digite o texto para análise:", height=150)

# if st.button("Analisar Texto"):
#     if texto_usuario.strip() == "":
#         st.warning("Por favor, digite um texto para analisar.")
#     else:
#         # A. Fatiamento (Tokenização) e Remoção de Palavras Desnecessárias (Stop Words/Pontuação)
#         doc = nlp(texto_usuario)
        
#         # Filtra tokens que não são stop words e não são pontuação
#         tokens_limpos = [token.text for token in doc if not token.is_stop and not token.is_punct]
#         texto_limpo = " ".join(tokens_limpos)
        
#         # B. Análise de Sentimento e Compound
#         # Nota: O VADER é otimizado para inglês. Em aplicações PT-BR estritas,
#         # costuma-se traduzir o texto antes ou usar a biblioteca LeIA (fork do VADER para PT).
#         scores = sia.polarity_scores(texto_usuario)
#         compound = scores['compound']
        
#         if compound >= 0.05:
#             sentimento = "Positivo 😃"
#         elif compound <= -0.05:
#             sentimento = "Negativo 😔"
#         else:
#             sentimento = "Neutro 😐"
            
#         # C. Salvar no Banco de Dados
#         salvar_no_banco(texto_usuario, sentimento, compound)
        
#         # D. Exibir Resultados
#         st.subheader("Resultados da Análise")
        
#         col1, col2 = st.columns(2)
#         with col1:
#             st.info(f"**Sentimento:** {sentimento}")
#         with col2:
#             st.info(f"**Compound:** {compound}")
            
#         st.write("**Tokens (Texto limpo sem stop words):**")
#         st.code(tokens_limpos)

# st.divider()

# # 5. Visualização do Banco de Dados
# st.subheader("🗄️ Histórico salvo no Banco de Dados (SQLite3)")
# historico_df = ler_banco()

# if not historico_df.empty:
#     st.dataframe(historico_df, use_container_width=True)
# else:
#     st.write("Nenhum histórico encontrado ainda.")