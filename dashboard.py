# dashboard.py (CORRIGIDO: Erros de HTML, Jinja2, int64 e use_container_width)
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import spacy
from collections import Counter
import emoji
import subprocess
import sys
import locale
import base64
import io
from datetime import datetime, timedelta, date
import sqlite3
import tempfile
import json
import glob
from jinja2 import Environment, FileSystemLoader
import re

# --- NOVOS IMPORTS PARA A IA DO GEMINI ---
import google.generativeai as genai
import time
import logging
from src.utils.config import CONFIG # Você precisa do CONFIG para os prompts

# --- Importa as funções de IA do seu outro arquivo ---
try:
    from src.relatorios.gerar_relatorio import gerar_resumo_com_gemini, gerar_resumo_executivo
except ImportError:
    st.error("ERRO CRÍTICO: Falha ao importar 'gerar_resumo_com_gemini' ou 'gerar_resumo_executivo' de 'src.relatorios.gerar_relatorio'.")
    # Define funções "falsas" para evitar que o app quebre
    def gerar_resumo_com_gemini(*args): 
        st.session_state['last_ai_log_stderr'] = "Erro: Falha ao importar 'gerar_resumo_com_gemini'."
        return "[ERRO DE IMPORTAÇÃO]"
    def gerar_resumo_executivo(*args): 
        st.session_state['last_ai_log_stderr'] = "Erro: Falha ao importar 'gerar_resumo_executivo'."
        return "[ERRO DE IMPORTAÇÃO]"
# --- FIM DOS NOVOS IMPORTS ---


# Configura o logging (útil para debug das funções importadas)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] [%(asctime)s] - %(message)s')


# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Dashboard de Análise", layout="wide")

# --- CSS INJETADO ---
st.markdown("""<style>
.summary-box { border-left: 6px solid #1e88e5; padding: 1.5rem; border-radius: 5px; margin-bottom: 2rem; min-height: 100px; }
.critical-alert { background-color: #ffebee; border: 2px solid #F44336; color: #c62828; font-weight: bold; padding: 1rem; border-radius: 5px; margin-bottom: 1rem; text-align: center; }
.stTabs [data-baseweb="tab-list"] { gap: 24px; }
div[data-testid="column"]:nth-child(2) div[data-testid="metric-value"] { color: #4CAF50; }
div[data-testid="column"]:nth-child(3) div[data-testid="metric-value"] { color: #9E9E9E; }
div[data-testid="column"]:nth-child(4) div[data-testid="metric-value"] { color: #F44336; }
</style>""", unsafe_allow_html=True)

st.title("📊 Dashboard de Análise de Mídias Sociais")

# --- CONSTANTES E CONFIGURAÇÕES ---
COLOR_MAP = {'Negativo': '#F44336', 'Neutro': '#9E9E9E', 'Positivo': '#4CAF50'}
HTML_OUTPUT_DIR = "relatorios_html"
DB_DIR = "dados"
PROCESSED_DATA_DIR = "relatorios_processados" # <- Fonte principal de dados
AI_SCRIPT_TIMEOUT = 180
CRITICAL_NEG_THRESHOLD = 25.0
STOP_WORDS_PT = [ "rapaz", "gente", "ruma", "coisa", "tudo", "nada", "disse", "mano", "cara", "vei", "tipo", "aí", "ne", "pra", "pro", "tá", "q", "vc", "vcs", "ja", "la", "ter", "ser", "ir", "fazer", "dizer", "querer", "ficar", "deixar", "dar", "assim", "então", "aqui", "agora", "hoje", "sempre", "muito", "pouco", "grande", "pequeno", "bom", "mau", "dia", "noite", "mês", "ano", "vez" ]

# Cria diretórios se não existirem
for dir_path in [DB_DIR, HTML_OUTPUT_DIR, PROCESSED_DATA_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# --- FUNÇÕES UTILITÁRIAS E DE CARREGAMENTO ---
@st.cache_resource
def load_spacy_model():
    try: model = spacy.load("pt_core_news_md"); print("Modelo Spacy (Medium) carregado."); return model
    except OSError:
        st.error("Modelo Spacy 'pt_core_news_md' não encontrado."); st.info("Tentando baixar...");
        try: from spacy.cli import download; download("pt_core_news_md"); st.success("Modelo baixado!"); model = spacy.load("pt_core_news_md"); st.rerun(); return model
        except Exception as e: st.error(f"Falha ao baixar/carregar: {e}"); return None

@st.cache_data(ttl=60)
def load_run_metadata():
    metadata_path = os.path.join(DB_DIR, "run_metadata.json") # Metadata ainda pode vir da pasta 'dados'
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f: return json.load(f)
        except Exception as e: print(f"Erro ao ler run_metadata.json: {e}"); return {}
    return {}

def _load_data_from_json_files(search_path: str) -> pd.DataFrame:
    json_files = glob.glob(search_path)
    if not json_files: st.warning(f"Nenhum JSON em '{search_path}'."); return pd.DataFrame()
    print(f"Encontrados {len(json_files)} JSONs.")
    df_list = []
    for f_path in json_files:
        try: 
            df_list.append(pd.read_json(f_path, orient='records', lines=False, encoding='utf-8'))
        except Exception as e: st.error(f"Erro ao ler '{os.path.basename(f_path)}': {e}")
    if not df_list: st.error("Falha ao carregar JSONs."); return pd.DataFrame()
    try: df_full = pd.concat(df_list, ignore_index=True); print(f"Total de {len(df_full)} linhas."); return df_full
    except Exception as e: st.error(f"Erro ao concatenar: {e}"); return pd.DataFrame()

def _preprocess_dataframe(df_full: pd.DataFrame) -> pd.DataFrame:
    if df_full.empty: return pd.DataFrame()
    if 'sentimento_final' in df_full.columns: df_full.rename(columns={'sentimento_final': 'sentimento'}, inplace=True)
    if 'genero_final' in df_full.columns: df_full.rename(columns={'genero_final': 'genero_previsto'}, inplace=True)
    default_values = { 'conteudo': "", 'data_hora': pd.NaT, 'sentimento': 'Neutro', 'genero_previsto': 'Desconhecido', 'fonte_coleta': 'N/A', 'run_id': 'N/A', 'emojis': [], 'texto_puro': "", 'tamanho_comentario': 0, 'parent_url': None }
    for col, default in default_values.items():
        if col not in df_full.columns: df_full[col] = default

    df_full['data_hora'] = pd.to_datetime(df_full['data_hora'], errors='coerce')
    
    for col in ['sentimento', 'genero_previsto', 'fonte_coleta', 'run_id']: df_full[col] = df_full[col].fillna(default_values[col])
    df_full['genero_previsto'] = df_full['genero_previsto'].replace(['indeterminado', 'unknown'], 'Desconhecido', regex=False)
    
    if 'emojis' not in df_full.columns or df_full['emojis'].apply(lambda x: not isinstance(x, list)).any(): df_full['emojis'] = df_full['conteudo'].apply(lambda t: [e['emoji'] for e in emoji.emoji_list(str(t))] if pd.notna(t) else [])
    if 'texto_puro' not in df_full.columns or df_full['texto_puro'].isnull().all(): df_full['texto_puro'] = df_full['conteudo'].apply(lambda t: emoji.replace_emoji(str(t), replace='') if pd.notna(t) else "")
    if 'tamanho_comentario' not in df_full.columns or (df_full['tamanho_comentario']==0).all(): df_full['tamanho_comentario'] = df_full['texto_puro'].str.len().fillna(0).astype(int)
    
    df_comments = df_full.copy()
    initial_count = len(df_comments)
    df_comments.dropna(subset=['data_hora'], inplace=True)
    dropped_count = initial_count - len(df_comments)
    
    if dropped_count > 0: print(f"Removidas {dropped_count} linhas com data inválida.")
    print(f"Retornando {len(df_comments)} válidos."); return df_comments

@st.cache_data(ttl=300)
def load_processed_data_for_profile(profile_name: str) -> pd.DataFrame:
    print(f"Carregando dados para: {profile_name}")
    search_path = os.path.join(PROCESSED_DATA_DIR, f"{profile_name}_*.json") 
    df_raw = _load_data_from_json_files(search_path)
    if df_raw.empty: return pd.DataFrame()
    return _preprocess_dataframe(df_raw)

@st.cache_data 
def clean_text_spacy(text: str) -> list:
    nlp_model = load_spacy_model() 
    if not nlp_model or not isinstance(text, str): return []
    doc = nlp_model(text.lower())
    return [ token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha and token.lemma_ not in STOP_WORDS_PT and len(token.lemma_) > 1 ]

def get_cleaned_words_for_freq(_texts_tuple: tuple) -> list:
    if not _texts_tuple: return []
    full_text = " ".join(_texts_tuple)
    if not full_text.strip(): return []
    else: return clean_text_spacy(full_text) 

def plot_metrics(df_period_B, df_period_A):
    st.subheader(f"Métricas Gerais")
    if df_period_B.empty and df_period_A.empty: st.warning("Sem dados nos períodos."); return
    b_total, b_pos, b_neu, b_neg, total_calc_b = 0, 0, 0, 0, 0
    if not df_period_B.empty and 'sentimento' in df_period_B.columns:
        b_total = len(df_period_B); counts_b = df_period_B['sentimento'].value_counts()
        b_pos, b_neu, b_neg = counts_b.get('Positivo', 0), counts_b.get('Neutro', 0), counts_b.get('Negativo', 0); total_calc_b = b_pos + b_neu + b_neg
    a_total, a_pos, a_neu, a_neg = 0, 0, 0, 0
    has_period_A = not df_period_A.empty and 'sentimento' in df_period_A.columns
    if has_period_A: a_total = len(df_period_A); counts_a = df_period_A['sentimento'].value_counts(); a_pos, a_neu, a_neg = counts_a.get('Positivo', 0), counts_a.get('Neutro', 0), counts_a.get('Negativo', 0)
    delta_total = b_total - a_total if has_period_A else None; delta_pos = b_pos - a_pos if has_period_A else None; delta_neu = b_neu - a_neu if has_period_A else None; delta_neg = b_neg - a_neg if has_period_A else None
    delta_strs = { "total": f"{delta_total:+}" if delta_total is not None else None, "pos": f"{delta_pos:+}" if delta_pos is not None else None, "neu": f"{delta_neu:+}" if delta_neu is not None else None, "neg": f"{delta_neg:+}" if delta_neg is not None else None }
    percents = { "pos": f"{round((b_pos*100)/total_calc_b if total_calc_b > 0 else 0, 1)}%", "neu": f"{round((b_neu*100)/total_calc_b if total_calc_b > 0 else 0, 1)}%", "neg": f"{round((b_neg*100)/total_calc_b if total_calc_b > 0 else 0, 1)}%" }
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total", b_total, delta=delta_strs["total"], delta_color="off")
    col2.metric(f"Positivos ({percents['pos']})", b_pos, delta=delta_strs["pos"], delta_color="off")
    col3.metric(f"Neutros ({percents['neu']})", b_neu, delta=delta_strs["neu"], delta_color="off")
    col4.metric(f"Negativos ({percents['neg']})", b_neg, delta=delta_strs["neg"], delta_color="off")
    st.caption("Variação (Δ) vs. Período Comparação.")

def _create_empty_fig(title: str, message: str = "Sem dados") -> go.Figure:
    fig = go.Figure(); fig.update_layout(title_text=title, annotations=[dict(text=message, showarrow=False)]); return fig
    
def get_fig_pie_chart(df):
    if df.empty or 'sentimento' not in df.columns or df['sentimento'].nunique() == 0: return _create_empty_fig("Distribuição de Sentimentos")
    contagem = df["sentimento"].value_counts().reset_index(); contagem.columns = ['sentimento', 'total']
    fig = px.pie(contagem, values='total', names='sentimento', title="Distribuição de Sentimentos", color='sentimento', color_discrete_map=COLOR_MAP); fig.update_layout(showlegend=True, legend_title_text='Sentimento'); return fig

def get_fig_gender_chart(df):
    if df.empty or 'genero_previsto' not in df.columns or 'sentimento' not in df.columns or df['genero_previsto'].nunique() == 0: return _create_empty_fig("Sentimento por Gênero")
    df_valid = df.dropna(subset=['genero_previsto', 'sentimento']);
    if df_valid.empty: return _create_empty_fig("Sentimento por Gênero", "Sem dados válidos")
    gender_sentiment = df_valid.groupby(['genero_previsto', 'sentimento']).size().reset_index(name='total')
    if gender_sentiment.empty: return _create_empty_fig("Sentimento por Gênero")
    fig = px.bar(gender_sentiment, x='genero_previsto', y='total', color='sentimento', title="Sentimento por Gênero", labels={'genero_previsto': 'Gênero', 'total': 'Total'}, barmode='group', color_discrete_map=COLOR_MAP); fig.update_layout(legend_title_text='Sentimento'); return fig

def get_word_frequency_fig(df): 
    if df.empty or 'texto_puro' not in df.columns: return _create_empty_fig("Top 20 Termos")
    textos = df["texto_puro"].astype(str).tolist();
    if not textos: return _create_empty_fig("Top 20 Termos", "Sem texto")
    palavras = get_cleaned_words_for_freq(tuple(textos)) 
    if not palavras: return _create_empty_fig("Top 20 Termos", "Sem palavras válidas")
    freq = pd.Series(palavras).value_counts().nlargest(20).sort_values(ascending=True)
    fig = px.bar(freq, x=freq.values, y=freq.index, orientation='h', title="Top 20 Termos Frequentes", labels={'x': 'Freq.', 'y': 'Termo'}); fig.update_layout(yaxis={'categoryorder':'total ascending'}); return fig

def get_fig_comment_length(df):
    if df.empty or 'tamanho_comentario' not in df.columns or 'sentimento' not in df.columns or df['tamanho_comentario'].isnull().all(): return _create_empty_fig("Tamanho dos Comentários")
    fig = px.box(df, x='sentimento', y='tamanho_comentario', color='sentimento', title="Tamanho (Texto) por Sentimento", labels={'sentimento': 'Sentimento', 'tamanho_comentario': 'Nº Caracteres'}, color_discrete_map=COLOR_MAP); fig.update_layout(showlegend=False); return fig

def get_fig_timeline(df):
    if df.empty or 'data_hora' not in df.columns or df['data_hora'].isnull().all(): return _create_empty_fig("Linha do Tempo")
    df_copy = df.copy(); df_copy['data_hora'] = pd.to_datetime(df_copy['data_hora'], errors='coerce'); df_copy.dropna(subset=['data_hora'], inplace=True)
    if df_copy.empty: return _create_empty_fig("Linha do Tempo", "Sem datas válidas")
    try:
        df_copy['data_hora'] = df_copy['data_hora'].dt.tz_convert(None) 
    except TypeError:
        pass
    df_copy['data'] = df_copy['data_hora'].dt.date
    timeline_counts = df_copy.groupby('data').size().sort_index()
    if timeline_counts.empty: return _create_empty_fig("Linha do Tempo")
    fig = px.line(timeline_counts, x=timeline_counts.index, y=timeline_counts.values, title="Linha do Tempo", labels={'index': 'Data', 'value': 'Quantidade'}, markers=True); return fig

def get_fig_top_emojis(df):
    if df.empty or 'emojis' not in df.columns: return _create_empty_fig("Top Emojis")
    df['emojis'] = df['emojis'].apply(lambda x: x if isinstance(x, list) else [])
    all_emojis = [e for sublist in df['emojis'] for e in sublist]
    if not all_emojis: return _create_empty_fig("Top Emojis", "Sem emojis")
    freq_emojis = pd.Series(all_emojis).value_counts().nlargest(15).sort_values(ascending=True)
    fig = px.bar(freq_emojis, x=freq_emojis.values, y=freq_emojis.index, orientation='h', title="Top 15 Emojis", labels={'x': 'Freq.', 'y': 'Emoji'}); fig.update_layout(yaxis={'categoryorder':'total ascending', 'tickfont':{'size':18}}); return fig


# --- <<< INÍCIO DA LÓGICA DE IA (GEMINI) >>> ---

@st.cache_data(ttl=900) # Cacheia o resumo por 15 minutos
def _run_gemini_logic_on_dataframe(df_filtrado: pd.DataFrame) -> str:
    """
    Nova função que replica a lógica de 'gerar_relatorio.py' mas usando um DataFrame.
    """
    print("Iniciando _run_gemini_logic_on_dataframe...")
    st.session_state['last_ai_log_stderr'] = "" # Limpa log antigo
    try:
        # 1. Pegar a API Key dos Secrets do Streamlit
        gemini_key = st.secrets["GEMINI_API_KEY"]
        if not gemini_key:
            logging.error("ERRO_API_CONFIG: 'GEMINI_API_KEY' não encontrada nos segredos (secrets) do Streamlit.")
            return "ERRO_API_CONFIG: 'GEMINI_API_KEY' não encontrada nos segredos (secrets) do Streamlit."
        
        genai.configure(api_key=gemini_key)
        modelo = genai.GenerativeModel(
            model_name=CONFIG['gemini']['model_name'],
            generation_config=CONFIG['gemini']['generation_config']
        )
        print("API Gemini configurada com sucesso.")
    
    except KeyError:
         logging.error("ERRO_API_CONFIG: 'GEMINI_API_KEY' não encontrada nos segredos (secrets) do Streamlit.")
         return "ERRO_API_CONFIG: 'GEMINI_API_KEY' não encontrada nos segredos (secrets) do Streamlit."
    except Exception as e:
         logging.error(f"ERRO_API_CONFIG: Falha ao configurar a API Gemini: {e}")
         return f"ERRO_API_CONFIG: Falha ao configurar a API Gemini: {e}"

    # 2. Replicar a lógica de 'carregar_conteudo_periodo'
    #    (mas usando o DataFrame em memória)
    if df_filtrado.empty:
        logging.warning("DataFrame vazio fornecido para IA.")
        return "Não há dados no período selecionado para gerar o resumo."
    
    df_analise = df_filtrado.copy()
    
    # Baseado no seu JSON de exemplo, os itens são posts/comentários e o 
    # texto está em 'texto_puro'. Vamos analisar tudo o que for passado.
    if 'texto_puro' not in df_analise.columns:
         logging.error("Erro: Coluna 'texto_puro' não encontrada no DataFrame.")
         return "Erro: Coluna 'texto_puro' não encontrada no DataFrame."
         
    # Agrupa por fonte_coleta e cria o dicionário, assim como o script original
    grupos_de_conteudo = df_analise.groupby('fonte_coleta')['texto_puro'].apply(list).to_dict()

    if not grupos_de_conteudo:
        logging.warning("Nenhum grupo de conteúdo encontrado após agrupar.")
        return "Não foram encontrados textos válidos para o período."

    # 3. Chamar a lógica de IA (exatamente como no gerar_relatorio.py)
    try:
        logging.info(f"Iniciando análise com Gemini para {len(grupos_de_conteudo)} fontes.")
        detailed_analysis_results = {
            nome: gerar_resumo_com_gemini(conteudos, nome, modelo)
            for nome, conteudos in grupos_de_conteudo.items()
        }
        resumos_validos = [res for res in detailed_analysis_results.values() if not res.startswith("[")]
        
        if not resumos_validos:
            logging.warning("Nenhum resumo detalhado válido foi gerado pela IA.")
            st.session_state['last_ai_log_stderr'] = "Todos os resumos detalhados falharam ou foram bloqueados."
            return "Não foi possível gerar resumos (provavelmente bloqueados pela API)."

        executive_summary = gerar_resumo_executivo(resumos_validos, modelo)
        logging.info("Resumo executivo gerado pela IA.")
        
        return executive_summary
        
    except Exception as e:
        logging.error(f"Erro inesperado durante a chamada da lógica Gemini: {e}")
        st.session_state['last_ai_log_stderr'] = f"Erro inesperado na IA: {e}"
        return f"Erro inesperado durante a chamada da IA: {e}"


# --- FUNÇÃO WRAPPER ATUALIZADA ---
def run_ai_summary_generation(df_filtrado: pd.DataFrame):
    """
    Wrapper para a nova função de IA do Gemini, que usa DataFrame.
    """
    st.info("Executando resumo IA (Gemini)...")
    st.session_state['last_ai_log_stderr'] = "" # Limpa log antigo
    st.session_state['last_ai_log_stdout'] = ""

    if df_filtrado.empty:
        st.warning("Não há dados no período selecionado para o resumo IA.")
        st.session_state['last_ai_log_stderr'] = "DataFrame vazio fornecido."
        return None
    
    try:
        # Chama a nova função de lógica do Gemini
        # A função _run_gemini_logic_on_dataframe é cacheada
        summary = _run_gemini_logic_on_dataframe(df_filtrado)
        
        if summary.startswith(("ERRO_API_CONFIG", "Erro", "Não há dados", "[ERRO")):
            st.error(f"Falha IA: {summary}")
            st.session_state['last_ai_log_stderr'] = summary
            return None
        
        log_success = f"Resumo (Gemini) gerado com sucesso ({len(summary)} caracteres)."
        st.session_state['last_ai_log_stdout'] = log_success
        print(log_success)
        return summary
    
    except Exception as e:
        error_msg = f"Erro inesperado ao chamar IA (Gemini): {e}"
        st.error(error_msg)
        st.session_state['last_ai_log_stderr'] = str(e)
        print(error_msg)
        return None

# --- <<< FIM DA LÓGICA DE IA >>> ---


# --- <<< INÍCIO DA CORREÇÃO DEFINITIVA: HTML (Gráficos + int64 em TUDO) >>> ---
def generate_html_report(df_to_save: pd.DataFrame, summary_text: str, profile_name_for_file: str,
                         start_date: date, end_date: date, original_profile_basename: str): 
    
    report_data = {} # Inicia o dicionário de dados
    
    if df_to_save.empty:
        st.warning("DataFrame vazio para o relatório HTML, o relatório conterá apenas o resumo.")
        # O código continua, mas 'report_data' estará vazio, o que é tratado pelo template.
    
    # --- 1. Métricas e Pizza de Sentimento (Bloco Try/Except individual) ---
    try:
        counts_b = df_to_save['sentimento'].value_counts()
        b_total = len(df_to_save)
        b_pos = counts_b.get('Positivo', 0)
        b_neu = counts_b.get('Neutro', 0)
        b_neg = counts_b.get('Negativo', 0)
        
        # CORREÇÃO (int64): Convertemos para Python int()
        metric_data = [ 
            ("Total Comentários", int(b_total), "#17a2b8", "fa-comments"), 
            ("Positivos", int(b_pos), COLOR_MAP['Positivo'], "fa-smile"), 
            ("Neutros", int(b_neu), COLOR_MAP['Neutro'], "fa-meh"), 
            ("Negativos", int(b_neg), COLOR_MAP['Negativo'], "fa-frown") 
        ]
        report_data['metrics'] = [ {"label": lbl, "value": val, "delta": f"{round((val*100)/b_total if b_total > 0 else 0, 1)}%", "color": col, "icon": ico} for lbl, val, col, ico in metric_data ]
        
        # CORREÇÃO (int64): Conversão explícita para lista de 'int'
        report_data['sentiment_counts'] = {
            "labels": counts_b.index.tolist(), 
            "data": [int(v) for v in counts_b.values]
        }
    except Exception as e:
        print(f"Erro ao gerar métricas/sentimento HTML: {e}")
        st.warning(f"Falha ao gerar dados de métricas/sentimento para o HTML: {e}")

    # --- 2. Pizza de Gênero (Bloco Try/Except individual) ---
    try:
        gender_counts = df_to_save['genero_previsto'].value_counts()
        if not gender_counts.empty:
            # CORREÇÃO (int64): Conversão explícita para lista de 'int'
            report_data['gender_pie_counts'] = {
                "labels": gender_counts.index.tolist(), 
                "data": [int(v) for v in gender_counts.values]
            }
    except Exception as e:
        print(f"Erro ao gerar gender_pie HTML: {e}")
        st.warning(f"Falha ao gerar dados de gênero (pizza) para o HTML: {e}")

    # --- 3. Barras Gênero x Sentimento (Bloco Try/Except individual) ---
    try:
        gender_sentiment = df_to_save.groupby(['genero_previsto', 'sentimento']).size().unstack(fill_value=0)
        if not gender_sentiment.empty:
            # CORREÇÃO (int64): Conversão explícita para lista de 'int' DENTRO do loop
            datasets = []
            for sent in gender_sentiment.columns:
                datasets.append({
                    "label": str(sent), 
                    "data": [int(v) for v in gender_sentiment[sent].values] # <-- AQUI
                })
            
            report_data['gender_bar_counts'] = {
                "labels": gender_sentiment.index.tolist(), 
                "datasets": datasets
            }
    except Exception as e:
        print(f"Erro ao gerar gender_bar HTML: {e}")
        st.warning(f"Falha ao gerar dados de gênero (barras) para o HTML: {e}")

    # --- 4. Frequência de Palavras (Bloco Try/Except individual) ---
    try:
        palavras = get_cleaned_words_for_freq(tuple(df_to_save["texto_puro"].astype(str).tolist()))
        if palavras: 
            freq_p = pd.Series(palavras).value_counts().nlargest(20)
            # CORREÇÃO (int64): Conversão explícita para lista de 'int'
            report_data['word_freq'] = {
                "labels": freq_p.index.tolist(), 
                "data": [int(v) for v in freq_p.values]
            }
    except Exception as e:
        print(f"Erro ao gerar word_freq HTML: {e}")
        st.warning(f"Falha ao gerar dados de frequência de palavras para o HTML: {e}")

    # --- 5. Frequência de Emojis (Bloco Try/Except individual) ---
    try:
        df_to_save['emojis'] = df_to_save['emojis'].apply(lambda x: x if isinstance(x, list) else [])
        all_emojis = [e for sublist in df_to_save['emojis'] for e in sublist]
        if all_emojis: 
            freq_e = pd.Series(all_emojis).value_counts().nlargest(15)
            # CORREÇÃO (int64): Conversão explícita para lista de 'int'
            report_data['emoji_freq'] = {
                "labels": freq_e.index.tolist(), 
                "data": [int(v) for v in freq_e.values]
            }
    except Exception as e:
        print(f"Erro ao gerar emoji_freq HTML: {e}")
        st.warning(f"Falha ao gerar dados de frequência de emojis para o HTML: {e}")

    # --- 6. Timeline (Bloco Try/Except individual) ---
    try:
        df_copy = df_to_save.copy()
        try: df_copy['data_hora'] = df_copy['data_hora'].dt.tz_convert(None)
        except TypeError: pass # Ignora se não tiver timezone
        
        df_copy['data'] = pd.to_datetime(df_copy['data_hora'], errors='coerce').dt.date
        df_copy.dropna(subset=['data'], inplace=True)
        
        if not df_copy.empty: 
            t_counts = df_copy.groupby('data').size().sort_index()
            if not t_counts.empty:
                labels_index = pd.to_datetime(t_counts.index)
                # CORREÇÃO (int64): Conversão explícita para lista de 'int'
                report_data['timeline'] = {
                    "labels": labels_index.strftime('%Y-%m-%d').tolist(), 
                    "data": [int(v) for v in t_counts.values]
                }
    except Exception as e:
        print(f"Erro ao gerar timeline HTML: {e}")
        st.warning(f"Falha ao gerar dados da timeline para o HTML: {e}")
        
    # --- 7. Gráfico de Tamanho Médio (Bloco Try/Except individual) ---
    try:
        if 'tamanho_comentario' in df_to_save.columns and 'sentimento' in df_to_save.columns:
            avg_length = df_to_save.groupby('sentimento')['tamanho_comentario'].mean().round(1)
            if not avg_length.empty:
                # CORREÇÃO (float64): Conversão explícita para lista de 'float'
                report_data['avg_length'] = {
                    "labels": avg_length.index.tolist(), 
                    "data": [float(v) for v in avg_length.values]
                }
    except Exception as e:
        print(f"Erro ao gerar avg_length HTML: {e}")
        st.warning(f"Falha ao gerar dados de tamanho médio para o HTML: {e}")

    # --- Geração da Tabela e JSON ---
    try:
        cols = ['usuario', 'conteudo', 'sentimento', 'genero_previsto', 'data_hora', 'fonte_coleta']
        ex_cols = [c for c in cols if c in df_to_save.columns]
        df_d = df_to_save.sort_values(by='data_hora', ascending=False)[ex_cols].head(10).copy()
        
        if 'data_hora' in df_d.columns:
            try: 
                df_d['data_hora'] = df_d['data_hora'].dt.tz_convert('America/Fortaleza').dt.strftime('%d/%m/%Y %H:%M')
            except: 
                df_d['data_hora'] = df_d['data_hora'].dt.strftime('%d/%m/%Y %H:%M')
        
        tabela_html = df_d.to_html(classes='table table-striped table-hover table-sm', index=False, escape=True, border=0)
    except Exception as e_table: 
        st.error(f"Erro tabela HTML: {e_table}"); tabela_html = "<p>Erro ao gerar tabela de amostra.</p>"
        
    try: 
        # Agora esta linha é segura, pois todos os valores são 'int' ou 'float' nativos
        report_data_json = json.dumps(report_data, ensure_ascii=False)
    except Exception as e_json: 
        st.error(f"Erro JSON HTML: {e_json}"); report_data_json = "{}"
        
    context = { 
        "report_name": f"{profile_name_for_file} ({start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')})", 
        "resumo_executivo": summary_text.replace('\n', '<br>') if summary_text else "Nenhum resumo executivo foi gerado para este período.", 
        "report_data_json": report_data_json, 
        "tabela_amostra": tabela_html 
    }
    
    safe_name = ''.join(c for c in profile_name_for_file if c.isalnum() or c in (' ', '_', '-')).rstrip().replace(' ', '_')
    html_filename = f"{safe_name}_relatorio_{start_date.strftime('%Y%m%d')}_a_{end_date.strftime('%Y%m%d')}.html"
    out_folder = os.path.join(HTML_OUTPUT_DIR, original_profile_basename); os.makedirs(out_folder, exist_ok=True); html_filepath = os.path.join(out_folder, html_filename)
    
    try:
        project_root = os.getcwd() 
        env = Environment(loader=FileSystemLoader(project_root), autoescape=True)
        template = env.get_template("report_template.html")
        
        html_content = template.render(context);
        with open(html_filepath, 'w', encoding='utf-8') as f: f.write(html_content)
        st.success(f"Relatório HTML '{html_filename}' gerado!"); return html_filepath
    except FileNotFoundError: 
        st.error("ERRO CRÍTICO: 'report_template.html' não encontrado. Verifique se ele está na raiz do seu projeto."); return None
    except Exception as e: 
        st.error(f"Erro final ao renderizar Jinja2 HTML: {e}"); return None

# --- <<< FIM DA CORREÇÃO DEFINITIVA >>> ---


def get_binary_file_downloader_html(bin_file, file_label='Arquivo'):
    try:
        with open(bin_file, 'rb') as f: data = f.read(); b64 = base64.b64encode(data).decode(); return f'<a href="data:text/html;base64,{b64}" download="{os.path.basename(bin_file)}">{file_label}</a>'
    except Exception as e: st.error(f"Erro link download: {e}"); return None

def check_critical_situation(df, threshold_neg_percent):
    if df.empty or 'sentimento' not in df.columns: return None
    counts = df['sentimento'].value_counts(); neg = counts.get('Negativo', 0); total = counts.sum()
    if total == 0: return None
    neg_p = (neg * 100) / total
    if neg_p >= threshold_neg_percent: return f"🚨 ATENÇÃO: {neg_p:.1f}% negativos (limite: {threshold_neg_percent:.1f}%)."
    return None

def extract_network_name(source_string: str) -> str:
    if not isinstance(source_string, str): return "Desconhecida"
    match = re.match(r"^(.*?):", source_string); return match.group(1) if match else "Desconhecida"

def display_dashboard_content(
    df_display: pd.DataFrame,
    profile_name: str,
    profile_metadata: dict,
    date_range_A: tuple,
    date_range_B: tuple,
    network_name: str
):
    """Exibe os filtros adicionais, gráficos E BOTÃO DE EXPORTAR para um DataFrame específico (já filtrado por rede)."""

    st.sidebar.divider() 
    st.sidebar.header(f"Filtros Adicionais ({network_name}) 🔍")

    selected_source = "All Sources"
    
    selected_run_id = "All Runs"; selected_run_id_display = "All Runs"
    if 'run_id' in df_display.columns:
        run_ids = df_display['run_id'].unique().tolist()
        if len(run_ids) > 1:
            try: sorted_run_ids = sorted([r for r in run_ids if r != 'N/A'], reverse=True)
            except Exception: sorted_run_ids = [r for r in run_ids if r != 'N/A']
            
            run_id_options_map = {}
            
            for r_id in sorted_run_ids:
                full_f_name = profile_metadata.get(r_id) 
                
                filtered_name_parts = []
                f_name = "" 
                
                if full_f_name:
                    parts = full_f_name.split(' | ')
                    for part in parts:
                        if part.lower().startswith(network_name.lower()):
                            filtered_name_parts.append(part)
                
                if filtered_name_parts:
                    f_name = " | ".join(filtered_name_parts)
                else:
                    f_name = full_f_name if full_f_name else r_id 
                
                try:
                    date_part = r_id.split('_')[-1] 
                    date_obj = datetime.strptime(date_part, '%Y%m%d-%H%M%S')
                    friendly_date = date_obj.strftime('%d/%m/%Y %H:%M')
                    display = f"{f_name}  ({friendly_date})" 
                except (ValueError, IndexError):
                    display = f"{f_name} ({r_id})" 
                
                run_id_options_map[display] = r_id

            if 'N/A' in run_ids: run_id_options_map['N/A'] = 'N/A'
            
            final_options = ["All Runs"] + list(run_id_options_map.keys())
            
            selected_run_id_display = st.sidebar.selectbox(
                f"Coleta Específica ({network_name}):", final_options, key=f"run_id_filter_{profile_name}_{network_name}",
                help="Selecione uma coleta (run_id) dentro desta rede."
            )
            selected_run_id = run_id_options_map.get(selected_run_id_display, "All Runs") if selected_run_id_display != "All Runs" else "All Runs"

    df_filtered_A, df_filtered_B = pd.DataFrame(), pd.DataFrame()
    valid_A, valid_B = False, False
    _df_final_filtered = df_display

    if selected_run_id != "All Runs":
        _df_final_filtered = _df_final_filtered[_df_final_filtered['run_id'] == selected_run_id]
        print(f"[{network_name}] Filtro Run ID: {selected_run_id}. Linhas: {len(_df_final_filtered)}")

    active_filters_list = []
    
    if selected_run_id != "All Runs": 
        active_filters_list.append(f"Coleta: {selected_run_id_display}")
    
    filter_title_string = f" ({', '.join(active_filters_list)})" if active_filters_list else ""

    if len(date_range_A) == 2:
        start_A, end_A = date_range_A
        if start_A and end_A:
            if start_A > end_A: start_A, end_A = end_A, start_A
            try:
                df_filtered_A = _df_final_filtered[(_df_final_filtered['data_hora'].dt.date >= start_A) & (_df_final_filtered['data_hora'].dt.date <= end_A)].copy()
                valid_A = True
                print(f"[{network_name}] Período A OK: {len(df_filtered_A)} linhas.")
            except Exception as e: st.error(f"Erro filtro A: {e}")
    if len(date_range_B) == 2:
        start_B, end_B = date_range_B
        if start_B and end_B:
            if start_B > end_B: start_B, end_B = end_B, start_B
            try:
                df_filtered_B = _df_final_filtered[(_df_final_filtered['data_hora'].dt.date >= start_B) & (_df_final_filtered['data_hora'].dt.date <= end_B)].copy()
                valid_B = True
                print(f"[{network_name}] Período B OK: {len(df_filtered_B)} linhas.")
            except Exception as e: st.error(f"Erro filtro B: {e}")

    if _df_final_filtered.empty and active_filters_list:
        st.warning(f"Nenhum dado encontrado para os filtros selecionados nesta rede.")
    elif not valid_B:
        st.error("Período Principal (B) inválido ou sem dados para os filtros selecionados.")
    else: 
        critical_msg = check_critical_situation(df_filtered_B, threshold_neg_percent=CRITICAL_NEG_THRESHOLD)
        if critical_msg: st.markdown(f'<div class="critical-alert">{critical_msg}</div>', unsafe_allow_html=True)

        plot_metrics(df_filtered_B, df_filtered_A if valid_A else pd.DataFrame())

        st.header(f"Análise Detalhada (Período Principal{filter_title_string})")

        sub_tab1, sub_tab2, sub_tab3 = st.tabs(["📊 Sent./Gênero", "📝 Conteúdo", "⏰ Timeline/Amostra"])
        
        # --- CORREÇÃO: width='stretch' -> use_container_width=True ---
        with sub_tab1:
             if not df_filtered_B.empty:
                 col1, col2 = st.columns(2)
                 with col1: st.plotly_chart(get_fig_pie_chart(df_filtered_B), use_container_width=True)
                 with col2: st.plotly_chart(get_fig_gender_chart(df_filtered_B), use_container_width=True)
             else: st.info(f"Sem dados de sentimento/gênero para os filtros.")
        with sub_tab2:
            if not df_filtered_B.empty:
                col3, col4 = st.columns(2)
                with col3:
                    with st.spinner("Analisando freq..."):
                        st.plotly_chart(get_word_frequency_fig(df_filtered_B), use_container_width=True) 
                with col4: st.plotly_chart(get_fig_top_emojis(df_filtered_B), use_container_width=True)
                st.plotly_chart(get_fig_comment_length(df_filtered_B), use_container_width=True)
            else: st.info(f"Sem dados de conteúdo para os filtros.")
        with sub_tab3:
            if not df_filtered_B.empty:
                st.plotly_chart(get_fig_timeline(df_filtered_B), use_container_width=True)
                st.subheader(f"Amostra de Dados (Período Principal{filter_title_string})")
                cols_s = ['usuario', 'conteudo', 'sentimento', 'genero_previsto', 'data_hora', 'fonte_coleta', 'run_id']; ex_s = [c for c in cols_s if c in df_filtered_B.columns]
                df_ds = df_filtered_B[ex_s].head(20).copy()
                
                if 'data_hora' in df_ds.columns:
                    try: 
                        df_ds['data_hora'] = df_ds['data_hora'].dt.tz_convert('America/Fortaleza').dt.strftime('%d/%m/%Y %H:%M')
                    except: 
                        df_ds['data_hora'] = df_ds['data_hora'].dt.strftime('%d/%m/%Y %H:%M')
                st.dataframe(df_ds, use_container_width=True)
            else: st.info(f"Sem dados de timeline/amostra para os filtros.")
        # --- FIM DA CORREÇÃO ---


    st.sidebar.header(f"Exportar ({network_name}) 📄")
    valid_B_global = len(date_range_B) == 2 and date_range_B[0] and date_range_B[1]
    
    if valid_B and valid_B_global: 
        start_B_html, end_B_html = date_range_B
        
        profile_name_for_report = profile_name
        if network_name != "Visão Geral":
            profile_name_for_report += f"_({network_name})"
            
        if selected_run_id != "All Runs":
            f_name = selected_run_id_display
            safe_r = ''.join(c for c in f_name if c.isalnum() or c in (' ', '_', '-', '(', ')', ':', '/')).rstrip().replace(' ', '_')
            profile_name_for_report += f"_({safe_r})"

        html_button_help = f"Gera HTML de {start_B_html.strftime('%d/%m')} a {end_B_html.strftime('%d/%m')}."
        if active_filters_list:
            html_button_help += f" (Filtros: {', '.join(active_filters_list)})"

        summary_key_current = f"summary_{profile_name}_{date_range_B[0]}_{date_range_B[1]}"
        
        # --- CORREÇÃO: width='stretch' -> use_container_width=True ---
        if st.sidebar.button("Gerar Relatório HTML (Visão Atual)", key=f"btn_gen_{profile_name}_{network_name}", use_container_width=True, help=html_button_help):
            summary_for_html = st.session_state.get('generated_summary') if st.session_state.get('summary_period_key') == summary_key_current else "Resumo IA não gerado/válido para este período."
            
            df_to_export = df_filtered_B 

            if not df_to_export.empty or (summary_for_html and not summary_for_html.startswith("Resumo")):
                with st.spinner("Gerando HTML..."):
                    generated_html_path = generate_html_report(
                        df_to_export, summary_for_html, profile_name_for_report,
                        start_B_html, end_B_html, profile_name
                    ) 
                    if generated_html_path and os.path.exists(generated_html_path):
                        st.session_state[f'download_path_{profile_name}_{network_name}'] = generated_html_path
                        st.session_state[f'download_name_{profile_name}_{network_name}'] = os.path.basename(generated_html_path)
                        st.rerun()
                    else:
                        st.sidebar.error("Falha ao gerar o arquivo HTML. Verifique os logs.")
                        
            else:
                st.warning(f"Sem dados na visão atual para gerar HTML.")

    dl_path_key = f'download_path_{profile_name}_{network_name}'
    dl_name_key = f'download_name_{profile_name}_{network_name}'
    if dl_path_key in st.session_state and os.path.exists(st.session_state.get(dl_path_key,'')):
        dl_link = get_binary_file_downloader_html(st.session_state[dl_path_key], f'Download {st.session_state[dl_name_key]}')
        if dl_link: st.sidebar.markdown(dl_link, unsafe_allow_html=True)

# --- LÓGICA PRINCIPAL (REVISADA) ---
try:
    # 1. Procura perfis nos dados JSON processados (fonte primária)
    json_files = glob.glob(os.path.join(PROCESSED_DATA_DIR, "*.json"))
    profile_set = set()
    
    if json_files:
        for f in json_files:
            basename = os.path.basename(f)
            profile_name_match = basename.rsplit('_', 1)
            if len(profile_name_match) == 2 and profile_name_match[0]:
                profile_set.add(profile_name_match[0])

    # 2. Procura perfis nos arquivos .db (para o caso de só existir o DB)
    if os.path.exists(DB_DIR):
        db_files = [f for f in os.listdir(DB_DIR) if f.endswith('.db') and not f.startswith('.')]
        for f in db_files:
            profile_set.add(f.replace(".db", ""))
        
except FileNotFoundError:
    st.error(f"ERRO: Pasta '{PROCESSED_DATA_DIR}' ou '{DB_DIR}' não encontrada.")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(DB_DIR, exist_ok=True)
    st.stop()

# 3. Verifica se encontrou algum perfil
if not profile_set:
    st.warning(f"Nenhum dado encontrado em '{PROCESSED_DATA_DIR}' ou DB em '{DB_DIR}'. Execute 'run_all.py'.")
    st.stop()

# 4. Cria a lista de opções
profile_names = sorted(list(profile_set))
options_list = ["--- Selecione um Perfil ---"] + profile_names

selected_profile_name = st.sidebar.selectbox(
    "Selecione o Perfil:",
    options_list,
    index=0,  
    key="select_profile"
)

st.session_state.setdefault('generated_summary', None)
st.session_state.setdefault('summary_period_key', None)
st.session_state.setdefault('last_ai_log_stderr', "")
st.session_state.setdefault('ai_failed', False)

if selected_profile_name != "--- Selecione um Perfil ---":
    profile_name = selected_profile_name
    db_path = os.path.join(DB_DIR, f"{profile_name}.db") # O db_path ainda é útil para a 'load_run_metadata'

    if st.session_state.get('last_profile_name') != profile_name:
        last_profile_name = st.session_state.get("last_profile_name", "")
        for key in list(st.session_state.keys()):
            if key.startswith(f'download_path_{last_profile_name}') or \
               key.startswith(f'download_name_{last_profile_name}') or \
               key in ['generated_summary', 'summary_period_key', 'last_ai_log_stderr', 'ai_failed']:
                del st.session_state[key]
        st.session_state['last_profile_name'] = profile_name
        print(f"Mudou para perfil {profile_name}, limpando estado.")
        st.cache_data.clear() 
        st.rerun() 

    all_metadata = load_run_metadata()
    profile_metadata = all_metadata.get(profile_name, {})
    
    # Carrega o DataFrame dos JSONs
    _df_full = load_processed_data_for_profile(profile_name)

    if not _df_full.empty and 'data_hora' in _df_full.columns and not _df_full['data_hora'].isnull().all():

        st.sidebar.header("Filtros por Período 🗓️")
        date_range_A, date_range_B = (), ()
        try:
            min_date_limit_full = _df_full['data_hora'].min().date() 
            max_date_limit_full = _df_full['data_hora'].max().date()
            date_range_B = st.sidebar.date_input( "Período Principal (B):", (min_date_limit_full, max_date_limit_full), min_value=min_date_limit_full, max_value=max_date_limit_full, key=f"date_filter_B_{profile_name}" )
            default_A_end = date_range_B[0] - timedelta(days=1) if len(date_range_B) == 2 and date_range_B[0] > min_date_limit_full else min_date_limit_full
            default_A_start = max(default_A_end - timedelta(days=6), min_date_limit_full); default_A_end = min(default_A_end, max_date_limit_full); default_A_end = max(default_A_end, default_A_start)
            date_range_A = st.sidebar.date_input( "Período de Comparação (A):", (default_A_start, default_A_end), min_value=min_date_limit_full, max_value=max_date_limit_full, key=f"date_filter_A_{profile_name}" )
        except Exception as e:
            st.sidebar.error(f"Erro nas datas: {e}")

        st.sidebar.divider()
        st.sidebar.header("Ações Globais ⚙️")
        valid_B_global = len(date_range_B) == 2 and date_range_B[0] and date_range_B[1]
        summary_key_current = f"summary_{profile_name}_{date_range_B[0]}_{date_range_B[1]}" if valid_B_global else None

        if valid_B_global:
            start_B_ai, end_B_ai = date_range_B
            
            # --- CORREÇÃO: width='stretch' -> use_container_width=True ---
            if st.sidebar.button("Gerar Resumo IA (Período Principal)", key=f"btn_ai_{profile_name}", use_container_width=True, help=f"Analisa {start_B_ai.strftime('%d/%m')} a {end_B_ai.strftime('%d/%m')} (TODAS as redes)."):
                st.session_state['ai_failed'] = False; st.session_state['last_ai_log_stderr'] = "Executando..."
                
                df_para_ia = pd.DataFrame() # Inicia vazio
                
                # 1. Filtramos o DataFrame principal (_df_full) com as datas
                try:
                    df_para_ia = _df_full[
                        (_df_full['data_hora'].dt.date >= start_B_ai) & 
                        (_df_full['data_hora'].dt.date <= end_B_ai)
                    ].copy()
                    print(f"Filtrados {len(df_para_ia)} registros para IA.")
                except Exception as e:
                    st.error(f"Erro ao filtrar dados para IA: {e}")
                    df_para_ia = pd.DataFrame() # Garante que está vazio em caso de erro

                # 2. Passamos o DataFrame filtrado para a função
                with st.spinner("Gerando Resumo IA (Gemini)..."):
                    # A função agora só precisa do DataFrame
                    summary_result = run_ai_summary_generation(df_para_ia) 
                    
                    if summary_result: 
                        st.session_state['generated_summary'] = summary_result
                        st.session_state['summary_period_key'] = summary_key_current
                        st.session_state['ai_failed'] = False
                        st.rerun()
                    else: 
                        st.session_state['ai_failed'] = True
            # --- FIM DA CORREÇÃO ---

        st.header("Resumo Executivo (Gerado por IA)")
        if st.session_state.get('summary_period_key') == summary_key_current and st.session_state.get('generated_summary'):
            st.markdown(f'<div class="summary-box">{st.session_state["generated_summary"].replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
            st.caption("Nota: Resumo IA analisa **todas as redes sociais** do período, independente da aba selecionada.")
        elif summary_key_current: st.info("Clique em 'Gerar Resumo IA' na barra lateral.")
        else: st.warning("Selecione um Período Principal válido.")

        if st.session_state.get('last_ai_log_stderr'):
            with st.expander("Ver log da última execução de IA", expanded=st.session_state.get('ai_failed', False)):
                st.subheader("Logs / Erros (stderr):"); st.code(st.session_state['last_ai_log_stderr'], language='bash')
        
        st.divider() 

        _df_full['network'] = _df_full['fonte_coleta'].apply(extract_network_name)
        available_networks = sorted([n for n in _df_full['network'].unique() if n != "Desconhecida"])

        if len(available_networks) > 1:
            network_tabs = st.tabs(available_networks)
            for i, network_name in enumerate(available_networks):
                with network_tabs[i]:
                    df_network = _df_full[_df_full['network'] == network_name]
                    display_dashboard_content(df_network, profile_name, profile_metadata, date_range_A, date_range_B, network_name)

        elif len(available_networks) == 1:
            network_name = available_networks[0]
            st.subheader(f"Análise - {network_name}")
            display_dashboard_content(_df_full, profile_name, profile_metadata, date_range_A, date_range_B, network_name)
        
        else:
            st.warning("Nenhuma rede social conhecida encontrada nos dados.")

    elif selected_profile_name:
        st.error(f"Falha ao carregar dados do perfil '{selected_profile_name}'. Verifique JSONs em '{PROCESSED_DATA_DIR}' ou execute coleta.")

else:
    st.info("Por favor, selecione um perfil na barra lateral para carregar os dados.")