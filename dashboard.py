import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import json
import glob
from datetime import datetime, timedelta

# --- CONFIGURAÇÃO VISUAL WAR ROOM ---
st.set_page_config(page_title="Monitoramento", layout="wide", page_icon="🛡️")

# CSS Injetado: Dark Mode Profissional
st.markdown("""<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    h1, h2, h3 { color: #E5E7EB !important; }
    
    /* Cards de Métricas */
    div[data-testid="metric-container"] {
        background-color: #1F2937;
        border: 1px solid #374151;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
    }
    div[data-testid="metric-value"] { color: #F3F4F6 !important; font-weight: 700 !important; }
    div[data-testid="metric-label"] { color: #9CA3AF !important; }
    
    /* Alertas */
    .critical-alert {
        background-color: #450a0a; border: 1px solid #ef4444; color: #fca5a5;
        padding: 1rem; border-radius: 8px; text-align: center; font-weight: bold;
        animation: pulse 2s infinite; margin-bottom: 20px;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
</style>""", unsafe_allow_html=True)

st.title("🛡️ WAR ROOM ELEITORAL 2026")

# --- FUNÇÕES DE CARREGAMENTO ---
PROCESSED_DATA_DIR = "relatorios_processados"
COLOR_MAP = {'Negativo': '#EF4444', 'Neutro': '#9CA3AF', 'Positivo': '#10B981'}

@st.cache_data(ttl=60)
def load_data(profile_name):
    files = glob.glob(os.path.join(PROCESSED_DATA_DIR, f"{profile_name}_*.json"))
    if not files: return pd.DataFrame()
    
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_json(f, orient='records', convert_dates=['data_hora']))
        except: pass
    
    if not dfs: return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    
    # Normalização
    if 'sentimento_final' in df.columns: df['sentimento'] = df['sentimento_final']
    if 'sentimento' not in df.columns: df['sentimento'] = 'Neutro'
    if 'analise_tematica_json' not in df.columns: df['analise_tematica_json'] = "{}"
    
    return df

# --- GRÁFICOS AVANÇADOS ---

def plot_metrics_nss(df):
    st.subheader("🚦 Termômetro da Campanha (NSS)")
    if df.empty: st.warning("Sem dados."); return
    
    total = len(df)
    counts = df['sentimento'].value_counts()
    pos = counts.get('Positivo', 0)
    neg = counts.get('Negativo', 0)
    
    # Cálculo do NSS (Net Sentiment Score)
    nss = ((pos - neg) / total) * 100 if total > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Volume Total", total)
    
    delta_color = "normal" if nss > 0 else "inverse"
    col2.metric("NSS (Aprovação Líquida)", f"{nss:.1f}", delta=f"{nss:.1f} pts", delta_color=delta_color, help="De -100 a +100")
    
    with col3:
        pct_neg = (neg/total)*100
        st.metric("Taxa de Rejeição", f"{pct_neg:.1f}%", delta=None)
        if pct_neg > 30:
            st.markdown('<div class="critical-alert">🚨 ALERTA: REJEIÇÃO ALTA (>30%)</div>', unsafe_allow_html=True)

def plot_heatmap_tematico(df):
    st.subheader("🔥 Mapa de Calor Temático (IA)")
    if df.empty or 'analise_tematica_json' not in df.columns: 
        st.info("Sem dados temáticos de IA."); return

    temas_count = {}
    
    # Parseia o JSON gerado pelo Gemini
    for json_str in df['analise_tematica_json']:
        try:
            if not json_str or json_str == "{}": continue
            dados = json.loads(json_str)
            for tema, sent in dados.items():
                if tema not in temas_count: temas_count[tema] = {'Positivo': 0, 'Negativo': 0}
                if sent in ['Positivo', 'Negativo']:
                    temas_count[tema][sent] += 1
        except: continue

    if not temas_count: st.info("IA não detectou temas relevantes."); return

    rows = []
    for tema, sents in temas_count.items():
        # Score de Crise: Ordena pelo que tem mais Negativo
        score = sents['Negativo'] + sents['Positivo']
        if score > 0:
            rows.append({'Tema': tema, 'Sentimento': 'Negativo', 'Qtd': -sents['Negativo'], 'Abs': sents['Negativo']})
            rows.append({'Tema': tema, 'Sentimento': 'Positivo', 'Qtd': sents['Positivo'], 'Abs': sents['Positivo']})
    
    df_chart = pd.DataFrame(rows)
    if df_chart.empty: return
    
    # Ordena para os problemas aparecerem em cima
    total_por_tema = df_chart.groupby('Tema')['Abs'].sum().sort_values(ascending=True)
    
    fig = px.bar(df_chart, x='Qtd', y='Tema', color='Sentimento', orientation='h',
                 color_discrete_map={'Negativo': '#EF4444', 'Positivo': '#10B981'},
                 category_orders={'Tema': total_por_tema.index.tolist()},
                 title="Balanço Temático (Esquerda: Crise | Direita: Vitória)")
    
    fig.update_layout(barmode='relative', xaxis_title="Volume", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(fig, use_container_width=True)

def plot_influencers(df):
    st.subheader("📢 Radar de Influência")
    if df.empty: return
    
    user_stats = df.groupby('usuario').agg(
        Total=('conteudo', 'count'),
        Negativos=('sentimento', lambda x: (x == 'Negativo').sum()),
        Positivos=('sentimento', lambda x: (x == 'Positivo').sum())
    ).reset_index()
    
    user_stats = user_stats[user_stats['Total'] > 1] # Filtra ocasionais
    
    c1, c2 = st.columns(2)
    with c1:
        st.write("🛡️ **Top Defensores**")
        st.dataframe(user_stats.sort_values('Positivos', ascending=False).head(5)[['usuario', 'Positivos']], hide_index=True, use_container_width=True)
    with c2:
        st.write("🤬 **Top Detratores**")
        st.dataframe(user_stats.sort_values('Negativos', ascending=False).head(5)[['usuario', 'Negativos']], hide_index=True, use_container_width=True)

# --- EXECUÇÃO DO DASHBOARD ---

# Sidebar para Seleção
profiles = [f.split('_run')[0] for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('.json')]
profiles = list(set(profiles))

if not profiles:
    st.error("Nenhum dado encontrado. Execute 'run_all.py' primeiro.")
else:
    selected_profile = st.sidebar.selectbox("Selecione o Alvo:", profiles)
    df = load_data(selected_profile)
    
    if not df.empty:
        # Filtro de Data
        min_date = df['data_hora'].min().date()
        max_date = df['data_hora'].max().date()
        range_date = st.sidebar.date_input("Período:", [min_date, max_date], min_value=min_date, max_value=max_date)
        
        if len(range_date) == 2:
            df = df[(df['data_hora'].dt.date >= range_date[0]) & (df['data_hora'].dt.date <= range_date[1])]

        # Renderização
        plot_metrics_nss(df)
        
        tab1, tab2, tab3 = st.tabs(["📊 Temas & Crises", "👥 Influenciadores", "📝 Feed Bruto"])
        
        with tab1:
            plot_heatmap_tematico(df)
            col_a, col_b = st.columns(2)
            col_a.plotly_chart(px.pie(df, names='sentimento', title="Distribuição Geral", color='sentimento', color_discrete_map=COLOR_MAP), use_container_width=True)
            # Nuvem de palavras simples (legado) pode ir aqui se quiser
            
        with tab2:
            plot_influencers(df)
            
        with tab3:
            st.dataframe(df[['data_hora', 'usuario', 'conteudo', 'sentimento', 'analise_tematica_json']].sort_values('data_hora', ascending=False), use_container_width=True)