# gerar_relatorio.py (FINALMENTE CORRIGIDO)
import os
import sqlite3
import time
import pandas as pd
import json
import google.generativeai as genai
import logging
import sys
import argparse
import io # Importado para forçar UTF-8
from typing import List, Dict, Optional
from src.utils.config import CONFIG, GEMINI_API_KEY
from datetime import datetime

# --- FUNÇÃO DE CONSULTA AO BANCO (CORRIGIDA) ---
def carregar_conteudo_periodo(db_path: str, profile_name: str, start_date_str: str, end_date_str: str) -> Dict[str, List[str]]:
    """Carrega os comentários de um perfil dentro de um intervalo de datas, agrupados por fonte."""
    conn = sqlite3.connect(db_path)
    # start_date_str e end_date_str já estão no formato 'AAAA-MM-DD'
    
    has_parent_url_col = False
    try:
        logging.info("--- DEBUG: VERIFICANDO ESTRUTURA DO BANCO ---")
        df_cols = pd.read_sql_query("SELECT * FROM posts LIMIT 1", conn)
        col_list = df_cols.columns.tolist()
        logging.info(f"COLUNAS ENCONTRADAS: {col_list}")
        
        has_parent_url_col = 'parent_url' in col_list
        logging.info(f"Verificação de colunas: 'parent_url' existe? {has_parent_url_col}")
        logging.info("--- FIM DO DEBUG ---")
    except Exception as e_cols:
        logging.warning(f"Não foi possível verificar colunas da tabela: {e_cols}.")
        has_parent_url_col = False

    # --- [CORREÇÃO SQL] ---
    # A query base agora usa a função date() do SQLite
    # para comparar as datas corretamente, ignorando o 'T...Z'
    query_base = f"""
    SELECT fonte_coleta, conteudo
    FROM posts
    WHERE date(data_hora) BETWEEN '{start_date_str}' AND '{end_date_str}'
      AND conteudo IS NOT NULL AND conteudo != ''
    """

    if has_parent_url_col:
        query_filter = " AND parent_url IS NOT NULL"
        logging.info(f"QUERY: Filtrando por comentários (parent_url IS NOT NULL) e datas.")
    else:
        query_filter = "" 
        logging.warning("QUERY: Coluna 'parent_url' não encontrada. Analisando TODOS os posts.")

    query = query_base + query_filter
    
    try:
        df = pd.read_sql_query(query, conn)
        if df.empty:
            logging.warning(f"Nenhum comentário/post encontrado para '{profile_name}' entre {start_date_str} e {end_date_str}.")
            # Log da query exata que falhou
            logging.debug(f"Query executada (sem resultados): {query}") 
            return {}
        logging.info(f"Encontrados {len(df)} comentários/posts no período para análise com IA.")
        return df.groupby('fonte_coleta')['conteudo'].apply(list).to_dict()
    except Exception as e:
         logging.error(f"Erro ao consultar o banco de dados por período: {e}")
         return {}
    finally:
        if conn:
            conn.close()

# --- FUNÇÕES GEMINI (gerar_resumo_com_gemini, gerar_resumo_executivo - Sem mudanças) ---
def gerar_resumo_com_gemini(conteudos: List[str], nome_grupo: str, modelo) -> str:
    if not conteudos: return "[SEM COMENTÁRIOS NESTE GRUPO]"
    conteudos_para_analise = conteudos[:CONFIG.get('gemini', {}).get('max_comments_per_group', 250)]
    conteudos_formatados = "\n".join([f'- "{str(c).strip()}"' for c in conteudos_para_analise])
    prompt = CONFIG.get('gemini', {}).get('detailed_summary_prompt', "").format(
        nome_grupo=nome_grupo,
        conteudos_formatados=conteudos_formatados
    )
    if not prompt:
        logging.error("Prompt para resumo detalhado não encontrado.")
        return "[ERRO: PROMPT NÃO CONFIGURADO]"
    try:
        time.sleep(CONFIG.get('gemini', {}).get('api_delay_seconds', 2))
        response = modelo.generate_content(prompt)
        if response.parts: return response.text.strip()
        else:
            reason = response.candidates[0].finish_reason if response.candidates else "DESCONHECIDO"
            logging.warning(f"API Gemini bloqueou resposta para '{nome_grupo}'. Motivo: {reason}.")
            return f"[RESPOSTA BLOQUEADA PELA API - Motivo: {reason}]"
    except Exception as e:
        logging.error(f"Erro na API Gemini para '{nome_grupo}': {e}")
        return "[ERRO AO GERAR RESUMO DETALHADO]"

def gerar_resumo_executivo(resumos_detalhados: List[str], modelo) -> str:
    if not resumos_detalhados or all(r.startswith("[") for r in resumos_detalhados):
        return "Não foi possível gerar o resumo executivo devido a erros ou falta de dados nas análises detalhadas."
    texto_dos_resumos = "\n\n---\n\n".join(filter(lambda r: not r.startswith("["), resumos_detalhados))
    prompt = CONFIG.get('gemini', {}).get('executive_summary_prompt', "").format(
        texto_dos_resumos=texto_dos_resumos
    )
    if not prompt:
        logging.error("Prompt para resumo executivo não encontrado.")
        return "[ERRO: PROMPT NÃO CONFIGURADO]"
    try:
        time.sleep(CONFIG.get('gemini', {}).get('api_delay_seconds', 2))
        response = modelo.generate_content(prompt)
        if response.parts: return response.text.strip()
        else:
            reason = response.candidates[0].finish_reason if response.candidates else "DESCONHECIDO"
            logging.warning(f"API Gemini bloqueou resposta para Resumo Executivo. Motivo: {reason}.")
            return f"[RESUMO EXECUTIVO BLOQUEADO PELA API - Motivo: {reason}]"
    except Exception as e:
        logging.error(f"Erro na API Gemini para Resumo Executivo: {e}")
        return "[ERRO AO GERAR O RESUMO EXECUTIVO]"

def main():
    # --- [CORREÇÃO DE ENCODING (WINDOWS)] ---
    # Força stdout e stderr a usarem UTF-8, independentemente do console
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    # Configura o logging para ir para o STDERR (como é o padrão)
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] [%(asctime)s] - %(message)s',
                        handlers=[logging.StreamHandler(sys.stderr)]) 
    
    logging.info("--- INICIANDO GERADOR_RELATORIO.PY (Modo UTF-8 Forçado) ---") 

    parser = argparse.ArgumentParser(description="Gerador de Resumos com IA para um período específico.")
    parser.add_argument('--db-path', required=True, help="Caminho para o banco de dados.")
    parser.add_argument('--profile-name', required=True, help="Nome do perfil sendo analisado.")
    parser.add_argument('--start-date', required=True, help="Data de início (AAAA-MM-DD).")
    parser.add_argument('--end-date', required=True, help="Data de fim (AAAA-MM-DD).")
    args = parser.parse_args()

    # Configura API Gemini
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        modelo = genai.GenerativeModel(
            model_name=CONFIG['gemini']['model_name'],
            generation_config=CONFIG['gemini']['generation_config']
        )
        logging.info("API Gemini configurada.")
    except Exception as e:
        logging.critical(f"Falha ao configurar a API Gemini: {e}")
        print(f"ERRO_API_CONFIG: Falha ao configurar a API Gemini: {e}", file=sys.stdout)
        sys.exit(1)

    # Carrega dados do período
    grupos_de_conteudo = carregar_conteudo_periodo(args.db_path, args.profile_name, args.start_date, args.end_date)
    executive_summary = f"Não foram encontrados comentários para o perfil '{args.profile_name}' entre {args.start_date} e {args.end_date}."

    if grupos_de_conteudo:
        logging.info(f"Iniciando análise com Gemini para {len(grupos_de_conteudo)} fontes no período.")
        detailed_analysis_results = {
            nome: gerar_resumo_com_gemini(conteudos, nome, modelo)
            for nome, conteudos in grupos_de_conteudo.items()
        }
        resumos_validos = [res for res in detailed_analysis_results.values() if not res.startswith("[")]
        executive_summary = gerar_resumo_executivo(resumos_validos, modelo)
        logging.info("Resumos gerados pela IA.")

    # Imprime o resumo final (ou a mensagem de falha) no STDOUT
    print(executive_summary, file=sys.stdout)
    logging.info("Resumo executivo enviado para stdout.")

if __name__ == "__main__":
    main()