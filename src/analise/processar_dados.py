# src/analise/processar_dados.py
import sqlite3
import pandas as pd
import argparse
import os
import sys
import logging
import json
import io # Para forçar UTF-8 no Windows
import emoji # <<< NOVO IMPORT >>>
from datetime import datetime

# --- Configuração do Logging (Forçando UTF-8 para evitar erros no Windows) ---
# Remove handlers antigos para evitar duplicação se o script for importado
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Força stdout/stderr para UTF-8 (útil quando chamado via subprocess no Windows)
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
except Exception as e:
    print(f"[processar_dados.py Aviso] Não foi possível reconfigurar stdout/stderr para UTF-8: {e}", file=sys.stderr)

# Configura logging para ir para stderr
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] [%(asctime)s] - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)

# --- Diretório de Saída ---
try:
    SCRIPT_DIR = os.path.dirname(__file__)
    BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..')) # Sobe dois níveis (de src/analise para a raiz)
    OUTPUT_DIR = os.path.join(BASE_DIR, "relatorios_processados")
except NameError:
    BASE_DIR = os.getcwd()
    OUTPUT_DIR = os.path.join(BASE_DIR, "relatorios_processados")
    logging.warning(f"Não foi possível determinar o diretório base a partir do script, usando CWD: {BASE_DIR}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.info(f"Diretório de saída definido como: {OUTPUT_DIR}")


# --- Função Principal de Processamento ---
def processar_run(db_path: str, run_id: str):
    """Lê, processa e salva dados para um run_id específico."""
    logging.info(f"Iniciando processamento para DB: '{db_path}' e Run ID: '{run_id}'...")

    if not os.path.isabs(db_path):
        db_path_abs = os.path.join(BASE_DIR, db_path) 
        logging.info(f"Convertendo path relativo do DB ('{db_path}') para absoluto: {db_path_abs}")
        if not os.path.exists(db_path_abs):
             logging.error(f"Erro: Banco de dados não encontrado no caminho absoluto: {db_path_abs}")
             db_path_cwd = os.path.abspath(db_path)
             logging.warning(f"Tentando caminho relativo ao diretório atual: {db_path_cwd}")
             if not os.path.exists(db_path_cwd):
                  logging.error(f"ERRO CRÍTICO: Banco de dados não encontrado em {db_path_abs} nem {db_path_cwd}")
                  return 
             else:
                  db_path = db_path_cwd
        else:
            db_path = db_path_abs


    conn = sqlite3.connect(db_path)
    df_processed = pd.DataFrame() 
    db_basename = os.path.splitext(os.path.basename(db_path))[0]
    base_filename = f"{db_basename}_{run_id}"
    json_path = os.path.join(OUTPUT_DIR, f"{base_filename}.json")
    txt_path = os.path.join(OUTPUT_DIR, f"{base_filename}.txt")

    try:
        query = f"SELECT * FROM posts WHERE run_id = ?"
        df_raw = pd.read_sql_query(query, conn, params=(run_id,))
        logging.info(f"Lidas {len(df_raw)} linhas brutas do DB para o Run ID: {run_id}")

        if df_raw.empty:
            logging.warning("DataFrame bruto está vazio para este Run ID. Nenhum dado para processar.")
            with open(json_path, 'w', encoding='utf-8') as f_json: json.dump([], f_json)
            with open(txt_path, 'w', encoding='utf-8') as f_txt: f_txt.write("Nenhum comentário encontrado para este Run ID.")
            logging.info(f"Arquivos vazios criados em: {OUTPUT_DIR}")
            return 

        logging.info("\n--- AMOSTRA DE DADOS BRUTOS (5 PRIMEIRAS LINHAS) ---")
        try:
            logging.info("\n" + df_raw.head().to_string())
        except Exception as e_print:
             logging.error(f"Erro ao tentar printar amostra de dados: {e_print}")
        logging.info("--- FIM DA AMOSTRA ---")

        # 2. Processamento e Limpeza
        df_processed = df_raw.copy()

        required_cols = {'conteudo': "", 'data_hora': pd.NaT, 'sentimento': None, 'genero_previsto': None, 'usuario': 'Desconhecido', 'fonte_coleta': 'N/A'}
        for col, default in required_cols.items():
             if col not in df_processed.columns:
                 df_processed[col] = default
                 logging.warning(f"Coluna '{col}' não encontrada, adicionada com valor padrão.")

        if 'data_hora' in df_processed.columns:
            logging.info("Convertendo coluna 'data_hora'...")
            df_processed['data_hora'] = pd.to_datetime(df_processed['data_hora'], errors='coerce', utc=True)
            linhas_invalidas_inicial = df_processed['data_hora'].isnull().sum()
            if linhas_invalidas_inicial > 0:
                 logging.warning(f"AVISO: {linhas_invalidas_inicial} linhas com data inválida detectadas APÓS A CONVERSÃO inicial.")
                 df_processed.dropna(subset=['data_hora'], inplace=True)
                 logging.info(f"Linhas com data inválida removidas. Restam {len(df_processed)} linhas.")

            if df_processed.empty and linhas_invalidas_inicial > 0: 
                 logging.error("ERRO CRÍTICO: Todas as linhas foram removidas devido a datas inválidas.")
                 with open(json_path, 'w', encoding='utf-8') as f_json: json.dump([], f_json)
                 with open(txt_path, 'w', encoding='utf-8') as f_txt: f_txt.write("ERRO: Todas as datas eram inválidas.")
                 logging.info(f"Arquivos de erro criados em: {OUTPUT_DIR}")
                 return
        else:
            logging.error("ERRO CRÍTICO: Coluna 'data_hora' não encontrada.")
            return

        if df_processed.empty:
             logging.warning("DataFrame ficou vazio após remoção de datas inválidas.")
             with open(json_path, 'w', encoding='utf-8') as f_json: json.dump([], f_json)
             with open(txt_path, 'w', encoding='utf-8') as f_txt: f_txt.write("Nenhum comentário válido após filtro de data.")
             logging.info(f"Arquivos vazios criados em: {OUTPUT_DIR}")
             return


        logging.info(f"Processando {len(df_processed)} comentários válidos...")

        # --- <<< INÍCIO DA MUDANÇA: PROCESSAMENTO PESADO MOVIDO PARA CÁ >>> ---
        logging.info("Iniciando extração de emojis e limpeza de texto...")
        # (Estas funções foram copiadas do seu 'gerador.py')
        df_processed['emojis'] = df_processed['conteudo'].apply(lambda t: [e['emoji'] for e in emoji.emoji_list(str(t))] if pd.notna(t) else [])
        df_processed['texto_puro'] = df_processed['conteudo'].apply(lambda t: emoji.replace_emoji(str(t), replace='') if pd.notna(t) else "")
        df_processed['tamanho_comentario'] = df_processed['texto_puro'].str.len().fillna(0).astype(int)
        logging.info("Extração de texto/emoji concluída.")
        # --- <<< FIM DA MUDANÇA >>> ---

        # Mapeamento/Limpeza Final
        sentiment_map = {'neg': 'Negativo', 'neu': 'Neutro', 'pos': 'Positivo'}
        df_processed['sentimento_final'] = df_processed['sentimento'].map(sentiment_map).fillna('Neutro')
        df_processed['genero_final'] = df_processed['genero_previsto'].fillna('Desconhecido').replace(['indeterminado', 'unknown'], 'Desconhecido', regex=False)

        logging.info("DataFrame processado e limpo com sucesso.")

        # 3. Salvar Resultados (JSON e TXT)
        colunas_para_salvar = [
            'id', 'fonte_coleta', 'post_url', 'usuario', 'conteudo',
            'data_hora', 'parent_url', 'run_id',
            'sentimento_final', 
            'genero_final',
            # --- <<< INÍCIO DA MUDANÇA: SALVANDO COLUNAS PROCESSADAS >>> ---
            'emojis',
            'texto_puro',
            'tamanho_comentario'
            # --- <<< FIM DA MUDANÇA >>> ---
        ]
        colunas_existentes = [col for col in colunas_para_salvar if col in df_processed.columns]

        if not df_processed.empty:
            try:
                df_to_save = df_processed[colunas_existentes].copy()
                
                # Salva como JSON (records orient)
                # (date_format='iso' é CRÍTICO para o pandas ler de volta)
                df_to_save.to_json(json_path, orient='records', lines=False, indent=4, force_ascii=False, date_format='iso') 
                logging.info(f"Arquivo JSON salvo em: {json_path}")
            except Exception as e_json:
                 logging.error(f"Erro ao salvar arquivo JSON: {e_json}")
        else:
             logging.warning("Nenhum comentário válido para salvar no JSON após processamento.")
             with open(json_path, 'w', encoding='utf-8') as f_json: json.dump([], f_json)

        # Salvar Resumo TXT
        if not df_processed.empty:
             resumo = f"Resumo do Processamento para Run ID: {run_id}\n"
             resumo += f"Perfil: {db_basename}\n"
             resumo += f"Total de comentários processados: {len(df_processed)}\n\n"
             resumo += "Contagem de Sentimentos (Final):\n"
             resumo += df_processed['sentimento_final'].value_counts().to_string() + "\n\n"
             resumo += "Contagem de Gênero (Final):\n"
             resumo += df_processed['genero_final'].value_counts().to_string()
        else:
             resumo = f"Nenhum comentário válido processado para Run ID: {run_id}"

        try:
            with open(txt_path, 'w', encoding='utf-8') as f_txt:
                f_txt.write(resumo)
            logging.info(f"Arquivo TXT de resumo salvo em: {txt_path}")
        except Exception as e_txt:
            logging.error(f"Erro ao salvar arquivo TXT: {e_txt}")

    except pd.errors.DatabaseError as e_sql:
         logging.error(f"Erro de SQL ao ler dados do Run ID {run_id}: {e_sql}")
    except Exception as e:
        logging.error(f"Erro GERAL durante o processamento do Run ID {run_id}: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()
            logging.info("Conexão com o banco de dados fechada.")

# --- Ponto de Entrada do Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processa dados coletados de um run específico no banco de dados, gerando JSON e TXT.")
    parser.add_argument("--db-path", required=True, help="Caminho para o arquivo do banco de dados SQLite.")
    parser.add_argument("--run-id", required=True, help="O Run ID específico a ser processado.")
    args = parser.parse_args()

    if not os.path.exists(args.db_path):
        abs_db_path = os.path.abspath(args.db_path)
        logging.warning(f"DB não encontrado em '{args.db_path}'. Tentando caminho absoluto: '{abs_db_path}'")
        if not os.path.exists(abs_db_path):
             logging.error(f"Erro: Banco de dados não encontrado em '{args.db_path}' nem '{abs_db_path}'")
             sys.exit(1)
        else:
             args.db_path = abs_db_path 

    processar_run(args.db_path, args.run_id)

    logging.info("Script de processamento finalizado.")