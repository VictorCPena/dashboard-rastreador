import os
import sqlite3
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer
import logging
import sys
import argparse
from typing import List, Dict, Optional, Tuple, Any
from src.utils.config import CONFIG

def carregar_modelo_local_keras(model_path: str) -> Tuple[Optional[Any], Optional[Any]]:
    """Carrega um modelo Keras e seu tokenizer a partir de um caminho local."""
    try:
        logging.info(f"Carregando tokenizer e modelo Keras de: '{model_path}'...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = tf.keras.models.load_model(model_path)
        logging.info("Modelo e tokenizer carregados com sucesso!")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Erro ao carregar modelo local de '{model_path}': {e}", exc_info=True)
        return None, None

def carregar_dados_para_analise(db_path: str, run_id: str) -> pd.DataFrame:
    """Carrega todos os dados da execução para serem re-analisados."""
    try:
        conn = sqlite3.connect(db_path)
        query = f"SELECT id, conteudo, usuario FROM posts WHERE run_id = '{run_id}'"
        df = pd.read_sql_query(query, conn)
        conn.close()
        logging.info(f"Encontrados {len(df)} posts/comentários da execução '{run_id}' para analisar/re-analisar.")
        return df
    except sqlite3.Error as e:
        logging.error(f"Erro de banco de dados ao carregar dados: {e}")
        return pd.DataFrame()

def prever_em_lote(textos: List[str], model, tokenizer, label_map: Dict[int, str], max_length: int) -> List[Optional[str]]:
    """Função de previsão inteligente que se adapta às entradas do modelo."""
    if not textos or model is None or tokenizer is None:
        return [None] * len(textos)
    try:
        inputs = tokenizer(textos, padding='max_length', truncation=True, return_tensors="tf", max_length=max_length)
        
        # Constrói um DICIONÁRIO de inputs, que é mais robusto
        model_inputs = {}
        for key in model.input_names:
            if key in inputs:
                model_inputs[key] = inputs[key]
        
        logits = model.predict(model_inputs)
        predicted_classes_indices = np.argmax(logits, axis=1)
        
        return [label_map.get(int(idx)) for idx in predicted_classes_indices]
    except Exception as e:
        logging.error(f"Erro durante a previsão em lote: {e}", exc_info=True)
        return [None] * len(textos)

def atualizar_banco(df: pd.DataFrame, db_path: str):
    """Atualiza o banco de dados com os resultados da previsão."""
    if df.empty: return
    try:
        conn = sqlite3.connect(db_path)
        update_query = "UPDATE posts SET sentimento = ?, genero_previsto = ? WHERE id = ?"
        df_update = df.dropna(subset=['id'])
        update_data = [(row.get("sentimento"), row.get("genero_previsto"), row["id"]) for _, row in df_update.iterrows()]
        conn.executemany(update_query, update_data)
        conn.commit()
        conn.close()
        logging.info(f"{len(df_update)} registros atualizados no banco de dados.")
    except sqlite3.Error as e:
        logging.error(f"Erro de banco de dados ao atualizar registros: {e}")

def main():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] [%(asctime)s] - %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--db-path', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--profile-name', required=True)
    args = parser.parse_args()

    modelo_sent, tokenizer_sent = carregar_modelo_local_keras(CONFIG['paths']['sentiment_model'])
    modelo_gen, tokenizer_gen = carregar_modelo_local_keras(CONFIG['paths']['gender_model'])
    
    if not all([modelo_sent, tokenizer_sent, modelo_gen, tokenizer_gen]):
        sys.exit(1)
        
    df_para_analise = carregar_dados_para_analise(args.db_path, args.run_id)
    if df_para_analise.empty: return

    # Mapas de labels corretos, baseados nos seus scripts de treino
    mapa_sentimento = {0: 'neu', 1: 'pos', 2: 'neg'}
    mapa_genero = {0: 'masculino', 1: 'feminino', 2: 'indeterminado'}
    
    logging.info("Analisando sentimento dos conteúdos com seu modelo local...")
    df_para_analise['sentimento'] = prever_em_lote(
        df_para_analise["conteudo"].fillna("").tolist(), modelo_sent, tokenizer_sent, mapa_sentimento, max_length=128
    )
    
    logging.info("Prevendo gênero dos usuários com seu modelo local...")
    df_para_analise['genero_previsto'] = prever_em_lote(
        df_para_analise["usuario"].fillna("").tolist(), modelo_gen, tokenizer_gen, mapa_genero, max_length=32
    )
    
    atualizar_banco(df_para_analise, args.db_path)
    logging.info(f"Análise da execução '{args.run_id}' concluída com sucesso.")

if __name__ == '__main__':
    main()