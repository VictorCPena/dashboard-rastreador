# run_all.py (Com Processamento Automático e Metadados)
import argparse
import logging
import subprocess
import sys
import os
import json # <<< NOVO IMPORT
from datetime import datetime
from src.utils.config import CONFIG # Importa CONFIG

PYTHON_EXECUTABLE = sys.executable
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Assume que run_all.py está na raiz
SRC_DIR = os.path.join(BASE_DIR, "src")
DATA_DIR = os.path.join(BASE_DIR, "dados") # <<< NOVO CAMINHO

# Garante que a pasta 'dados' exista
os.makedirs(DATA_DIR, exist_ok=True)

def run_script(script_path: str, *args: str) -> bool: # <<< MUDANÇA: Retorna True/False >>>
    """Executa um script Python como módulo, tratando erros. Retorna True se sucesso, False se falha."""
    script_name = os.path.basename(script_path)
    try:
        # Tenta construir o path do módulo relativo a BASE_DIR
        module_path = os.path.relpath(script_path, BASE_DIR).replace('.py', '').replace(os.sep, '.')
    except ValueError:
        # Fallback se relpath falhar (ex: caminhos em drives diferentes no Windows)
        # Remove BASE_DIR e a barra inicial, depois substitui separadores
        relative_part = script_path.replace(BASE_DIR, '').lstrip(os.sep)
        module_path = relative_part.replace('.py', '').replace(os.sep, '.')
        logging.warning(f"Usando path alternativo para módulo: {module_path}")

    command = [PYTHON_EXECUTABLE, "-m", module_path] + list(args)
    logging.info(f"Executando: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True, cwd=BASE_DIR, capture_output=True, text=True, encoding='utf-8')
        logging.info(f"Script '{script_name}' concluído com sucesso.")
        if result.stderr:
             logging.warning(f"Avisos de {script_name}:\n{result.stderr[-500:]}") # Mostra últimos 500 chars de warnings
        return True # <<< MUDANÇA >>>
    except subprocess.CalledProcessError as e:
        logging.error(f"O script '{script_name}' FALHOU com código {e.returncode}.")
        if e.stdout: logging.error(f"Saída (stdout):\n{e.stdout}")
        if e.stderr: logging.error(f"Erro (stderr):\n{e.stderr}")
        return False # <<< MUDANÇA >>>
    except FileNotFoundError:
        logging.error(f"Erro: O executável Python '{PYTHON_EXECUTABLE}' ou o módulo '{module_path}' não foi encontrado.")
        return False # <<< MUDANÇA >>>
    except Exception as e:
        logging.error(f"Erro inesperado ao executar {script_name}: {e}")
        return False # <<< MUDANÇA >>>

# --- <<< INÍCIO DA MUDANÇA: FUNÇÃO DE METADADOS >>> ---
def save_run_metadata(
    profile_name: str, 
    run_id: str, 
    twitter_profiles: list, 
    twitter_keywords: list, 
    instagram_profiles: list, 
    instagram_tags: list
):
    """
    Salva um mapa de run_id para suas keywords/tags em um arquivo JSON.
    Isso permite que o dashboard mostre um nome "amigável".
    """
    
    terms = []
    if twitter_profiles:
        terms.append(f"Twitter (Perfis): {', '.join(twitter_profiles)}")
    if twitter_keywords:
        terms.append(f"Twitter (Keywords): {', '.join(twitter_keywords)}")
    if instagram_profiles:
        terms.append(f"Instagram (Perfis): {', '.join(instagram_profiles)}")
    if instagram_tags:
        # O "tags" do instagram são as "keywords" de busca
        terms.append(f"Instagram (Keywords): {', '.join(instagram_tags)}")
    
    if not terms:
        logging.info("Nenhum alvo de coleta especificado, pulando salvamento de metadados.")
        return

    friendly_desc = " | ".join(terms)
    metadata_path = os.path.join(DATA_DIR, "run_metadata.json")
    
    metadata = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except json.JSONDecodeError:
            logging.warning(f"Arquivo de metadados '{metadata_path}' corrompido. Criando um novo.")
            metadata = {}
    
    if profile_name not in metadata:
        metadata[profile_name] = {}
        
    metadata[profile_name][run_id] = friendly_desc
    
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logging.info(f"Metadados salvos para Run ID '{run_id}' ({friendly_desc}) em {metadata_path}")
    except Exception as e:
        logging.warning(f"Falha ao salvar metadados do Run ID: {e}")
# --- <<< FIM DA MUDANÇA >>> ---


def main():
    log_format = '[%(levelname)s] [%(asctime)s] - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[logging.StreamHandler(sys.stdout)])

    parser = argparse.ArgumentParser(description="Pipeline de COLETA, ANÁLISE BÁSICA e PROCESSAMENTO de mídias sociais.")
    parser.add_argument('--profile', type=str, help='Nome de um perfil de monitoramento definido no config.yaml.')
    parser.add_argument('--plataforma', type=str, choices=['instagram', 'twitter', 'todas'], default='todas', help='Especifica qual coletor rodar.')
    parser.add_argument('--twitter-profiles', nargs='*', default=[])
    parser.add_argument('--twitter-keywords', nargs='*', default=[])
    parser.add_argument('--instagram-profiles', nargs='*', default=[])
    parser.add_argument('--instagram-tags', nargs='*', default=[])
    parser.add_argument('--chrome-profile-path', help='Caminho perfil Chrome (sobrescreve config).')
    parser.add_argument('--headless', action='store_true', help="Executar headless.")
    args = parser.parse_args()

    # --- Determina Chrome Path ---
    chrome_path = args.chrome_profile_path or CONFIG.get('settings', {}).get('chrome_profile_path')
    if not chrome_path or not os.path.exists(chrome_path):
        logging.error(f"Caminho do perfil Chrome ('{chrome_path}') inválido ou não encontrado."); sys.exit(1)
    logging.info(f"Usando perfil Chrome: {chrome_path}")

    # --- Determina alvos da coleta ---
    profile_name = "coleta_manual"; twitter_profiles = args.twitter_profiles; twitter_keywords = args.twitter_keywords
    instagram_profiles = args.instagram_profiles; instagram_tags = args.instagram_tags
    if args.profile:
        profile_name = args.profile; logging.info(f"Usando perfil: '{profile_name}'")
        profile_data = CONFIG.get('monitor_profiles', {}).get(profile_name)
        if not profile_data: logging.error(f"Perfil '{profile_name}' não encontrado!"); sys.exit(1)
        # Permite sobrescrever/adicionar alvos via linha de comando
        twitter_profiles.extend(profile_data.get('twitter_profiles', [])); twitter_keywords.extend(profile_data.get('twitter_keywords', []))
        instagram_profiles.extend(profile_data.get('instagram_profiles', [])); instagram_tags.extend(profile_data.get('instagram_tags', []))
    
    # Remove duplicatas
    twitter_profiles = list(set(twitter_profiles)); twitter_keywords = list(set(twitter_keywords))
    instagram_profiles = list(set(instagram_profiles)); instagram_tags = list(set(instagram_tags))


    # --- Define DB path e Run ID ---
    db_name = f"{profile_name}.db"; db_path = os.path.join(DATA_DIR, db_name) # <<< MUDANÇA: Usa DATA_DIR
    run_id = datetime.now().strftime('%Y%m%d-%H%M%S')
    logging.info(f"BANCO DE DADOS: {db_path}"); logging.info(f"ID DA EXECUÇÃO (RUN_ID): {run_id}")
    
    # --- <<< INÍCIO DA MUDANÇA: SALVA METADADOS (passa todos os argumentos) >>> ---
    save_run_metadata(
        profile_name, run_id, 
        twitter_profiles, twitter_keywords, 
        instagram_profiles, instagram_tags
    )
    # --- <<< FIM DA MUDANÇA >>> ---

    has_twitter_targets = twitter_profiles or twitter_keywords; has_instagram_targets = instagram_profiles or instagram_tags 
    if not has_twitter_targets and not has_instagram_targets: logging.error("Nenhum alvo de coleta válido."); sys.exit(1)

    # --- ETAPA 1: COLETA ---
    logging.info("====================== INICIANDO ETAPA 1: COLETA DE DADOS ======================")
    common_selenium_args = ['--profile-path', chrome_path]; collector_db_args = ['--db-path', db_path, '--run-id', run_id]
    if args.headless: common_selenium_args.append('--headless')
    run_twitter = (args.plataforma in ['twitter', 'todas']) and has_twitter_targets
    run_instagram = (args.plataforma in ['instagram', 'todas']) and has_instagram_targets
    coleta_success = True 

    if run_twitter:
        script = os.path.join(SRC_DIR, "coletores", "coletor_twitter.py"); cmd = common_selenium_args + collector_db_args
        if twitter_profiles: cmd.extend(['--profiles'] + twitter_profiles)
        if twitter_keywords: cmd.extend(['--keywords'] + twitter_keywords)
        if not run_script(script, *cmd): coleta_success = False 
    if run_instagram and coleta_success: 
        script = os.path.join(SRC_DIR, "coletores", "coletor_instagram.py"); cmd = common_selenium_args + collector_db_args
        if instagram_profiles: cmd.extend(['--profiles'] + instagram_profiles)
        if instagram_tags: cmd.extend(['--tags'] + instagram_tags)
        if not run_script(script, *cmd): coleta_success = False

    if not run_twitter and not run_instagram: logging.warning("Nenhum coletor executado."); sys.exit(0)
    if not coleta_success:
         logging.critical("Pipeline interrompido devido a erro na ETAPA 1: COLETA.")
         sys.exit(1)

    # --- ETAPA 2: ANÁLISE LOCAL ---
    logging.info("====================== INICIANDO ETAPA 2: ANÁLISE SENTIMENTO/GÊNERO ======================")
    script_analise = os.path.join(SRC_DIR, "analise", "analisar_sentimentos.py")
    analise_success = run_script(script_analise, '--db-path', db_path, '--run-id', run_id, '--profile-name', profile_name) 

    if not analise_success:
        logging.critical("Pipeline interrompido devido a erro na ETAPA 2: ANÁLISE LOCAL.")
        sys.exit(1)

    # --- ETAPA 3: PROCESSAMENTO FINAL ---
    logging.info("====================== INICIANDO ETAPA 3: PROCESSAMENTO FINAL (JSON/TXT) ======================")
    script_processamento = os.path.join(SRC_DIR, "analise", "processar_dados.py") 
    processamento_success = run_script(script_processamento, '--db-path', db_path, '--run-id', run_id)

    if not processamento_success:
        logging.critical("Pipeline interrompido devido a erro na ETAPA 3: PROCESSAMENTO FINAL.")
        sys.exit(1)

    logging.info("======================== PIPELINE COMPLETO ========================")
    logging.info(f"Run ID finalizado: {run_id}. Dados salvos em {db_path}")
    logging.info(f"Resultados processados (JSON/TXT) salvos em: {os.path.join(BASE_DIR, 'relatorios_processados')}")


if __name__ == '__main__':
    main()