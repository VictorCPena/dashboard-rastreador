# src/utils/config.py
import os
import sys
import yaml
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

def get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_config() -> Dict[str, Any]:
    root = get_project_root()
    config_path = os.path.join(root, "config.yaml")

    if not os.path.exists(config_path):
        print(f"ERRO CRÍTICO: 'config.yaml' não encontrado em: {root}")
        sys.exit(1)

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"ERRO CRÍTICO: 'config.yaml' tem um erro de formatação.")
        print(f"Detalhes: {e}")
        sys.exit(1)

    for key, path in config_data.get("paths", {}).items():
        abs_path = os.path.join(root, path)
        config_data["paths"][key] = abs_path
        if "output" in key or "db" in key:
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            
    return config_data

def get_env_variable(var_name: str) -> str:
    value = os.environ.get(var_name)
    if value is None:
        raise ValueError(f"Variável de ambiente '{var_name}' não definida no .env")
    return value

try:
    CONFIG = load_config()
    GEMINI_API_KEY = get_env_variable("GEMINI_API_KEY")
    GMAIL_APP_PASSWORD = get_env_variable("GMAIL_APP_PASSWORD")
except (ValueError, FileNotFoundError) as e:
    print(f"ERRO CRÍTICO NA CONFIGURAÇÃO: {e}")
    sys.exit(1)