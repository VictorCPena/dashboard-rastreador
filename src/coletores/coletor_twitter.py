import time
import sqlite3
import logging
import argparse
import os
# --- NOVO IMPORT ADICIONADO ---
from urllib.parse import quote
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from typing import List, Set
# Esta é a importação que busca seu config.yaml
from src.utils.config import CONFIG 
from datetime import datetime

class TwitterCollector:
    def __init__(self, profile_path: str, db_path: str, run_id: str, headless: bool = False):
        self.db_path = db_path
        self.run_id = run_id
        self._init_db()
        options = Options()
        options.add_argument(f"user-data-dir={profile_path}")
        options.add_argument('--window-size=1920,1080')
        if headless:
            options.add_argument('--headless=new')
            options.add_argument('--disable-gpu')
        self.driver = webdriver.Chrome(options=options)
        logging.info(f"TwitterCollector iniciado (headless={headless}) usando perfil: {profile_path}") # Log com path

    # --- <<< INÍCIO DA CORREÇÃO >>> ---
    def _init_db(self):
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        # Garante que a tabela tenha as colunas de sentimento/gênero,
        # correspondendo ao schema do coletor_instagram.py
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT, fonte_coleta TEXT, post_url TEXT,
                usuario TEXT, conteudo TEXT, data_hora TEXT, parent_url TEXT,
                run_id TEXT,
                sentimento TEXT,
                genero_previsto TEXT
            )""")
        
        # Adiciona o índice único (necessário para o INSERT OR IGNORE do Twitter)
        # de forma separada e segura (IF NOT EXISTS)
        # Captura tanto OperationalError (ex: índice já existe) quanto
        # IntegrityError (ex: dados antigos duplicados)
        try:
            self.cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_post_url_unique ON posts (post_url)")
        except (sqlite3.OperationalError, sqlite3.IntegrityError) as e:
            logging.warning(f"Não foi possível criar/aplicar índice único: {e}")
            logging.warning("Isso pode ocorrer se o DB antigo continha duplicatas. O script continuará usando 'INSERT OR IGNORE'.")
            
        self.conn.commit()
    # --- <<< FIM DA CORREÇÃO >>> ---


    # --- MÉTODO 'COLLECT' ATUALIZADO ---
    def collect(self, profiles: List[str], keywords: List[str]):
        # Usa o CONFIG importado
        cfg = CONFIG['coleta']['twitter'] 
        
        # Parte dos perfis (continua igual)
        for profile in profiles:
            url = f"https://x.com/{profile.lstrip('@')}"
            self._process_url(url, origem=f"Twitter:perfil:{profile}", cfg=cfg, scrape_comments=True)
        
        # Parte das keywords (modificada para incluir filtros)
        for term in keywords:
            # 1. Definimos os filtros avançados que você pediu
            advanced_filters = " lang:pt -filter:links -filter:replies"
            
            # 2. Criamos a query de busca completa (ex: "flamengo lang:pt -filter:links...")
            full_query = f"{term}{advanced_filters}"
            
            # 3. Codificamos a query inteira para a URL (importante para os espaços e :)
            encoded_query = quote(full_query)

            # 4. Montamos a URL final igual à que você passou como exemplo
            url = f"https://x.com/search?q={encoded_query}&f=live&src=typed_query"
            
            logging.info(f"URL de busca (keyword) gerada: {url}") # Log extra para confirmar
            
            self._process_url(url, origem=f"Twitter:keyword:{term}", cfg=cfg, scrape_comments=False)
        
        self.conn.commit()
    # --- FIM DA ATUALIZAÇÃO DO 'COLLECT' ---

    def _process_url(self, url: str, origem: str, cfg: dict, scrape_comments: bool):
        logging.info(f"Coletando links de: {url} (Comentários: {scrape_comments})")
        try:
            self.driver.get(url)
            WebDriverWait(self.driver, cfg.get('timeout', 10)).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "article[data-testid='tweet']"))
            )
        except TimeoutException:
            logging.warning(f"Timeout inicial ao carregar {url}. Nenhum post encontrado.")
            return 

        urls_para_visitar = set()
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        
        max_scrolls = cfg.get('max_scrolls', 10) 
        scroll_delay = cfg.get('scroll_delay', 2.5) 
        
        for i in range(max_scrolls):
            artigos = self.driver.find_elements(By.CSS_SELECTOR, "article[data-testid='tweet']")
            new_urls_found_this_scroll = 0
            for art in artigos:
                try:
                    post_link_element = art.find_element(By.XPATH, ".//time/ancestor::a")
                    post_url = post_link_element.get_attribute('href')
                    if post_url and post_url not in urls_para_visitar:
                        urls_para_visitar.add(post_url)
                        new_urls_found_this_scroll += 1
                except Exception: 
                    continue
                
            self.driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
            time.sleep(scroll_delay)
            
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height and new_urls_found_this_scroll < 5: 
                logging.info(f"Rolagem {i+1}/{max_scrolls}: Parando (poucos posts novos/altura estável).")
                break 
            last_height = new_height
            logging.debug(f"Rolagem {i+1}/{max_scrolls} OK. Posts únicos: {len(urls_para_visitar)}")

        logging.info(f"Encontrados {len(urls_para_visitar)} posts únicos. Iniciando coleta de detalhes.")
        for post_url in list(urls_para_visitar):
            self._scrape_post_and_optionally_comments(post_url, origem, cfg, scrape_comments)

    def _scrape_post_and_optionally_comments(self, post_url: str, origem: str, cfg: dict, scrape_comments: bool):
        try:
            logging.debug(f"Processando post: {post_url}") 
            self.driver.get(post_url)
            wait = WebDriverWait(self.driver, cfg.get('timeout', 10))
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "article[data-testid='tweet']")))
            time.sleep(3) 

            all_articles = self.driver.find_elements(By.CSS_SELECTOR, "article[data-testid='tweet']")
            
            if all_articles:
                post_principal = all_articles[0]
                post_saved = False 
                try:
                    texto = post_principal.find_element(By.CSS_SELECTOR, "div[data-testid='tweetText']").text.strip()
                    usuario = post_url.split('/')[3] 
                    timestamp = post_principal.find_element(By.TAG_NAME, 'time').get_attribute('datetime')
                    
                    self.cursor.execute(
                        "INSERT OR IGNORE INTO posts (fonte_coleta, post_url, usuario, conteudo, data_hora, run_id) VALUES (?, ?, ?, ?, ?, ?)",
                        (origem, post_url, usuario, texto, timestamp, self.run_id)
                    ) 
                    if self.cursor.rowcount > 0: post_saved = True
                        
                except Exception as e:
                    logging.warning(f"Não processou post principal {post_url}: {e}")

                comment_count = 0 
                if scrape_comments:
                    comentarios = all_articles[1:] 
                    if comentarios:
                        logging.debug(f"Coletando {len(comentarios)} comentários para {post_url}")
                        for comment in comentarios:
                            try:
                                comment_text = comment.find_element(By.CSS_SELECTOR, "div[data-testid='tweetText']").text.strip()
                                if not comment_text: continue 

                                user_element = comment.find_element(By.CSS_SELECTOR, "div[data-testid='User-Name'] a")
                                comment_user = user_element.get_attribute('href').split('/')[-1]
                                
                                time_element = comment.find_element(By.TAG_NAME, 'time')
                                comment_timestamp = time_element.get_attribute('datetime')
                                comment_url_element = time_element.find_element(By.XPATH, "./ancestor::a")
                                comment_url = comment_url_element.get_attribute('href')

                                self.cursor.execute(
                                    "INSERT OR IGNORE INTO posts (fonte_coleta, post_url, usuario, conteudo, data_hora, parent_url, run_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                                    (origem, comment_url, comment_user, comment_text, comment_timestamp, post_url, self.run_id)
                                ) 
                                if self.cursor.rowcount > 0: comment_count += 1
                                    
                            except Exception as e_comment:
                                logging.warning(f"Não processou um comentário: {e_comment}")
                                continue 
                        logging.debug(f"Salvos {comment_count} novos comentários para {post_url}")
                    else:
                        logging.debug(f"Nenhum comentário visível para {post_url}") 
                else:
                    logging.debug(f"Coleta de comentários pulada para {post_url}") 
                
                if post_saved or comment_count > 0:
                    self.conn.commit() 

        except TimeoutException:
             logging.warning(f"Timeout ao carregar PÁGINA DO POST {post_url}. Pulando este post.")
        except Exception as e:
            logging.error(f"Falha GERAL ao processar página do post {post_url}. Erro: {e}")

    def close(self):
        if self.driver: self.driver.quit()
        if self.conn: self.conn.close()
        logging.info("Coletor do Twitter finalizado.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] [%(asctime)s] - %(message)s')

    # --- Bloco de Leitura do Config.yaml ---
    try:
        # Ajuste esta chave se a estrutura do seu config for diferente
        default_profile_path = CONFIG['coleta']['twitter']['chrome_profile_path']
        logging.info(f"Caminho do perfil do Chrome carregado do config.yaml: {default_profile_path}")
    except KeyError:
        logging.warning("Chave 'chrome_profile_path' não encontrada no config.yaml. O caminho deve ser fornecido via argumento.")
        default_profile_path = None

    parser = argparse.ArgumentParser(description='Coletor de Dados do Twitter.')
    parser.add_argument('--db-path', help='Caminho para o banco de dados.')
    parser.add_argument('--run-id', help='ID único para esta execução.')
    
    # Argumento '--profile-path' agora usa o 'default' do config.yaml
    parser.add_argument('--profile-path', 
                        default=default_profile_path, 
                        help='Caminho para a pasta de perfil do Chrome. (Padrão: pego do config.yaml)')
    
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--profiles', nargs='*', default=[])
    parser.add_argument('--keywords', nargs='*', default=[])
    args = parser.parse_args()
    
    # Verificação para garantir que o caminho do perfil existe
    if not args.profile_path:
        logging.error("ERRO: O caminho do perfil do Chrome não foi encontrado no config.yaml nem fornecido via --profile-path.")
        logging.error("Execute com: --profile-path \"/caminho/do/perfil\" ou adicione 'chrome_profile_path' no seu config.yaml")
        exit(1) # Para o script se não houver caminho

    db_path = args.db_path or 'dados/teste_twitter.db'
    run_id = args.run_id or f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    os.makedirs(os.path.dirname(db_path), exist_ok=True) 
    
    logging.info(f"Iniciando execução com ID: {run_id}")
    logging.info(f"Salvando dados em: {db_path}")

    collector = None # Inicializa para garantir o 'finally'
    try:
        collector = TwitterCollector(profile_path=args.profile_path, db_path=db_path, run_id=run_id, headless=args.headless)
        collector.collect(profiles=args.profiles, keywords=args.keywords)
        logging.info(f"Coleta (run_id: {run_id}) finalizada com sucesso.")
    
    except Exception as e:
        logging.error(f"Erro inesperado na coleta (run_id: {run_id}): {e}")
    
    finally:
        # Bloco de limpeza: fecha o driver e a conexão
        if collector:
            collector.close()
        logging.info("Script encerrado.")