import time
import sqlite3
import logging
import argparse
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from typing import List, Set, Dict # <--- ALTERAÇÃO: Importar Dict
from src.utils.config import CONFIG
from datetime import datetime

class InstagramCollector:
    def __init__(self, profile_path: str, db_path: str, run_id: str, headless: bool = False):
        self.db_path = db_path
        self.run_id = run_id
        self._init_db()
        options = Options()
        options.add_argument(f"user-data-dir={profile_path}")
        options.add_argument('--window-size=1920,1080')
        options.add_argument("--mute-audio")
        options.add_experimental_option("prefs", {"profile.default_content_setting_values.notifications": 2})
        if headless:
            options.add_argument('--headless=new')
            options.add_argument('--disable-gpu')
        self.driver = webdriver.Chrome(options=options)
        logging.info(f"InstagramCollector iniciado (headless={headless})")

    def _init_db(self):
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT, fonte_coleta TEXT, post_url TEXT,
                usuario TEXT, conteudo TEXT, data_hora TEXT, parent_url TEXT,
                run_id TEXT,
                sentimento TEXT,
                genero_previsto TEXT
            )""")
        self.conn.commit()

    def collect(self, profiles: List[str], tags: List[str]):
        cfg = CONFIG['coleta']['instagram']
        tags_lower = [tag.lower() for tag in tags]
        for profile in profiles:
            logging.info(f"Iniciando coleta para o perfil do Instagram: {profile}")
            
            # <--- ALTERAÇÃO: Agora recebe um dicionário {url: alt_text}
            posts_data = self._get_post_urls(profile, tags_lower, cfg) 
            
            if not posts_data:
                logging.warning(f"Nenhum post encontrado para '{profile}' com os critérios fornecidos.")
                continue
            
            logging.info(f"Encontrados {len(posts_data)} posts para '{profile}'. Coletando detalhes...")
            
            # <--- ALTERAÇÃO: Itera sobre os itens do dicionário
            for url, alt_text_caption in posts_data.items():
                # <--- ALTERAÇÃO: Passa o alt_text (legenda) para o método de scraping
                self._scrape_post_and_comments(url, profile, alt_text_caption)
                
        self.conn.commit()

    # <--- ALTERAÇÃO: O método agora retorna um Dicionário [url, alt_text]
    def _get_post_urls(self, profile: str, tags: List[str], cfg: dict) -> Dict[str, str]:
        profile_url = f"https://www.instagram.com/{profile}/"
        self.driver.get(profile_url)
        time.sleep(5)
        
        posts_encontrados = {} # <--- ALTERAÇÃO: De set() para dict {}
        
        for i in range(cfg['scrolls_perfil']):
            post_links = self.driver.find_elements(By.XPATH, "//a[descendant::img[@alt]]")
            for link in post_links:
                try:
                    href = link.get_attribute('href')
                    if not href or ('/p/' not in href and '/reel/' not in href): continue
                    if href in posts_encontrados: continue # <--- ALTERAÇÃO: Checa no dict

                    # <--- ALTERAÇÃO: Pega o alt_text ANTES de verificar as tags
                    img = link.find_element(By.TAG_NAME, 'img')
                    alt_text = (img.get_attribute('alt') or "").strip() # Pega o texto exato
                    
                    if not tags: 
                        posts_encontrados[href] = alt_text # <--- ALTERAÇÃO: Salva URL e alt_text
                    else:
                        alt_text_lower = alt_text.lower() # <--- ALTERAÇÃO: Gera o lower só para o filtro
                        if any(tag in alt_text_lower for tag in tags):
                            posts_encontrados[href] = alt_text # <--- ALTERAÇÃO: Salva URL e alt_text
                            
                    if len(posts_encontrados) >= cfg['max_posts_por_perfil']: 
                        return posts_encontrados # <--- ALTERAÇÃO: Retorna o dict
                        
                except NoSuchElementException: continue
                
            self.driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
            logging.info(f"Scroll {i+1}/{cfg['scrolls_perfil']} no perfil '{profile}'. Posts encontrados: {len(posts_encontrados)}")
            time.sleep(cfg['scroll_delay'])
            
        return posts_encontrados # <--- ALTERAÇÃO: Retorna o dict

    # <--- ALTERAÇÃO: Adicionado novo parâmetro 'alt_text_caption'
    def _scrape_post_and_comments(self, post_url: str, profile: str, alt_text_caption: str):
        try:
            self.driver.get(post_url)
            wait = WebDriverWait(self.driver, CONFIG['coleta']['instagram']['timeout'])
            
            # <--- ALTERAÇÃO: Usa o alt_text recebido como legenda
            conteudo = alt_text_caption 
            
            username = profile
            timestamp = ""
            
            try:
                # <--- ALTERAÇÃO: Este bloco agora só busca o username e o timestamp.
                # A busca pela legenda (conteudo) foi removida.
                
                main_block = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div[class='html-div xdj266r x14z9mp xat24cr x1lziwak xyri2b x1c1uobl x9f619 xjbqb8w x78zum5 x15mokao x1ga7v0g x16uus16 xbiv7yw xsag5q8 xz9dl7a x1uhb9sk x1plvlek xryxfnj x1c4vz4f x2lah0s xdt5ytf xqjyukv x1qjc9v5 x1oa3qoh x1nhvcw1']")))
                username = main_block.find_element(By.CSS_SELECTOR, "a[class='_ap3a _aaco _aacw _aacx _aad7 _aade']").text.strip()
                timestamp = self.driver.find_element(By.CSS_SELECTOR,'[class="x1ejq31n x18oe1m7 x1sy0etr xstzfhl x1roi4f4 xexx8yu xyri2b x18d9i69 x1c1uobl x1n2onr6"]').get_attribute('datetime')
                
                # A lógica antiga de buscar 'text_block' e 'caption_parts' foi removida
                
                logging.info(f"Username e Timestamp encontrados para o post {post_url}.")
                
            except (TimeoutException, NoSuchElementException):
                logging.warning(f"Não foi possível encontrar username/timestamp para o post {post_url}. Usando dados de fallback.")
                # <--- ALTERAÇÃO: Fallback para o timestamp se falhar
                if not timestamp:
                    timestamp = datetime.now().isoformat()

            fonte = f"Instagram:perfil:{profile}"
            
            # <--- ALTERAÇÃO: Salva o post usando o 'conteudo' (alt_text) recebido
            if username and timestamp:
                self.cursor.execute(
                    "INSERT INTO posts (fonte_coleta, post_url, usuario, conteudo, data_hora, run_id) VALUES (?, ?, ?, ?, ?, ?)",
                    (fonte, post_url, username, conteudo, timestamp, self.run_id)
                )

            # O restante do código (coleta de comentários) permanece o mesmo
            try:
                comment_area_selector = "x5yr21d.xw2csxc.x1odjw0f.x1n2onr6"
                comment_area = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, f"div.{comment_area_selector.replace(' ', '.')}")))
                logging.info(f"Área de comentários encontrada. Rolando para carregar...")
                for _ in range(5):
                    self.driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", comment_area)
                    time.sleep(2)
            except (TimeoutException, NoSuchElementException):
                logging.warning(f"Não foi possível encontrar a área de comentários para rolar em {post_url}.")

            comment_elements = self.driver.find_elements(By.CSS_SELECTOR, "[class='html-div xdj266r x14z9mp xat24cr x1lziwak xexx8yu xyri2b x18d9i69 x1c1uobl x9f619 xjbqb8w x78zum5 x15mokao x1ga7v0g x16uus16 xbiv7yw x1uhb9sk x1plvlek xryxfnj x1iyjqo2 x2lwn1j xeuugli x1q0g3np xqjyukv x1qjc9v5 x1oa3qoh x1nhvcw1']")
            logging.info(f"Encontrados {len(comment_elements)} elementos de comentários (pode incluir a legenda).")
            
            comments_saved = 0
            for comment_block in comment_elements:
                try:
                    comment_user = comment_block.find_element(By.CSS_SELECTOR, '[class="html-div xdj266r x14z9mp xat24cr x1lziwak xexx8yu xyri2b x18d9i69 x1c1uobl x9f619 xjbqb8w x78zum5 x15mokao x1ga7v0g x16uus16 xbiv7yw x1n2onr6 x1plvlek xryxfnj x1c4vz4f x2lah0s x1q0g3np xqjyukv x6s0dn4 x1oa3qoh x1nhvcw1"]').text.strip()
                    if comment_user == username and comments_saved == 0: continue
                    comment_text = comment_block.find_element(By.CSS_SELECTOR, '[class="html-div xdj266r x14z9mp xat24cr x1lziwak xexx8yu xyri2b x18d9i69 x1c1uobl x9f619 xjbqb8w x78zum5 x15mokao x1ga7v0g x16uus16 xbiv7yw x1uhb9sk x1plvlek xryxfnj x1c4vz4f x2lah0s xdt5ytf xqjyukv x1cy8zhl x1oa3qoh x1nhvcw1"]').text.strip()
                    datetime_str = comment_block.find_element(By.CSS_SELECTOR, '[class="x1ejq31n x18oe1m7 x1sy0etr xstzfhl x1roi4f4 xexx8yu xyri2b x18d9i69 x1c1uobl x1n2onr6"]').get_attribute("datetime")
                    
                    if not comment_text: continue
                    comment_url = f"{post_url}comment/{hash(comment_user + comment_text)}"
                    self.cursor.execute(
                        "INSERT OR IGNORE INTO posts (fonte_coleta, post_url, usuario, conteudo, data_hora, parent_url, run_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (fonte, comment_url, comment_user, comment_text, datetime_str, post_url, self.run_id)
                    )
                    comments_saved += 1
                except Exception as e:
                    logging.warning(f"Não foi possível extrair detalhes de um comentário. Erro: {e}")
                    continue
            
            if comments_saved > 0:
                self.conn.commit()
            logging.info(f"Salvados {comments_saved} comentários novos no banco de dados.")

        except Exception as e:
            logging.error(f"Falha geral ao processar o post {post_url}. Erro: {e}")

    def close(self):
        if self.driver: self.driver.quit()
        if self.conn: self.conn.close()
        logging.info("Coletor do Instagram finalizado.")

# --- BLOCO CORRIGIDO (Sem alterações aqui, apenas no 'if __name__ ...' abaixo) ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] [%(asctime)s] - %(message)s')
    parser = argparse.ArgumentParser(description='Coletor de Dados do Instagram (Execução Individual).')
    
    parser.add_argument('--db-path', help='Caminho para o banco de dados.')
    parser.add_argument('--run-id', help='ID único para esta execução.')
    parser.add_argument('--profile-path', required=True)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--profiles', nargs='*', default=[])
    parser.add_argument('--tags', nargs='*', default=[])
    
    args = parser.parse_args()

    db_path = args.db_path or 'dados/teste_instagram.db'
    run_id = args.run_id or f"teste_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    if not args.profiles:
        logging.error("É necessário fornecer ao menos um perfil com --profiles.")
    else:
        collector = InstagramCollector(
            profile_path=args.profile_path,
            db_path=db_path,
            run_id=run_id,
            headless=args.headless
        )
        collector.collect(profiles=args.profiles, tags=args.tags)
        collector.close()