import os
import smtplib
import argparse
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import logging
from src.utils.config import CONFIG, GMAIL_APP_PASSWORD

def main():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] [%(asctime)s] - %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--db-path', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--profile-name', required=True)
    args = parser.parse_args()
    
    cfg_email = CONFIG['email']
    relatorio_path = CONFIG['paths']['output_report']
    graficos_path = CONFIG['paths']['output_graphs']
    
    if not os.path.exists(relatorio_path):
        logging.error(f"Arquivo de relatório não encontrado: {relatorio_path}. Abortando envio.")
        return

    msg = MIMEMultipart()
    msg["Subject"] = f"{cfg_email['subject']} ({args.profile_name} @ {args.run_id})"
    msg["From"] = cfg_email['sender']
    msg["To"] = ", ".join(cfg_email['recipients'])
    corpo_texto = "Olá,\n\nSegue em anexo o relatório consolidado de análise de mídias sociais e os gráficos gerados.\n\nAtenciosamente,\nSistema de Análise Automatizada."
    msg.attach(MIMEText(corpo_texto, "plain", "utf-8"))

    with open(relatorio_path, "rb") as f:
        part = MIMEApplication(f.read(), Name=os.path.basename(relatorio_path))
        part["Content-Disposition"] = f'attachment; filename="{os.path.basename(relatorio_path)}"'
        msg.attach(part)
    if os.path.isdir(graficos_path):
        for filename in sorted(os.listdir(graficos_path)):
            if filename.endswith(".png"):
                filepath = os.path.join(graficos_path, filename)
                with open(filepath, "rb") as f:
                    part = MIMEApplication(f.read(), Name=filename)
                    part["Content-Disposition"] = f'attachment; filename="{filename}"'
                    msg.attach(part)
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(cfg_email['sender'], GMAIL_APP_PASSWORD)
            smtp.sendmail(cfg_email['sender'], cfg_email['recipients'], msg.as_string())
        logging.info(f"E-mail enviado com sucesso para: {', '.join(cfg_email['recipients'])}")
    except Exception as e:
        logging.error(f"Falha ao enviar e-mail: {e}")

if __name__ == "__main__":
    main()