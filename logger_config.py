import logging
from datetime import datetime
import os

# cria pasta de logs se não existir
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# nome do arquivo com data
log_filename = datetime.now().strftime("logs/log_%Y-%m-%d.txt")

#configurações do logger
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO, #vou utilizar só o nível de informação
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger() #instaciamos a classe que inicia o logger com as configurações acima