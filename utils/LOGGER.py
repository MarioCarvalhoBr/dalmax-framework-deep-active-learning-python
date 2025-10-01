import logging
import time
import os 
# Configuração do logger
logger = logging.getLogger(__name__)  # Define o logger apenas para o seu módulo
logger.setLevel(logging.DEBUG)

# Criar um handler para escrever no arquivo de log
text_time_log = time.strftime('%Y-%m-%d-%H-%M-%S')
PATH_LOGS = 'results/logs/'
if not os.path.exists(PATH_LOGS):
    os.makedirs(PATH_LOGS)

PATH_LOG_FINAL = PATH_LOGS + text_time_log + '-log-dalmax.log'


def get_logger():
    file_handler = logging.FileHandler(PATH_LOG_FINAL)
    file_handler.setLevel(logging.DEBUG)

    # Criar um handler para imprimir no console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Definir o formato do log
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Adicionar os handlers ao logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def get_path_logger():
    return PATH_LOG_FINAL

def get_path_logs():
    return PATH_LOGS