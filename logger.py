import logging

logger = logging.getLogger("my_logger")

def add_stream_handler_with_formatter(logger, formatter):
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

def erase_previous_log_file(filename="./log.txt"):
    f = open(filename, 'w')
    f.write("")
    f.close()

def add_file_handler_with_formatter(logger, formatter, filename="./log.txt"):
    file_handler = logging.FileHandler('log.txt')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def setup_logger(logger):
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    add_stream_handler_with_formatter(logger, formatter)
    erase_previous_log_file()
    add_file_handler_with_formatter(logger, formatter)
    
    logger.info("Set up logger")

setup_logger(logger)