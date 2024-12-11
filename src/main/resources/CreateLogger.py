import logging.config
import json

class CreateLogger():

    def __init__(self, logger_name):
        # Load the logging configuration from the JSON file
        try:
            with open("/Users/leon/Uni/Master/Projektarbeit/Projektarbeit/src/main/resources/logging.json", 'r') as f:
                config = json.load(f)
                logging.config.dictConfig(config)
        except FileNotFoundError:
            print(f"Logging configuration file not found")
        self.logger = logging.getLogger(logger_name)
        print("Logger initialized")

    def return_logger(self):
        return self.logger