"""
This module helps to easily define a logger that can write log messages to the console, a file
The log messages are custom set in log_config.yaml
"""
import logging
import logging.config
from pathlib import Path
import yaml


def setup_logging() -> None:
    """sets up the logging with the log_config.yaml file"""
    path = Path("log_config.yml").absolute()
    with open(path, 'rt') as file:
        config = yaml.safe_load(file.read())
        logging.config.dictConfig(config)


if __name__ == '__main__':
    setup_logging()
    LOG = logging.getLogger(__name__)
    LOG.debug("test")
    LOG.info("test")
    LOG.warning("test")
    LOG.critical("test")
    LOG.error("test")
