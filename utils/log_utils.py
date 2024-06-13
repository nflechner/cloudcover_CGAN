import logging
from datetime import datetime

def write_log(message: str, level: str = 'info') -> None:
    '''
        Write a message to logs.

        Attributes
        ----------
        message: str
            Message to write to logs.
        level: str
            Log level. Supported: ['debug', 'info', 'warning', 'error', 'critical']

        Returns
        -------
        Void.
    '''
    if level not in ['debug', 'info', 'warning', 'error', 'critical']:
        raise ValueError(f'Log level {level} is unsupported.')

    log_message = {
        'debug': logging.debug,
        'info': logging.info,
        'warning': logging.warning,
        'error': logging.error,
        'critical': logging.critical
    }

    timestring = datetime.utcnow().strftime("%d/%m/%YT%H:%M:%S.%fZ")

    log_message[level](f'[{timestring}] {message}')
