from tqdm import tqdm

LEVELS = {
        'verbose': 3,
        'info': 2,
        'warning': 1,
        'fatal': 0
    }

LOGGERS = { }

LOG_LEVEL = LEVELS['info']

class Logger(object):    
    
    def __init__(self, loc, level):
        self.loc = loc
        self.level = level

    def _log(self, level, display_level, message):
        if self.level >= level:
            tqdm.write(f'<{self.loc}> [{display_level}]: {message}')

    def _log_str(self, level, display_level, message):
        if self.level >= level:
            return f'<{self.loc}> [{display_level}]: {message}'

    def verbose(self, message, as_str=False):
        if as_str:
            return self._log_str(3, 'DEBUG', message)
        self._log(3, 'DEBUG', message)
            

    def info(self, message, as_str=False):
        if as_str:
            return self._log_str(2, 'INFO', message)
        self._log(2, 'INFO', message)

    def warn(self, message, as_str=False):
        if as_str:
            return self._log_str(1, 'WARNING', message)
        self._log(1, 'WARNING', message)

    def fatal(self, message, as_str=False):
        if as_str:
            return self._log_str(0, 'FATAL', message)
        self._log(0, 'FATAL', message)

    @staticmethod
    def get_logger(loc):
        if loc not in LOGGERS.keys():
            logger = Logger(loc, LOG_LEVEL)
            LOGGERS[loc] = logger
            return logger
        else:
            return LOGGERS[loc]

    @staticmethod
    def set_log_level(level):
        if level not in LEVELS.keys():
            raise ValueError('invalid log level. '
                             f'valid levels: {list(LEVELS.keys())}')
        LOG_LEVEL = LEVELS[level]
        print(f'Log level set to: {level}, {LOG_LEVEL}')
        for logger in LOGGERS.values():
            logger.level = LOG_LEVEL
            
