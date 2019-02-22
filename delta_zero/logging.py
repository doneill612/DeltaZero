LEVELS = {
        'verbose': 3,
        'info': 2,
        'warning': 1,
        'fatal': 0
    }

LOGGERS = { }

LOG_LEVEL = LEVELS['verbose']

class Logger(object):    
    
    def __init__(self, loc, level):
        self.loc = loc
        self.level = level

    def _log(self, level, display_level, message):
        if self.level >= level:
            print(f'<{self.loc}> [{display_level}]: {message}')

    def verbose(self, message):
        self._log(3, 'DEBUG', message)

    def info(self, message):
        self._log(2, 'INFO', message)

    def warn(self, message):
        self._log(1, 'WARNING', message)

    def fatal(self, message):
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
            
if __name__ == '__main__':

    print(LOG_LEVEL)
    logger = Logger.get_logger('logging')
    logger.verbose(logger.level)
