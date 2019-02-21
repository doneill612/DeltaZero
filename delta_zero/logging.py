class Logger(object):

    LEVELS = {
        'verbose': 0,
        'info': 1,
        'warning': 2,
        'fatal': 3
    }

    LOGGERS = { }

    LOG_LEVEL = LEVELS['verbose']
    
    def __init__(self, loc, level):
        self.loc = loc
        self.level = level

    def _log(self, level, display_level, message):
        if self.level >= level:
            print(f'<{self.loc}> [{display_level}]: {message}')

    def verbose(self, message):
        self._log(0, 'DEBUG', message)

    def info(self, message):
        self._log(1, 'INFO', message)

    def warn(self, message):
        self._log(2, 'WARNING', message)

    def fatal(self, message):
        self._log(3, 'FATAL', message)

    @staticmethod
    def get_logger(loc):
        if loc not in LOGGERS.keys():
            logger = Logger(loc, LOG_LEVEL)
            LOGGERS[loc] = logger
            return logger
        else:
            return LOGGERS[loc]

    @static
    def set_log_level(level):
        if level not in LEVELS.keys():
            raise ValueError('invalid log level. '
                             f'valid levels: {list(LEVELS.keys())}')
        LOG_LEVEL = LEVELS[level]
        for logger in LOGGERS.values():
            logger.level = LOG_LEVEL
