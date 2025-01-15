import os

class Logger():
    verbose_level = 'normal'
    logger_type = 'print'
    logger_file = None

    def __init__(self, verbose='normal', logtype='print', logfile=None, overwrite=True):
        self.verbose_level = verbose
        self.logger_type = logtype
        self.logger_file = logfile
        assert verbose in ['normal', 'loud', 'silent'], 'Invalid verbosity level'
        assert logtype in ['print', 'file', 'print+file', 'none'], 'Invalid logger type'
        if 'file' in logtype:
            assert logfile is not None, 'Logger file not provided'
            if not os.path.exists(logfile):
                with open(logfile, 'w') as f:
                    f.write('')
            elif overwrite:
                with open(logfile, 'w') as f:
                    f.write('')

    def __log(self, message):
        if self.logger_type == 'none':
            return
        if 'file' in self.logger_type:
            with open(self.logger_file, 'a') as f:
                f.write(message + '\n')
        if 'print' in self.logger_type:
            print(message)

    def log(self, message, level='normal'):
        """
        Log messages based on the configured verbosity level.
        Levels: 'normal', 'loud', 'silent'
        """
        if self.verbose_level == 'silent' or level == 'silent':
            return
        if self.verbose_level == 'loud' or level == 'loud':
            self.__log(message)
            return
        if self.verbose_level == 'normal':
            self.__log(message)
