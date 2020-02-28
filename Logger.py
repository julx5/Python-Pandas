import logging

__ANSI_WHITE = "\u001b[37m"
__ANSI_YELLOW = "\u001b[33m"
__ANSI_RED = "\u001b[31m"
__ANSI_GREEN = "\u001b[32m"

_logger = None

def _init_logger():
    """Prepare the Logger by setting the Level and the Handler."""
    global _logger
    _logger = logging.getLogger("MM-Eval")
    _logger.setLevel(logging.INFO)
    # Print the Messages to the console
    _handler = logging.StreamHandler()
    # Only show the Logger Name and the Message (eg. SDS::INFO:    MSG)
    _loggingFormatter = logging.Formatter("%(name)s::%(message)s")
    _handler.setFormatter(_loggingFormatter)
    _logger.handlers = [_handler]
    _logger.propagate = False


def info(msg):
    _logger.info(f"INFO:\t{msg}")

def warn(msg):
    _logger.warn(f"{__ANSI_YELLOW}WARN:\t{msg}{__ANSI_WHITE}")

def fatal(msg):
    _logger.fatal(f"{__ANSI_RED}FATAL:\t{msg}{__ANSI_WHITE}")

def result(msg):
    _logger.info(f"{__ANSI_GREEN}RESULT:\t{msg}{__ANSI_WHITE}")

# Print iterations progress - Reference: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

if _logger is None:
    """Check whether the Logger has been initiated.
    This avoids multiple Handler-Adding through multiple Imports."""
    _init_logger()
