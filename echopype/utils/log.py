import logging
import sys
from typing import List, Optional

LOG_FORMAT = "{asctime}:{name}:{levelname}: {message}"
LOG_FORMATTER = logging.Formatter(LOG_FORMAT, style="{")
STDOUT_NAME = "stdout_stream_handler"
STDERR_NAME = "stderr_stream_handler"
LOGFILE_HANDLE_NAME = "logfile_file_handler"


class _ExcludeWarningsFilter(logging.Filter):
    def filter(self, record):  # noqa
        """Only lets through log messages with log level below ERROR ."""
        return record.levelno < logging.WARNING


def verbose(logfile: Optional[str] = None, override: bool = False) -> None:
    """Set the verbosity for echopype print outs.
    If called it will output logs to terminal by default.

    Parameters
    ----------
    logfile : str, optional
        Optional string path to the desired log file.
    override: bool
        Boolean flag to override verbosity,
        which turns off verbosity if the value is `False`.
        Default is `False`.

    Returns
    -------
    None
    """
    if not isinstance(override, bool):
        raise ValueError("override argument must be a boolean!")
    package_name = __name__.split(".")[0]  # Get the package name
    loggers = _get_all_loggers()
    verbose = True if override is False else False
    _set_verbose(verbose)
    for logger in loggers:
        if package_name in logger.name:
            handlers = [h.name for h in logger.handlers]
            if logfile is None:
                if LOGFILE_HANDLE_NAME in handlers:
                    # Remove log file handler if it exists
                    handler = next(filter(lambda h: h.name == LOGFILE_HANDLE_NAME, logger.handlers))
                    logger.removeHandler(handler)
            elif LOGFILE_HANDLE_NAME not in handlers:
                # Only add the logfile handler if it doesn't exist
                _set_logfile(logger, logfile)

            if isinstance(logfile, str):
                # Prevents multiple handler from propagating messages
                # this way there are no duplicate line in logfile
                logger.propagate = False
            else:
                logger.propagate = True


def _get_all_loggers() -> List[logging.Logger]:
    """Get all loggers"""
    loggers = [logging.getLogger()]  # get the root logger
    return loggers + [logging.getLogger(name) for name in logging.root.manager.loggerDict]


def _init_logger(name) -> logging.Logger:
    """Initialize logger with the default stdout stream handler

    Parameters
    ----------
    name : str
        Logger name

    Returns
    -------
    logging.Logger
    """
    # Logging setup
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Setup stream handler
    STREAM_HANDLER = logging.StreamHandler(sys.stdout)
    STREAM_HANDLER.setLevel(logging.INFO)
    STREAM_HANDLER.set_name(STDOUT_NAME)
    STREAM_HANDLER.setFormatter(LOG_FORMATTER)
    STREAM_HANDLER.addFilter(_ExcludeWarningsFilter())
    logger.addHandler(STREAM_HANDLER)

    # Setup err stream handler
    ERR_STREAM_HANDLER = logging.StreamHandler(sys.stderr)
    ERR_STREAM_HANDLER.setLevel(logging.WARNING)
    ERR_STREAM_HANDLER.set_name(STDERR_NAME)
    ERR_STREAM_HANDLER.setFormatter(LOG_FORMATTER)
    logger.addHandler(ERR_STREAM_HANDLER)
    return logger


def _set_verbose(verbose: bool) -> None:
    if not verbose:
        logging.disable(logging.WARNING)
    else:
        logging.disable(logging.NOTSET)


def _set_logfile(logger: logging.Logger, logfile: Optional[str] = None) -> logging.Logger:
    """Adds log file handler to logger"""
    if not logfile:
        raise ValueError("Please provide logfile path")
    file_handler = logging.FileHandler(logfile)
    file_handler.set_name(LOGFILE_HANDLE_NAME)
    file_handler.setFormatter(LOG_FORMATTER)
    logger.addHandler(file_handler)
