import logging
import sys
from typing import Dict, List, Optional

LOG_FORMAT = "{asctime}:{name}:{levelname}: {message}"
LOG_FORMATTER = logging.Formatter(LOG_FORMAT, style="{")
STDOUT_NAME = "stdout_stream_handler"
STDERR_NAME = "stderr_stream_handler"
LOGFILE_HANDLE_NAME = "logfile_file_handler"


class _ExcludeWarningsFilter(logging.Filter):
    def filter(self, record):  # noqa
        """Only lets through log messages with log level below ERROR."""
        return record.levelno < logging.WARNING


def verbose(
    logfile: Optional[str] = None,
    override: bool = True,
    package_verbosity: Optional[Dict[str, bool]] = None,
) -> None:
    """Set the verbosity for echopype print outs.
    If called it will output logs to terminal by default.

    Parameters
    ----------
    logfile : str, optional
        Optional string path to the desired log file.
    override: bool
        Boolean flag to override verbosity,
        which turns off verbosity if the value is `False`.
        Default is `True`.
    package_verbosity: dict, optional
        Dictionary of package names and their verbosity levels.
        Default is `None` which will use the `override` value for all packages.
        Example:
        {
            "echopype.convert": True,
            "echopype.calibrate": False,
        }

    Returns
    -------
    None
    """
    if not isinstance(override, bool):
        raise ValueError("override argument must be a boolean")

    if package_verbosity is not None:
        if not isinstance(package_verbosity, dict):
            raise ValueError("package_verbosity argument must be a dictionary")
        for logger_name, verbose in package_verbosity.items():
            if not isinstance(logger_name, str):
                raise ValueError(
                    f"package_verbosity keys must be strings, got {logger_name} for {verbose}"
                )
            if not isinstance(verbose, bool):
                raise ValueError(
                    f"package_verbosity values must be booleans, got {verbose} for {logger_name}"
                )
    else:
        package_verbosity = {}

    package_name = __name__.split(".")[0]  # Get the package name
    for logger in _get_all_loggers():
        _set_verbose(logger, package_verbosity.get(logger.name, override))
        if package_name not in logger.name:
            continue
        handlers = [h.name for h in logger.handlers]
        if logfile is None:
            if LOGFILE_HANDLE_NAME in handlers:
                # Remove log file handler if it exists
                handler = next(filter(lambda h: h.name == LOGFILE_HANDLE_NAME, logger.handlers))
                logger.removeHandler(handler)
        elif LOGFILE_HANDLE_NAME not in handlers:
            # Only add the logfile handler if it doesn't exist
            _set_logfile(logger, logfile)

        logger.propagate = logfile is None


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
    logger.setLevel(logging.DEBUG)

    # Setup stream handler
    STREAM_HANDLER = logging.StreamHandler(sys.stdout)
    STREAM_HANDLER.setLevel(logging.DEBUG)
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


def _set_verbose(logger: logging.Logger, verbose: bool) -> None:
    """Set the verbosity for echopype logs."""
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)


def _set_logfile(logger: logging.Logger, logfile: Optional[str] = None) -> logging.Logger:
    """Adds log file handler to logger"""
    if not logfile:
        raise ValueError("Please provide logfile path")
    file_handler = logging.FileHandler(logfile)
    file_handler.set_name(LOGFILE_HANDLE_NAME)
    file_handler.setFormatter(LOG_FORMATTER)
    logger.addHandler(file_handler)
