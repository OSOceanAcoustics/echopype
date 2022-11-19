import pytest
import os.path

EXPECTED_MESSAGE = "Testing log function"


def logging_func(logger):
    logger.info("Testing log function")


@pytest.fixture(params=[False, True])
def verbose(request):
    return request.param


def test_init_logger():
    import logging
    from echopype.utils import log
    logger = log._init_logger('echopype.testing0')
    handlers = [h.name for h in logger.handlers]

    assert isinstance(logger, logging.Logger) is True
    assert logger.name == 'echopype.testing0'
    assert len(logger.handlers) == 2
    assert log.STDERR_NAME in handlers
    assert log.STDOUT_NAME in handlers


def test_set_log_file():
    from echopype.utils import log
    logger = log._init_logger('echopype.testing1')
    from tempfile import TemporaryDirectory
    tmpdir = TemporaryDirectory()
    tmpfile = os.path.join(tmpdir.name, "testfile.log")
    log._set_logfile(logger, tmpfile)
    handlers = [h.name for h in logger.handlers]

    assert log.LOGFILE_HANDLE_NAME in handlers

    # when done with temporary directory
    # see: https://www.scivision.dev/python-tempfile-permission-error-windows/
    try:
        tmpdir.cleanup()
    except PermissionError:
        pass


def test_set_verbose(verbose, capsys):
    from echopype.utils import log
    logger = log._init_logger(f'echopype.testing_{str(verbose).lower()}')

    # To pass through in caplog need to propagate
    # logger.propagate = True

    log._set_verbose(verbose)

    logging_func(logger)

    captured = capsys.readouterr()

    if verbose:
        assert EXPECTED_MESSAGE in captured.out
    else:
        assert "" in captured.out


def test_get_all_loggers():
    import logging
    from echopype.utils import log
    all_loggers = log._get_all_loggers()
    loggers = [logging.getLogger()]  # get the root logger
    loggers = loggers + [logging.getLogger(name) for name in logging.root.manager.loggerDict]

    assert all_loggers == loggers


def run_verbose_test(logger, override, logfile, capsys):
    import echopype as ep
    import os

    ep.verbose(logfile=logfile, override=override)

    logging_func(logger)

    captured = capsys.readouterr()

    if override is True:
        assert captured.out == ""
    else:
        assert EXPECTED_MESSAGE in captured.out

    if logfile is not None:
        assert os.path.exists(logfile)
        with open(logfile) as f:
            assert EXPECTED_MESSAGE in f.read()


@pytest.mark.parametrize(["id", "override", "logfile"], [
    ("fn", True, None),
    ("tn", False, None),
    ("tf", False, 'test.log')
])
def test_verbose(id, override, logfile, capsys):
    from echopype.utils import log
    logger = log._init_logger(f'echopype.testing_{id}')

    if logfile is not None:
        from tempfile import TemporaryDirectory
        tmpdir = TemporaryDirectory()
        tmpfile = os.path.join(tmpdir.name, logfile)
        run_verbose_test(logger, override, tmpfile, capsys)

        # when done with temporary directory
        # see: https://www.scivision.dev/python-tempfile-permission-error-windows/
        try:
            tmpdir.cleanup()
        except PermissionError:
            pass
    else:
        run_verbose_test(logger, override, logfile, capsys)
