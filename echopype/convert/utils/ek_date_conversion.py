"""
Code originally developed for pyEcholab
(https://github.com/CI-CMG/pyEcholab)
by Rick Towler <rick.towler@noaa.gov> at NOAA AFSC.

Contains functions to convert date information.

TODO: merge necessary function into ek60.py or group everything into a class
TODO: fix docstring
"""

import datetime

from pytz import utc as pytz_utc

# NT epoch is Jan 1st 1601
UTC_NT_EPOCH = datetime.datetime(1601, 1, 1, 0, 0, 0, tzinfo=pytz_utc)
# Unix epoch is Jan 1st 1970
UTC_UNIX_EPOCH = datetime.datetime(1970, 1, 1, 0, 0, 0, tzinfo=pytz_utc)

EPOCH_DELTA_SECONDS = (UTC_UNIX_EPOCH - UTC_NT_EPOCH).total_seconds()

__all__ = ["nt_to_unix", "unix_to_nt"]


def nt_to_unix(nt_timestamp_tuple, return_datetime=True):
    """
    :param nt_timestamp_tuple: Tuple of two longs representing the NT date
    :type nt_timestamp_tuple: (long, long)

    :param return_datetime:  Return a datetime object instead of float
    :type return_datetime: bool


    Returns a datetime.datetime object w/ UTC timezone
    calculated from the nt time tuple

    lowDateTime, highDateTime = nt_timestamp_tuple

    The timestamp is a 64bit count of 100ns intervals since the NT epoch
    broken into two 32bit longs, least significant first:

    >>> dt = nt_to_unix((19496896L, 30196149L))
    >>> match_dt = datetime.datetime(2011, 12, 23, 20, 54, 3, 964000, pytz_utc)
    >>> assert abs(dt - match_dt) <= dt.resolution
    """

    lowDateTime, highDateTime = nt_timestamp_tuple
    sec_past_nt_epoch = ((highDateTime << 32) + lowDateTime) * 1.0e-7

    if return_datetime:
        return UTC_NT_EPOCH + datetime.timedelta(seconds=sec_past_nt_epoch)

    else:
        sec_past_unix_epoch = sec_past_nt_epoch - EPOCH_DELTA_SECONDS
        return sec_past_unix_epoch


def unix_to_nt(unix_timestamp):
    """
    Given a date, return the 2-element tuple used for timekeeping with SIMRAD echosounders


    #Simple conversion
    >>> dt = datetime.datetime(2011, 12, 23, 20, 54, 3, 964000, pytz_utc)
    >>> assert (19496896L, 30196149L) == unix_to_nt(dt)

    #Converting back and forth between the two standards:
    >>> orig_dt = datetime.datetime.now(tz=pytz_utc)
    >>> nt_tuple = unix_to_nt(orig_dt)

    #converting back may not yield the exact original date,
    #but will be within the datetime's precision
    >>> back_to_dt = nt_to_unix(nt_tuple)
    >>> d_mu_seconds = abs(orig_dt - back_to_dt).microseconds
    >>> mu_sec_resolution = orig_dt.resolution.microseconds
    >>> assert d_mu_seconds <= mu_sec_resolution
    """

    if isinstance(unix_timestamp, datetime.datetime):
        if unix_timestamp.tzinfo is None:
            unix_datetime = pytz_utc.localize(unix_timestamp)

        elif unix_timestamp.tzinfo == pytz_utc:
            unix_datetime = unix_timestamp

        else:
            unix_datetime = pytz_utc.normalize(unix_timestamp.astimezone(pytz_utc))

    else:
        unix_datetime = unix_to_datetime(unix_timestamp)

    sec_past_nt_epoch = (unix_datetime - UTC_NT_EPOCH).total_seconds()

    onehundred_ns_intervals = int(sec_past_nt_epoch * 1e7)
    lowDateTime = onehundred_ns_intervals & 0xFFFFFFFF
    highDateTime = onehundred_ns_intervals >> 32

    return lowDateTime, highDateTime


def unix_to_datetime(unix_timestamp):
    """
    :param unix_timestamp: Number of seconds since unix epoch (1/1/1970)
    :type unix_timestamp: float

    :param tz: timezone to use for conversion (default None = UTC)
    :type tz: None or tzinfo object (see datetime docs)

    :returns: datetime object
    :raises: ValueError if unix_timestamp is not of type float or datetime

    Returns a datetime object from a unix timestamp.  Simple wrapper for
    :func:`datetime.datetime.fromtimestamp`

    >>> from pytz import utc
    >>> from datetime import datetime
    >>> epoch = unix_to_datetime(0.0, tz=utc)
    >>> assert epoch == datetime(1970, 1, 1, tzinfo=utc)
    """

    if isinstance(unix_timestamp, datetime.datetime):
        if unix_timestamp.tzinfo is None:
            unix_datetime = pytz_utc.localize(unix_timestamp)

        elif unix_timestamp.tzinfo == pytz_utc:
            unix_datetime = unix_timestamp

        else:
            unix_datetime = pytz_utc.normalize(unix_timestamp.astimezone(pytz_utc))

    elif isinstance(unix_timestamp, float):
        unix_datetime = pytz_utc.localize(
            datetime.datetime.fromtimestamp(unix_timestamp)
        )

    else:
        errstr = "Looking for a timestamp of type datetime.datetime or # of sec past unix epoch.\n"
        errstr += "Supplied timestamp '%s' of type %s." % (
            str(unix_timestamp),
            type(unix_timestamp),
        )
        raise ValueError(errstr)

    return unix_datetime


def datetime_to_unix(datetime_obj):
    """
    :param datetime_obj: datetime object to convert
    :type datetime_obj: :class:`datetime.datetime`

    :param tz: Timezone to use for converted time -- if None, uses timezone
                information contained within datetime_obj
    :type tz: :class:datetime.tzinfo

    >>> from pytz import utc
    >>> from datetime import datetime
    >>> epoch = datetime(1970, 1, 1, tzinfo=utc)
    >>> assert datetime_to_unix(epoch) == 0
    """

    timestamp = (datetime_obj - UTC_UNIX_EPOCH).total_seconds()

    return timestamp
