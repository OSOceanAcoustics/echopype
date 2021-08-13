"""
Code originally developed for pyEcholab
(https://github.com/CI-CMG/pyEcholab)
by Rick Towler <rick.towler@noaa.gov> at NOAA AFSC.

Contains low-level functions called by ./ek_raw_parsers.py
"""

import logging
import struct
from io import SEEK_CUR, SEEK_END, SEEK_SET, BufferedReader, FileIO

import fsspec
from fsspec.implementations.local import LocalFileSystem

from . import ek_raw_parsers as parsers

__all__ = ["RawSimradFile"]

log = logging.getLogger(__name__)


class SimradEOF(Exception):
    def __init__(self, message="EOF Reached!"):
        self.message = message

    def __str__(self):
        return self.message


class DatagramSizeError(Exception):
    def __init__(self, message, expected_size_tuple, file_pos=(None, None)):
        self.message = message
        self.expected_size = expected_size_tuple[0]
        self.retrieved_size = expected_size_tuple[1]
        self.file_pos_bytes = file_pos[0]
        self.file_pos_dgrams = file_pos[1]

    def __str__(self):
        errstr = self.message + "%s != %s @ (%s, %s)" % (
            self.expected_size,
            self.retrieved_size,
            self.file_pos_bytes,
            self.file_pos_dgrams,
        )
        return errstr


class DatagramReadError(Exception):
    def __init__(self, message, expected_size_tuple, file_pos=(None, None)):
        self.message = message
        self.expected_size = expected_size_tuple[0]
        self.retrieved_size = expected_size_tuple[1]
        self.file_pos_bytes = file_pos[0]
        self.file_pos_dgrams = file_pos[1]

    def __str__(self):
        errstr = [self.message]
        if self.expected_size is not None:
            errstr.append("%s != %s" % (self.expected_size, self.retrieved_size))
        if self.file_pos_bytes is not None:
            errstr.append("@ (%sL, %s)" % (self.file_pos_bytes, self.file_pos_dgrams))

        return " ".join(errstr)


class RawSimradFile(BufferedReader):
    """
    A low-level extension of the built in python file object allowing the reading/writing
    of SIMRAD RAW files on datagram by datagram basis (instead of at the byte level.)

    Calls to the read method return parse datagrams as dicts.
    """

    #: Dict object with datagram header/python class key/value pairs
    DGRAM_TYPE_KEY = {
        "RAW": parsers.SimradRawParser(),
        "CON": parsers.SimradConfigParser(),
        "TAG": parsers.SimradAnnotationParser(),
        "NME": parsers.SimradNMEAParser(),
        "BOT": parsers.SimradBottomParser(),
        "DEP": parsers.SimradDepthParser(),
        "XML": parsers.SimradXMLParser(),
        "FIL": parsers.SimradFILParser(),
        "MRU": parsers.SimradMRUParser(),
    }

    def __init__(
        self,
        name,
        mode="rb",
        closefd=True,
        return_raw=False,
        buffer_size=1024 * 1024,
        storage_options={},
    ):

        #  9-28-18 RHT: Changed RawSimradFile to implement BufferedReader instead of
        #  io.FileIO to increase performance.

        #  create a raw file object for the buffered reader
        fmap = fsspec.get_mapper(name, **storage_options)
        if isinstance(fmap.fs, LocalFileSystem):
            fio = FileIO(name, mode=mode, closefd=closefd)
        else:
            fio = fmap.fs.open(fmap.root)

        #  initialize the superclass
        super().__init__(fio, buffer_size=buffer_size)
        self._current_dgram_offset = 0
        self._total_dgram_count = None
        self._return_raw = return_raw

    def _seek_bytes(self, bytes_, whence=0):
        """
        :param bytes_: byte offset
        :type bytes_: int

        :param whence:

        Seeks a file by bytes instead of datagrams.
        """

        super().seek(bytes_, whence)

    def _tell_bytes(self):
        """
        Returns the file pointer position in bytes.
        """

        return super().tell()

    def _read_dgram_size(self):
        """
        Attempts to read the size of the next datagram in the file.
        """

        buf = self._read_bytes(4)
        if len(buf) != 4:
            self._seek_bytes(-len(buf), SEEK_CUR)
            raise DatagramReadError(
                "Short read while getting dgram size",
                (4, len(buf)),
                file_pos=(self._tell_bytes(), self.tell()),
            )
        else:
            return struct.unpack("=l", buf)[0]  # This return value is an int object.

    def _bytes_remaining(self):
        old_pos = self._tell_bytes()
        self._seek_bytes(0, SEEK_END)
        end_pos = self._tell_bytes()
        offset = end_pos - old_pos
        self._seek_bytes(old_pos, SEEK_SET)

        return offset

    def _read_timestamp(self):
        """
        Attempts to read the datagram timestamp.
        """

        buf = self._read_bytes(8)
        if len(buf) != 8:
            self._seek_bytes(-len(buf), SEEK_CUR)
            raise DatagramReadError(
                "Short read while getting timestamp",
                (8, len(buf)),
                file_pos=(self._tell_bytes(), self.tell()),
            )

        else:
            lowDateField, highDateField = struct.unpack("=2L", buf)
            #  11/26/19 - RHT - modified to return the raw bytes
            return lowDateField, highDateField, buf

    def _read_dgram_header(self):
        """
        :returns: dgram_size, dgram_type, (low_date, high_date)

        Attempts to read the datagram header consisting of:

            long        dgram_size
            char[4]     type
            long        lowDateField
            long        highDateField
        """

        try:
            dgram_size = self._read_dgram_size()
        except Exception:
            if self.at_eof():
                raise SimradEOF()
            else:
                raise

        #  get the datagram type
        buf = self._read_bytes(4)

        if len(buf) != 4:
            if self.at_eof():
                raise SimradEOF()
            else:
                self._seek_bytes(-len(buf), SEEK_CUR)
                raise DatagramReadError(
                    "Short read while getting dgram type",
                    (4, len(buf)),
                    file_pos=(self._tell_bytes(), self.tell()),
                )
        else:
            dgram_type = buf
        dgram_type = dgram_type.decode()

        #  11/26/19 - RHT
        #  As part of the rewrite of read to remove the reverse seeking,
        #  store the raw header bytes so we can prepend them to the raw
        #  data bytes and pass it all to the parser.
        raw_bytes = buf

        #  read the timestamp - this method was also modified to return
        #  the raw bytes
        lowDateField, highDateField, buf = self._read_timestamp()

        #  add the timestamp bytes to the raw_bytes string
        raw_bytes += buf

        return dict(
            size=dgram_size,
            type=dgram_type,
            low_date=lowDateField,
            high_date=highDateField,
            raw_bytes=raw_bytes,
        )

    def _read_bytes(self, k):
        """
        Reads raw bytes from the file
        """

        return super().read(k)

    def _read_next_dgram(self):
        """
        Attempts to read the next datagram from the file.

        Returns the datagram as a raw string
        """

        #  11/26/19 - RHT - Modified this method so it doesn't "peek"
        #  at the next datagram before reading which was inefficient.
        #  To minimize changes to the code, methods to read the header
        #  and timestamp were modified to return the raw bytes which
        #  allows us to pass them onto the parser without having to
        #  rewind and read again as was previously done.

        #  store our current location in the file
        old_file_pos = self._tell_bytes()

        #  try to read the header of the next datagram
        try:
            header = self._read_dgram_header()
        except DatagramReadError as e:
            e.message = "Short read while getting raw file datagram header"
            raise e

        #  check for invalid time data
        if (header["low_date"], header["high_date"]) == (0, 0):
            log.warning(
                "Skipping %s datagram w/ timestamp of (0, 0) at %sL:%d",
                header["type"],
                str(self._tell_bytes()),
                self.tell(),
            )
            self.skip()
            return self._read_next_dgram()

        #  basic sanity check on size
        if header["size"] < 16:
            #  size can't be smaller than the header size
            log.warning(
                "Invalid datagram header: size: %d, type: %s, nt_date: %s.  dgram_size < 16",
                header["size"],
                header["type"],
                str((header["low_date"], header["high_date"])),
            )

            #  see if we can find the next datagram
            self._find_next_datagram()

            #  and then return that
            return self._read_next_dgram()

        #  get the raw bytes from the header
        raw_dgram = header["raw_bytes"]

        #  and append the rest of the datagram - we subtract 12
        #  since we have already read 12 bytes: 4 for type and
        #  8 for time.
        raw_dgram += self._read_bytes(header["size"] - 12)

        #  determine the size of the payload in bytes
        bytes_read = len(raw_dgram)

        #  and make sure it checks out
        if bytes_read < header["size"]:
            log.warning(
                "Datagram %d (@%d) shorter than expected length:  %d < %d",
                self.tell(),
                old_file_pos,
                bytes_read,
                header["size"],
            )
            self._find_next_datagram()
            return self._read_next_dgram()

        #  now read the trailing size value
        try:
            dgram_size_check = self._read_dgram_size()
        except DatagramReadError as e:
            self._seek_bytes(old_file_pos, SEEK_SET)
            e.message = (
                "Short read while getting trailing raw file datagram size for check"
            )
            raise e

        #  make sure they match
        if header["size"] != dgram_size_check:
            # self._seek_bytes(old_file_pos, SEEK_SET)
            log.warning(
                "Datagram failed size check:  %d != %d @ (%d, %d)",
                header["size"],
                dgram_size_check,
                self._tell_bytes(),
                self.tell(),
            )
            log.warning("Skipping to next datagram...")
            self._find_next_datagram()

            return self._read_next_dgram()

        #  add the header (16 bytes) and repeated size (4 bytes) to the payload
        #  bytes to get the total bytes read for this datagram.
        bytes_read = bytes_read + 20

        if self._return_raw:
            self._current_dgram_offset += 1
            return raw_dgram
        else:
            nice_dgram = self._convert_raw_datagram(raw_dgram, bytes_read)
            self._current_dgram_offset += 1
            return nice_dgram

    def _convert_raw_datagram(self, raw_datagram_string, bytes_read):
        """
        :param raw_datagram_string: bytestring containing datagram (first 4
            bytes indicate datagram type, such as 'RAW0')
        :type raw_datagram_string: str

        :param bytes_read: integer specifying the datagram size, including header
            in bytes,
        :type bytes_read: int

        Returns a formated datagram object using the data in raw_datagram_string
        """

        #  11/26/19 - RHT - Modified this method to pass through the number of
        #  bytes read so we can bubble that up to the user.

        dgram_type = raw_datagram_string[:3].decode()
        try:
            parser = self.DGRAM_TYPE_KEY[dgram_type]
        except KeyError:
            # raise KeyError('Unknown datagram type %s,
            # valid types: %s' % (str(dgram_type),
            # str(self.DGRAM_TYPE_KEY.keys())))
            return raw_datagram_string

        nice_dgram = parser.from_string(raw_datagram_string, bytes_read)
        return nice_dgram

    def _set_total_dgram_count(self):
        """
        Skips quickly through the file counting datagrams and stores the
        resulting number in self._total_dgram_count

        :raises: ValueError if self._total_dgram_count is not None (it has been set before)
        """
        if self._total_dgram_count is not None:
            raise ValueError(
                "self._total_dgram_count has already been set.  Call .reset() first if you really want to recount"  # noqa
            )

        # Save current position for later
        old_file_pos = self._tell_bytes()
        old_dgram_offset = self.tell()

        self._current_dgram_offset = 0
        self._seek_bytes(0, SEEK_SET)

        while True:
            try:
                self.skip()
            except (DatagramReadError, SimradEOF):
                self._total_dgram_count = self.tell()
                break

        # Return to where we started
        self._seek_bytes(old_file_pos, SEEK_SET)
        self._current_dgram_offset = old_dgram_offset

    def at_eof(self):
        old_pos = self._tell_bytes()
        self._seek_bytes(0, SEEK_END)
        eof_pos = self._tell_bytes()

        # Check to see if we're at the end of file and raise EOF
        if old_pos == eof_pos:
            return True

        # Othereise, go back to where we were and re-raise the original
        # exception
        else:
            offset = old_pos - eof_pos
            self._seek_bytes(offset, SEEK_END)
            return False

    def read(self, k):
        """
        :param k: Number of datagrams to read
        :type k: int

        Reads the next k datagrams.  A list of datagrams is returned if k > 1.  The entire
        file is read from the CURRENT POSITION if k < 0. (does not necessarily read from begining
        of file if previous datagrams were read)
        """

        if k == 1:
            try:
                return self._read_next_dgram()
            except Exception:
                if self.at_eof():
                    raise SimradEOF()
                else:
                    raise

        elif k > 0:

            dgram_list = []

            for m in range(k):
                try:
                    dgram = self._read_next_dgram()
                    dgram_list.append(dgram)

                except Exception:
                    break

            return dgram_list

        elif k < 0:
            return self.readall()

    def readall(self):
        """
        Reads the entire file from the beginning and returns a list of datagrams.
        """

        self.seek(0, SEEK_SET)
        dgram_list = []

        for raw_dgram in self.iter_dgrams():
            dgram_list.append(raw_dgram)

        return dgram_list

    def _find_next_datagram(self):
        old_file_pos = self._tell_bytes()
        log.warning("Attempting to find next valid datagram...")

        while self.peek()["type"][:3] not in list(self.DGRAM_TYPE_KEY.keys()):
            self._seek_bytes(1, 1)

        log.warning("Found next datagram:  %s", self.peek())
        log.warning("Skipped ahead %d bytes", self._tell_bytes() - old_file_pos)

    def tell(self):
        """
        Returns the current file pointer offset by datagram number
        """
        return self._current_dgram_offset

    def peek(self):
        """
        Returns the header of the next datagram in the file.  The file position is
        reset back to the original location afterwards.

        :returns: [dgram_size, dgram_type, (low_date, high_date)]
        """

        dgram_header = self._read_dgram_header()
        if dgram_header["type"].startswith("RAW0"):
            dgram_header["channel"] = struct.unpack("h", self._read_bytes(2))[0]
            self._seek_bytes(-18, SEEK_CUR)
        elif dgram_header["type"].startswith("RAW3"):
            chan_id = struct.unpack("128s", self._read_bytes(128))
            dgram_header["channel_id"] = chan_id.strip("\x00")
            self._seek_bytes(-(16 + 128), SEEK_CUR)
        else:
            self._seek_bytes(-16, SEEK_CUR)

        return dgram_header

    def __next__(self):
        """
        Returns the next datagram (synonomous with self.read(1))
        """

        return self.read(1)

    def prev(self):
        """
        Returns the previous datagram 'behind' the current file pointer position
        """

        self.skip_back()
        raw_dgram = self.read(1)
        self.skip_back()
        return raw_dgram

    def skip(self):
        """
        Skips forward to the next datagram without reading the contents of the current one
        """

        # dgram_size, dgram_type, (low_date, high_date) = self.peek()[:3]

        header = self.peek()

        if header["size"] < 16:
            log.warning(
                "Invalid datagram header: size: %d, type: %s, nt_date: %s.  dgram_size < 16",
                header["size"],
                header["type"],
                str((header["low_date"], header["high_date"])),
            )

            self._find_next_datagram()

        else:
            self._seek_bytes(header["size"] + 4, SEEK_CUR)
            dgram_size_check = self._read_dgram_size()

            if header["size"] != dgram_size_check:
                log.warning(
                    "Datagram failed size check:  %d != %d @ (%d, %d)",
                    header["size"],
                    dgram_size_check,
                    self._tell_bytes(),
                    self.tell(),
                )
                log.warning("Skipping to next datagram... (in skip)")

                self._find_next_datagram()

        self._current_dgram_offset += 1

    def skip_back(self):
        """
        Skips backwards to the previous datagram without reading it's contents
        """

        old_file_pos = self._tell_bytes()

        try:
            self._seek_bytes(-4, SEEK_CUR)
        except IOError:
            raise

        dgram_size_check = self._read_dgram_size()

        # Seek to the beginning of the datagram and read as normal
        try:
            self._seek_bytes(-(8 + dgram_size_check), SEEK_CUR)
        except IOError:
            raise DatagramSizeError

        try:
            dgram_size = self._read_dgram_size()

        except DatagramSizeError:
            print("Error reading the datagram")
            self._seek_bytes(old_file_pos, SEEK_SET)
            raise

        if dgram_size_check != dgram_size:
            self._seek_bytes(old_file_pos, SEEK_SET)
            raise DatagramSizeError
        else:
            self._seek_bytes(-4, SEEK_CUR)

        self._current_dgram_offset -= 1

    def iter_dgrams(self):
        """
        Iterates through the file, repeatedly calling self.next() until
        the end of file is reached
        """

        while True:
            # new_dgram = self.next()
            # yield new_dgram

            try:
                new_dgram = next(self)
            except Exception:
                log.debug("Caught EOF?")
                raise StopIteration

            yield new_dgram

    # Unsupported members
    def readline(self):
        """
        aliased to self.next()
        """
        return next(self)

    def readlines(self):
        """
        aliased to self.read(-1)
        """
        return self.read(-1)

    def seek(self, offset, whence):
        """
        Performs the familiar 'seek' operation using datagram offsets
        instead of raw bytes.
        """

        if whence == SEEK_SET:
            if offset < 0:
                raise ValueError("Cannot seek backwards from beginning of file")
            else:
                self._seek_bytes(0, SEEK_SET)
                self._current_dgram_offset = 0
        elif whence == SEEK_END:
            if offset > 0:
                raise ValueError(
                    "Use negative offsets when seeking backward from end of file"
                )

            # Do we need to generate the total number of datagrams w/in the file?
            try:
                self._set_total_dgram_count()
                # Throws a value error if _total_dgram_count has alread been set.  We can ignore it
            except ValueError:
                pass

            self._seek_bytes(0, SEEK_END)
            self._current_dgram_offset = self._total_dgram_count

        elif whence == SEEK_CUR:
            pass
        else:
            raise ValueError(
                "Illegal value for 'whence' (%s), use 0 (beginning), 1 (current), or 2 (end)"
                % (str(whence))
            )

        if offset > 0:
            for k in range(offset):
                self.skip()
        elif offset < 0:
            for k in range(-offset):
                self.skip_back()

    def reset(self):
        self._current_dgram_offset = 0
        self._total_dgram_count = None
        self._seek_bytes(0, SEEK_SET)
