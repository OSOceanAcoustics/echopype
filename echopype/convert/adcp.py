import os
import numpy as np
import dolfyn as dfn
from datetime import datetime as dt
from .set_groups import SetGroups
from echopype._version import get_versions
import matplotlib as mpl
ECHOPYPE_VERSION = get_versions()['version']
del get_versions

path = './echopype/test_data/adcp/Sig1000_IMU.ad2cp'


class ConvertADCP:
    """Class for converting Nortek ADCP `ad2cp` files """

    def __init__(self, _path=''):
        self.path = _path
        self.file_name = os.path.basename(self.path)
        self.platform_name = ''
        self.platform_type = ''
        self.platform_code_ICES = ''
        self.unpacked_data = None

    @property
    def model(self):
        if not self.unpacked_data:
            self.parse_raw()
        try:
            return self._model
        except AttributeError:
            self._model = self.unpacked_data.config.model
            return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def parse_raw(self):
        self.unpacked_data = dfn.read(self.path)

    def get_ping_time(self):
        if not self.unpacked_data:
            self.parse_raw()

        mpltime = np.array(self.unpacked_data.mpltime)
        ping_time = [time.timestamp() for time in mpl.dates.num2date(mpltime)]
        return ping_time

    def raw2nc(self):

        def _set_toplevel_dict():
            out_dict = {'conventions': 'CF-1.7, SONAR-netCDF4-1.0, ACDD-1.3',
                        'keywords': 'ADCP, AD2CP, Nortek',
                        'sonar_convention_authority': 'ICES',
                        'sonar_convention_name': 'SONAR-netCDF4',
                        'sonar_convention_version': '1.0',
                        'summary': '',
                        'title': ''}
            return out_dict

        def _set_env_dict():
            out_dict = {'ping_time': ping_time,
                        'sound_speed': self.unpacked_data.env.c_sound,
                        'temperature': self.unpacked_data.env.temp,
                        'pressure': self.unpacked_data.env.press}
            return out_dict

        def _set_platform_dict():
            tilt = self.unpacked_data.orient.raw
            out_dict = {'ping_time': ping_time,
                        'platform_name': self.platform_name,
                        'platform_type': self.platform_type,
                        'platform_code_ICES': self.platform_code_ICES,
                        'pitch': tilt.pitch,
                        'roll': tilt.roll,
                        'heading': tilt.heading}
            return out_dict

        def _set_prov_dict():
            out_dict = {'conversion_software_name': 'echopype',
                        'conversion_software_version': ECHOPYPE_VERSION,
                        'conversion_time': dt.utcnow().isoformat(timespec='seconds') + 'Z',
                        'source_filenames': self.file_name}
            return out_dict

        def _set_sonar_dict():
            out_dict = {'sonar_manufacturer': self.unpacked_data.props['inst_make'],
                        'sonar_model': self.unpacked_data.config.model,
                        'sonar_serial_number': self.unpacked_data.config.SerialNum,
                        'sonar_software_name': 'N/A',
                        'sonar_software_version': 'N/A',
                        'sonar_type': self.unpacked_data.props['inst_type']}
            return out_dict

        def _set_beam_dict():
            freq = self.unpacked_data.config['filehead config']['PLAN']['FREQ'] * 1000
            out_dict = {'frequency': freq,
                        'ping_time': ping_time,
                        'range_bin': self.unpacked_data.range}
            return out_dict

        if not self.unpacked_data:
            self.parse_raw()

        ping_time = self.get_ping_time()
        filename = os.path.splitext(os.path.basename(self.path))[0]
        self.nc_path = os.path.join(os.path.split(self.path)[0], filename + '.nc')

        if os.path.exists(self.nc_path):    # TODO remove eventually
            os.remove(self.nc_path)

        if os.path.exists(self.nc_path):
            print('          ... this file has already been converted to .nc, conversion not executed.')
        else:
            # Create SetGroups object
            grp = SetGroups(file_path=self.nc_path, echo_type='ADCP')
            grp.set_toplevel(_set_toplevel_dict())      # top-level group
            grp.set_env(_set_env_dict())                # environment group
            grp.set_provenance(os.path.basename(self.file_name),
                               _set_prov_dict())        # provenance group
            grp.set_platform(_set_platform_dict())      # platform group
            grp.set_sonar(_set_sonar_dict())            # sonar group
            grp.set_beam(_set_beam_dict())              # beam group
