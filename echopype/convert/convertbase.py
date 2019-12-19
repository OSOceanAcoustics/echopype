class ConvertBase:
    # Class for assigning attributes common to all echosounders
    def __init__(self):
        # self.parameters['platform_name'] = ''
        # self.parameters['platform_type'] = ''
        # self.parameters['platform_code_ICES'] = ''
        self._platform = {
            'platform_name': '',
            'platform_code_ICES': '',
            'platform_type': ''
        }

    @property
    def platform_name(self):
        return self._platform['platform_name']

    @platform_name.setter
    def platform_name(self, platform_name):
        self._platform['platform_name'] = platform_name

    @property
    def platform_type(self):
        return self._platform['platform_type']

    @platform_type.setter
    def platform_type(self, platform_type):
        self._platform['platform_type'] = platform_type

    @property
    def platform_code_ICES(self):
        return self._platform['platform_code_ICES']

    @platform_code_ICES.setter
    def platform_code_ICES(self, platform_code_ICES):
        self._platform['platform_code_ICES'] = platform_code_ICES

    def raw2nc(self):
        self.save(".nc")

    def raw2zarr(self):
        self.save(".zarr")
