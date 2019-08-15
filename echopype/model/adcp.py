from .modelbase import ModelBase


class ModelADCP(ModelBase):
    def __init__(self, file_path=''):
        ModelBase.__init__(self, file_path)
        print('yellow')
