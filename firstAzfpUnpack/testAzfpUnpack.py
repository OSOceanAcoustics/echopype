from azfpUnpack import *

input_file_path = r"C:\Users\mal83\Documents\GitHub\echopype\firstAzfpUnpack\18030100.01A"
zplsc_echogram_file_path = r"C:\Users\mal83\Documents\GitHub\echopype\firstAzfpUnpack"

class DataSetDriverConfigKeys():
    PARTICLE_MODULE = "particle_module"
    PARTICLE_CLASS = "particle_class"
    PARTICLE_CLASSES_DICT = "particle_classes_dict"
    DIRECTORY = "directory"
    STORAGE_DIRECTORY = "storage_directory"
    PATTERN = "pattern"
    FREQUENCY = "frequency"
    FILE_MOD_WAIT_TIME = "file_mod_wait_time"
    HARVESTER = "harvester"
    PARSER = "parser"
    MODULE = "module"
    CLASS = "class"
    URI = "uri"
    CLASS_ARGS = "class_args"

MODULE_NAME = 'mi.dataset.parser.zplsc_c'
CLASS_NAME = 'ZplscCRecoveredDataParticle'
CONFIG = {
    DataSetDriverConfigKeys.PARTICLE_MODULE: MODULE_NAME,
    DataSetDriverConfigKeys.PARTICLE_CLASS: CLASS_NAME
}

parser = ZplscCParser(CONFIG, open(input_file_path,'rb') , ZplscCParser.rec_exception_callback)
parser.create_echogram(zplsc_echogram_file_path)

import matplotlib.pyplot as plt

# plt.imshow(power_data_dict[38000.0],aspect='auto')
# plt.colorbar()
# plt.show()
#
# plt.imshow(power_data_dict[120000.0],aspect='auto')
# plt.colorbar()
# plt.show()