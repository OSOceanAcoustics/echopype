from azfpMarian import *

input_file_path = './data/18030100.01a'
zplsc_echogram_file_path = './data/18030100.png'
parser = ZplscCParser(None, open(input_file_path,'rb') , ZplscCParser.rec_exception_callback)
parser.create_echogram(zplsc_echogram_file_path)

import matplotlib.pyplot as plt

plt.imshow(power_data_dict[38000.0],aspect='auto')
plt.colorbar()
plt.show()

plt.imshow(power_data_dict[120000.0],aspect='auto')
plt.colorbar()
plt.show()