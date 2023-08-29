#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example using mask_impulse.py for masking impulse noise in acoustic data
collected norwest off South Georgia.

Created on Tue Jun 11 11:05:12 2019
@author: Alejandro Ariza, British Antarctic Survey
"""

#------------------------------------------------------------------------------
# import modules
import os
import numpy as np
from echolab2.instruments import EK60
from echopy.processing import mask_impulse as mIN
import matplotlib.pyplot as plt
from echopy.plotting.cmaps import cmaps

#------------------------------------------------------------------------------
# load rawfile
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
rawfile = os.path.join(path, 'JR230-D20091215-T121917.raw')
ek60    = EK60.EK60()
ek60.read_raw(rawfile)

#------------------------------------------------------------------------------
# get 120 kHz data
raw120 = ek60.get_raw_data(channel_number=2)
Sv120  = np.transpose(raw120.get_Sv().data)
t120   = raw120.get_Sv().ping_time
p120   = np.arange(len(t120))
r120   = raw120.get_Sv().range

#------------------------------------------------------------------------------
# Get mask for impulse noise with Ryan's algorithm
m120ryan , m120ryan_ = mIN.ryan(Sv120, r120, m=5, n=1, thr=10)

#------------------------------------------------------------------------------
# Clean impulse noise using Ryan's mask
Sv120ryan            = Sv120.copy()
Sv120ryan[m120ryan]  = np.nan

#------------------------------------------------------------------------------
# Clean Sv from impulse noise with Wang's algorithm
Sv120wang, m120wang_ = mIN.wang(Sv120)

# Note that Wang's algorithm does not return a mask with impulse noise detected
# but Sv without impulse noise and other Sv modifications. It also removes
# Sv signal below and above the target of interest. In this case, krill swarms.

#------------------------------------------------------------------------------
# Figures
plt.figure(figsize=(8,4))

# Sv original
plt.subplot(131).invert_yaxis()
plt.pcolormesh(p120, r120, Sv120, vmin=-80, vmax=-50, cmap=cmaps().ek500)
plt.ylabel('Depth (m)')
plt.title('IN on')

# IN removed with Ryan's algorithm
plt.subplot(132).invert_yaxis()
plt.pcolormesh(p120, r120, Sv120ryan, vmin=-80, vmax=-50, cmap=cmaps().ek500)
plt.tick_params(labelleft=False)
plt.xlabel('Number of pings')
plt.title('IN off (Ryan)')

# IN removed (and further signal) with Wang's algorithm
plt.subplot(133).invert_yaxis()
plt.pcolormesh(p120, r120, Sv120wang, vmin=-80, vmax=-50, cmap=cmaps().ek500)
plt.tick_params(labelleft=False)
plt.title('IN off (Wang)')

# Show and save
plt.tight_layout()
plt.show()
#plt.savefig('masking_impulse_noise.png', figsize=(8,4), dpi=150)