
# coding: utf-8

# In[1]:


import netCDF4
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


import datetime, time
# calculate the offset taking into account daylight saving time
utc_offset_sec = time.altzone if time.localtime().tm_isdst else time.timezone
utc_offset = datetime.timedelta(seconds=-utc_offset_sec)
date_created = datetime.datetime.now().replace(tzinfo=datetime.timezone(offset=utc_offset)).isoformat(timespec='seconds')


# ## Create .nc file

# In[3]:


# Open up .nc file for writing
ncfile = netCDF4.Dataset("../test.nc", "w", format="NETCDF4")


# In[4]:


print(ncfile)


# ## Create all groups

# In[5]:


# Create all groups
annotation  = ncfile.createGroup("Annotation")
environment = ncfile.createGroup("Environment")
platform    = ncfile.createGroup("Platform")
nmea        = ncfile.createGroup("Platform/NMEA")
provenance  = ncfile.createGroup("Provenance")
sonar       = ncfile.createGroup("Sonar")
beam1       = ncfile.createGroup("Sonar/Beam_group1")
vendor      = ncfile.createGroup("Vendor specific")


# ## Top-level group
# Everything in the Top-level group from SONAR-netCDF4 v1.7 is implemented.

# In[6]:


# Attributes
ncfile.Conventions = "CF-1.7, SONAR-netCDF4, ACDD-1.3"
ncfile.date_created = date_created
ncfile.keywords = "EK60"
ncfile.sonar_convention_authority = "ICES"
ncfile.sonar_convention_name = "SONAR-netCDF4"
ncfile.sonar_convention_version = "1.7"
ncfile.summary = "some test data"


# In[7]:


# Dimension
ping_time_dim = ncfile.createDimension("ping_time",None)

# Coorindate variables
ping_time = ncfile.createVariable("ping_time",np.float64,("ping_time",))
ping_time.axis = "T"
ping_time.calendar = "gregorian"
ping_time.long_name = "Timestamp of each ping"
ping_time.standard_name = "time"
ping_time.units = "nanoseconds since 1601-01-01 00:00:00Z"


# In[8]:


print(ncfile)


# ## Annotation group
# Nothing to add for now.

# ## Environment group
# Everything in the Environment group from SONAR-netCDF4 v1.7 is implemented.

# In[9]:


# Environment group
# Dimensions
frequency_dim = environment.createDimension("frequency",None)

# Coordinate variables
frequency = environment.createVariable("frequency","f8",("frequency",))
frequency.long_name = "Acoustic frequency"
frequency.standard_name = "sound_frequency"
frequency.units = "Hz"
frequency.valid_min = 0.0

# Variables
env_absorption = environment.createVariable("absorption_indicative","f8",("frequency",))
env_absorption.long_name = "Indicative acoustic absorption"
env_absorption.units = "dB/m"
env_absorption.valid_min = 0.0

env_sound_speed = environment.createVariable("sound_speed_indicative","f8")
env_sound_speed.long_name = "Indicative sound speed"
env_sound_speed.standard_name = "speed_of_sound_in_sea_water"
env_sound_speed.units = "m/s"
env_sound_speed.valid_min = 0.0


# In[10]:


print(environment)


# ## Platform group
# The Platform group from SONAR-netCDF4 v1.7 is partially implemented with additions.

# In[11]:


# Group attributes
platform.platform_code_ICES = ""
platform.platform_name = ""
platform.platform_type = ""


# In[12]:


# Dimensions
platform_time1_dim = platform.createDimension("time1",None)
platform_time2_dim = platform.createDimension("time2",None)


# In[13]:


# Coordinate variables
platform_time1 = platform.createVariable("time1",None)
platform_time1.axis = "T"
platform_time1.calendar = "gregorian"
platform_time1.long_name = "Timestamps for position data"
platform_time1.standard_name = "time"
platform_time1.units = "nanoseconds since 1601-01-01 00:00:00Z"


# In[14]:


platform_time2 = platform.createVariable("time2",None)
platform_time2.axis = "T"
platform_time2.calendar = "gregorian"
platform_time2.long_name = "Timestamps for position data"
platform_time2.standard_name = "time"
platform_time2.units = "nanoseconds since 1601-01-01 00:00:00Z"


# In[15]:


# Variables
platform_lat = platform.createVariable("latitude",np.float32,("time1"))
platform_lat.long_name = "Platform latitude"
platform_lat.standard_name = "latitude"
platform_lat.units = "degrees_north"
platform_lat.valid_range = (-90.0,90.0)

platform_long = platform.createVariable("longitude",np.float32,("time1"))
platform_long.long_name = "Platform longitude"
platform_long.standard_name = "longitude"
platform_long.units = "degrees_east"
platform_long.valid_range = (-180.0,180.0)


# In[16]:


# MRU stuff
platform_MRU_offset_x = platform.createVariable("MRU_offset_x",np.float32)
platform_MRU_offset_x.long_name = "Distance along the x-axis from the platform coor-dinate system origin to the motion reference unit sensor origin"
platform_MRU_offset_x.units = "m"

platform_MRU_offset_y = platform.createVariable("MRU_offset_y",np.float32)
platform_MRU_offset_y.long_name = "Distance along the y-axis from the platform coor-dinate system origin to the motion reference unit sensor origin"
platform_MRU_offset_y.units = "m"

platform_MRU_offset_z = platform.createVariable("MRU_offset_z",np.float32)
platform_MRU_offset_z.long_name = "Distance along the z-axis from the platform coor-dinate system origin to the motion reference unit sensor origin"
platform_MRU_offset_z.units = "m"

platform_MRU_rotation_x = platform.createVariable("MRU_rotation_x",np.float32)
platform_MRU_rotation_x.long_name = "Extrinsic rotation about the x-axis from the plat-form to MRU coordinate systems"
platform_MRU_rotation_x.units = "arc_degree"
platform_MRU_rotation_x.valid_range = (-180.0,180.0)

platform_MRU_rotation_y = platform.createVariable("MRU_rotation_y",np.float32)
platform_MRU_rotation_y.long_name = "Extrinsic rotation about the y-axis from the plat-form to MRU coordinate systems"
platform_MRU_rotation_y.units = "arc_degree"
platform_MRU_rotation_y.valid_range = (-180.0,180.0)

platform_MRU_rotation_z = platform.createVariable("MRU_rotation_z",np.float32)
platform_MRU_rotation_z.long_name = "Extrinsic rotation about the z-axis from the plat-form to MRU coordinate systems"
platform_MRU_rotation_z.units = "arc_degree"
platform_MRU_rotation_z.valid_range = (-180.0,180.0)


# In[17]:


# Position offset
platform_position_offset_x = platform.createVariable("position_offset_x",np.float32)
platform_position_offset_x.long_name = "Distance along the x-axis from the platform coor-dinate system origin to the latitude/longitude sen-sor origin"
platform_position_offset_x.units = "m"

platform_position_offset_y = platform.createVariable("position_offset_y",np.float32)
platform_position_offset_y.long_name = "Distance along the y-axis from the platform coor-dinate system origin to the latitude/longitude sen-sor origin"
platform_position_offset_y.units = "m"

platform_position_offset_z = platform.createVariable("position_offset_z",np.float32)
platform_position_offset_z.long_name = "Distance along the z-axis from the platform coor-dinate system origin to the latitude/longitude sen-sor origin"
platform_position_offset_z.units = "m"


# In[18]:


# Pitch/roll/heave
platform_pitch = platform.createVariable("pitch",np.float32,("time2",))
platform_pitch.long_name = "Platform pitch"
platform_pitch.standard_name = "platform_pitch_angle"
platform_pitch.units = "arc_degree"
platform_pitch.valid_range = (-90.0,90.0)

platform_roll = platform.createVariable("roll",np.float32,("time2",))
platform_roll.long_name = "Platform roll"
platform_roll.standard_name = "platform_roll_angle"
platform_roll.units = "arc_degree"
platform_roll.valid_range = (-180.0,180.0)

platform_heave = platform.createVariable("vertical_offset",np.float32,("time2",))
platform_heave.long_name = "Platform vertical offset from nominal"
platform_heave.units = "m"


# In[19]:


# Speed stuff
platform_speed_ground = platform.createVariable("speed_ground",np.float32,("time1",))
platform_speed_ground.long_name = "Platform speed over ground"
platform_speed_ground.standard_name = "platform_speed_wrt_ground"
platform_speed_ground.units = "m/s"
platform_speed_ground.valid_min = 0.0

platform_speed_relative = platform.createVariable("speed_relative",np.float32,("time2",))
platform_speed_relative.long_name = "Platform speed relative to water"
platform_speed_relative.standard_name = "platform_speed_wrt_seawater"
platform_speed_relative.units = "m/s"
platform_speed_relative.valid_min = 0.0


# In[20]:


# Transducer offset
platform_transducer_offset_x = platform.createVariable("transducer_offset_x",np.float32)
platform_transducer_offset_x.long_name = "x-axis distance from the platform coordinate sy-stem origin to the sonar transducer"
platform_transducer_offset_x.units = "m"

platform_transducer_offset_y = platform.createVariable("transducer_offset_y",np.float32)
platform_transducer_offset_y.long_name = "y-axis distance from the platform coordinate sy-stem origin to the sonar transducer"
platform_transducer_offset_y.units = "m"

platform_transducer_offset_z = platform.createVariable("transducer_offset_z",np.float32)
platform_transducer_offset_z.long_name = "z-axis distance from the platform coordinate sy-stem origin to the sonar transducer"
platform_transducer_offset_z.units = "m"


# In[21]:


# Water level
platform_water_level = platform.createVariable("water_level",np.float32)
platform_water_level.long_name = "Distance from the platform coordinate system ori-gin to the nominal water level along the z-axis"
platform_water_level.units = "m"


# ## Platform/NMEA

# In[22]:


# Group attributes
nmea.description = "All NMEA sensor datagrams"


# In[23]:


# Dimensions
nmea_time_dim = nmea.createDimension("time",None)


# In[24]:


# Coordinate variables
nmea_time = nmea.createVariable("time",np.float32,("time",))
nmea_time.axis = "T"
nmea_time.calendar = "gregorian"
nmea_time.long_name = "Timestamps for NMEA datagrams"
nmea_time.standard_time = "time"
nmea_time.units = "nanoseconds since 1601-01-01 00:00:00Z"


# In[25]:


# Variables
nmea_datagram = nmea.createVariable("NMEA_datagram",str,("time",))
nmea_datagram.long_name = "NMEA datagram"


# ## Provenance group
# Everything in the Provenance group from SONAR-netCDF4 v1.7 is implemented.

# In[26]:


# Provenance group
# Group attributes
provenance.conversion_software_name = ""
provenance.conversion_software_version = ""
provenance.conversion_time = ""
# Dimensions
filenames_dim = provenance.createDimension("filenames",None)
# Variables
prov_src_fnames = provenance.createVariable("filenames",str,"filenames")
prov_src_fnames.long_name = "Source filenames"


# In[27]:


print(provenance)


# ### *QUESTION: Environment- and platform-related variables recorded at each ping
# For echosounders, many variables, such as temperature, pitch/heave/roll, and lat/lon are recorded along with each ping. With the current organization of information, these should be recorded under `/Sonar/Beam_groupX`. However the essence of these variables are actually related to the environment and the platform. But unless there is a variable (something like "base_ping_time") that is on the top-level, it doesn't make sense to record them in the `Environment` group and `Platform` group.
# 
# What do you think would be the best way to approach this? Here I put them under `/Sonar/Beam_groupX`.

# ## Sonar group
# The Sonar group from SONAR-netCDF4 v1.7 is partially implemented below.
# 
# Anything that is related to the conversion equation for using the data has not been implemented.
# 
# I added the `sample_interval` Dimension and Coordinate Variable in the subgroup `/Sonar/Beam_groupX` so that it can be used to span the backscatter data and all associated data that are recorded on a ping-by-ping basis.

# In[28]:


# Sonar group
# Global attributes
sonar.sonar_manufacturer = "Simrad"
sonar.sonar_model = "EK60"
sonar.sonar_serial_number = ""
sonar.sonar_software_name = ""
sonar.sonar_software_version = ""
sonar.sonar_type = "echosounde"


# In[29]:


# Types: enum
sonar_beam_stab_dict = {b'not_stabilised':0, b'stabilised':1}
sonar_beam_type_dict = {b'single':0, b'split_aperture':1}
# sonar_convert_eq_dict = {b'type_1':1, b'type_2':2}
sonar_transmit_dict = {b'CW':0, b'LFM':1, b'HFM':2}


# In[30]:


sonar_beam_stab = sonar.createEnumType(np.uint8,'beam_stabilisation_t',sonar_beam_stab_dict)
sonar_beam_type = sonar.createEnumType(np.uint8,'beam_t',sonar_beam_type_dict)
# sonar_convert_eq = sonar.createEnumType(np.uint8,'conversion_eq',sonar_convert_eq_dict)
sonar_transmit = sonar.createEnumType(np.uint8,'transmit_t',sonar_transmit_dict)


# In[31]:


# Types: variable length data type
sample_t = sonar.createVLType(np.float64, "sample_t")


# ## Sonar/Beam_group1
# 
# ### Beamwidth
# The SONAR-netCDF4 format recommends 4 variables: `beamwidth_receive_major`, `beamwidth_receive_minor`, `beamwidth_transmit_major`, `beamwidth_transmit_minor`. Here I simplified them to be just two `beamwidth_alongship` and `beamwidth_athwartship` to conform with the EK terminology.
# 
# ### Calibration-related
# - use the variable `gain_correction` to store a scalar `gain` variable that comes with EK60 data.
# - added the variable `sa_correction` that comes with EK60 data.
# 
# ### Transmit
# - use `transmit_duration_nominal` to store `pulse_length` from EK60 data
# - `transmit_frequency_start` and `transmit_frequency_stop` are identical for EK60 since it's a single frequency system
# 
# ### Angles
# - added `beam1_angle_along` and `beam1_angle_athwart` to record electronic angle for each ping and range bin
# - added `beam1_angle_sens_along`, `beam1_angle_sens_athwart`, `beam1_angle_offset_along`, `beam1_angle_offset_athwart` for converting electronic angle to mechanical angle.
# 
# ### Others
# - transducer depth is already in Platform (water_level+transducer_offset_z).

# In[32]:


beam1.beam_mode = "vertical"
# beam1.conversion_equation_t = sonar_convert_eq_dict[b'type_1']


# In[33]:


# Dimensions
beam1_beam_dim = beam1.createDimension("beam", 1)

# beam1_ping_time_dim = beam1.createDimension("ping_time",None)


# In[34]:


# Coordinate variables
beam1_beam = beam1.createVariable("beam",str,("beam",))
beam1_beam.long_name = "Beam name"

# beam1_ping_time = beam1.createVariable("ping_time",np.float64,("ping_time",))
# beam1_ping_time.axis = "T"
# beam1_ping_time.calendar = "gregorian"
# beam1_ping_time.long_name = "Timestamp of each ping"
# beam1_ping_time.standard_name = "time"
# beam1_ping_time.units = "nanoseconds since 1601-01-01 00:00:00Z"


# In[35]:


# Variables: backscattering data
beam1_backscatter_r = beam1.createVariable("backscatter_r",sample_t,("ping_time","beam"))   # mtx with len data type
beam1_backscatter_r.long_name = "Raw backscatter measurements (real part)"
beam1_backscatter_r.units = "power (uncalibrated)"
beam1_backscatter_r.units_scale = "linear"   # this is additional to the SONAR-netCDF4 convention


# In[36]:


# Variables: split-beam angle data -- these are additional to the SONAR-netCDF4 convention
beam1_angle_along = beam1.createVariable("angle_along",sample_t,("ping_time","beam"))       # mtx with vlen data type
beam1_angle_along.long_name = "Electronic angle alongship"
beam1_angle_athwart = beam1.createVariable("angle_athwart",sample_t,("ping_time","beam"))   # mtx with vlen data type
beam1_angle_athwart.long_name = "Electronic angle athwartship"


# In[37]:


# Variables: ADDITIONAL TO SONAR-netCDF4 convention
beam1_bw_along = beam1.createVariable("beamwidth_alongship","f4")      # scalar
beam1_bw_along.long_name = "Half power beamwidth alongship"
beam1_bw_along.standard_name = "beamwidth alongship"
beam1_bw_along.units = "arc_degree"
beam1_bw_along.valid_range = (0.0, 360.0)

beam1_bw_athwart = beam1.createVariable("beamwidth_athwartship","f4")  # scalar
beam1_bw_athwart.long_name = "Half power beamwidth athwartship"
beam1_bw_athwart.standard_name = "beamwidth alongship"
beam1_bw_athwart.units = "arc_degree"
beam1_bw_athwart.valid_range = (0.0, 360.0)


# In[38]:


beam1_dir_x = beam1.createVariable("beam_direction_x","f4")  # scalar
beam1_dir_x.long_name = "x-component of the vector that gives the pointing direction of the beam, in sonar beam coorindate system"
beam1_dir_x.units = "1"
beam1_dir_x.valid_range = (-1.0,1.0)

beam1_dir_y = beam1.createVariable("beam_direction_y","f4")  # scalar
beam1_dir_y.long_name = "y-component of the vector that gives the pointing direction of the beam, in sonar beam coorindate system"
beam1_dir_y.units = "1"
beam1_dir_y.valid_range = (-1.0,1.0)

beam1_dir_z = beam1.createVariable("beam_direction_z","f4")  # scalar
beam1_dir_z.long_name = "z-component of the vector that gives the pointing direction of the beam, in sonar beam coorindate system"
beam1_dir_z.units = "1"
beam1_dir_z.valid_range = (-1.0,1.0)


# In[39]:


# # Beam position -- ADDITIONAL TO SONAR-netCDF4 convention
# THESE ARE THE SAME AS THE "TRANSDUCER_POSITION_X/Y/Z" IN PLATFORM GROUP

# beam1_pos_x = beam1.createVariable("transducer_position_x","f4")  # scalar
# beam1_pos_x.long_name = "x-component of the vector that gives the transducer position in the ship coordinate"
# beam1_pos_x.units = "m"
# beam1_pos_x.valid_min = 0.0

# beam1_pos_y = beam1.createVariable("transducer_position_y","f4")  # scalar
# beam1_pos_y.long_name = "y-component of the vector that gives the transducer position in the ship coordinate"
# beam1_pos_y.units = "m"
# beam1_pos_y.valid_min = 0.0

# beam1_pos_z = beam1.createVariable("transducer_position_z","f4")  # scalar
# beam1_pos_z.long_name = "z-component of the vector that gives the transducer position in the ship coordinate"
# beam1_pos_z.units = "m"
# beam1_pos_z.valid_min = 0.0


# In[40]:


# Enum type of beam variables -- scalar
beam1_stab = beam1.createVariable("beam_stabilisation",sonar_beam_stab)  # scalar
beam1_stab.long_name = "Beam stabilisation applied (or not)"

beam1_type = beam1.createVariable("beam_type",sonar_beam_type)           # scalar
beam1_type.long_name = "Type of beam"

beam1_equiv_beam = beam1.createVariable("equivalent_beam_angle","f8")    # scalar
beam1_equiv_beam.long_name = "Equivalent beam angle"
beam1_equiv_beam.units = "sr"
beam1_equiv_beam.valid_range = (0,4*np.pi)


# In[41]:


# Other variables that are changed to scalars
beam1_gain = beam1.createVariable("gain_correction","f8")  # scalar
beam1_gain.long_name = "Gain correction"
beam1_gain.units = "dB"


# In[42]:


# ADDITIONAL TO SONAR-netCDF4 convention
beam1_sa_corr = beam1.createVariable("sa_correction","f8")  # scalar
beam1_sa_corr.long_name = "Sa correction factor"
beam1_sa_corr.units = "dB"


# In[43]:


beam1_bandwidth = beam1.createVariable("transmit_bandwith","f8",("ping_time",))
beam1_bandwidth.long_name = "Nominal bandwidth of transmitted pulse"
beam1_bandwidth.units = "Hz"
beam1_bandwidth.valid_min = 0.0


# In[44]:


beam1_pulse_length = beam1.createVariable("transmit_duration_nominal","f8",("ping_time",))
beam1_pulse_length.long_name = "Nominal duration of transmitted pulse"
beam1_pulse_length.units = "s"
beam1_pulse_length.valid_min = 0.0


# In[45]:


beam1_freq_start = beam1.createVariable("transmit_frequency_start","f8",("ping_time","beam"))
beam1_freq_start.long_name = "Start frequency in transmitted pulse"
beam1_freq_start.standard_name = "sound_frequency"
beam1_freq_start.units = "Hz"
beam1_freq_start.valid_min = 0.0


# In[46]:


beam1_freq_stop = beam1.createVariable("transmit_frequency_stop","f8",("ping_time","beam"))
beam1_freq_stop.long_name = "Stop frequency in transmitted pulse"
beam1_freq_stop.standard_name = "sound_frequency"
beam1_freq_stop.units = "Hz"
beam1_freq_stop.valid_min = 0.0


# In[47]:


beam1_transmit_power = beam1.createVariable("transmit_power","f8",("ping_time",))
beam1_transmit_power.long_name = "Nominal transmit power"
beam1_transmit_power.units = "W"
beam1_transmit_power.valid_min = 0.0


# In[48]:


beam1_transmit_type = beam1.createVariable("transmit_type",sonar_transmit)
beam1_transmit_type.long_name = "Type of transmitted pulse"


# In[49]:


# Split-beam sensitivity -- ADDITIONAL TO SONAR-netCDF4 convention -- scalar
beam1_angle_sens_along = beam1.createVariable("angle_sensitivity_alongship","f8")
beam1_angle_sens_along.long_name = "Sensitivity to convert alongship electronic angle to mechanical angle"
beam1_angle_sens_along.units = "1"
beam1_angle_sens_athwart = beam1.createVariable("angle_sensitivity_athwartship","f8")
beam1_angle_sens_athwart.long_name = "Sensitivity to convert athwartship electronic angle to mechanical angle"
beam1_angle_sens_athwart.units = "1"

beam1_angle_offset_along = beam1.createVariable("angle_offset_alongship","f8")
beam1_angle_offset_along.long_name = "Offset needed to convert alongship electronic angle to mechanical angle"
beam1_angle_offset_along.units = "arc_degree"
beam1_angle_offset_athwart = beam1.createVariable("angle_offset_athwartship","f8")
beam1_angle_offset_athwart.long_name = "Offset needed to convert athwartship electronic angle to mechanical angle"
beam1_angle_offset_athwart.units = "arc_degree"


# In[50]:


print(sonar)


# In[51]:


print(beam1)


# In[ ]:


# Open up .nc file for writing
rootgrp = netCDF4.Dataset("../test.nc", "w", format="NETCDF4")


# In[ ]:


print(rootgrp)


# In[ ]:


fcstgrp = rootgrp.createGroup("forecasts")
analgrp = rootgrp.createGroup("analyses")


# In[ ]:


rootgrp.groups


# In[ ]:


fcstgrp1 = rootgrp.createGroup("/forecasts/model1")
fcstgrp2 = rootgrp.createGroup("/forecasts/model2")


# In[ ]:


rootgrp.groups


# In[ ]:


for p in rootgrp.groups.values():
    print(p)


# In[ ]:


# Generate random number to mimick echo and pitch/heave/roll data
Sv = pitch = np.random.uniform(-80,-30,(1046,10000))


# In[ ]:


fig,ax = plt.subplots(1,2,figsize=(15,3))
ax[0].imshow(Sv,aspect='auto')
ax[1].imshow(pitch,aspect='auto')
plt.show()


# In[ ]:


sz = Sv.shape
sz


# In[ ]:


# Set dimensions --> all common for ADCP and sonar
time = rootgrp.createDimension("time", None)
depth = rootgrp.createDimension("depth", sz[0])
longitude = rootgrp.createDimension("longitude",180)
latitude = rootgrp.createDimension("latitude",180)


# In[ ]:


rootgrp.dimensions


# In[ ]:


for dimobj in rootgrp.dimensions.values():
    print(dimobj)


# In[ ]:


depth.isunlimited()


# In[ ]:


# Create coordinate variables
time = rootgrp.createVariable("time","f8",("time",))
depth = rootgrp.createVariable("depth","f8",("depth",))
latitude = rootgrp.createVariable("latitude_good","f4",("latitude",))
longitude = rootgrp.createVariable("longitude_good","f4",("longitude",))


# In[ ]:


for dimobj in rootgrp.variables.values():
    print(dimobj)


# In[ ]:


temp = rootgrp.createVariable("temp","f4",("time","depth"))


# In[ ]:


for dimobj in rootgrp.variables.values():
    print(dimobj)


# In[ ]:


rootgrp.createVariable("/forecasts/model1/temp","f4",("time","depth","latitude","longitude",))


# In[ ]:


rootgrp["/forecasts/model1/temp"]


# In[ ]:


rootgrp.variables


# In[ ]:


import time
rootgrp.description = "bogus example script"
rootgrp.history = "Created " + time.ctime(time.time())
rootgrp.source = "netCDF4 python module tutorial"
latitude.units = "degrees north"
longitude.units = "degrees east"
temp.units = "K"
time.units = "hours since 0001-01-01 00:00:00.0"
time.calendar = "gregorian"


# In[ ]:


rootgrp


# In[ ]:


rootgrp.createVariable("transducer1/echo",'f8',("time","depth"))


# In[ ]:


rootgrp["transducer1"]


# In[ ]:


rootgrp.createGroup("transducer2")


# In[ ]:


rootgrp["transducer2"].createDimension("rand_dim", 100)


# In[ ]:


rootgrp["transducer2"]


# In[ ]:


rootgrp.createVariable("transducer2/rand_var",'f8',("time","rand_dim"))


# In[ ]:


rootgrp.close()


# In[ ]:


dsnet = netCDF4.Dataset('/Users/wu-jung/code_git/echopype/test.nc')


# In[ ]:


dsnet


# In[ ]:


dsnet.createVariable("transducer2/time",'f8',("time"))
dsnet.createVariable("transducer2/echo",'f8',("time","depth"))


# In[ ]:


dsnet = netCDF4.Dataset('/Users/wu-jung/code_git/echopype/test3.nc',mode="w")


# In[ ]:


dsnet.createDimension("time",None)
dsnet.createDimension("depth",1046)
dsnet.createVariable("transducer1/echo",'f8',("time","depth"))


# In[ ]:


dsnet


# In[ ]:


dsnet["transducer1"]


# In[ ]:


dsnet["transducer1"].createDimension("time",None)
dsnet["transducer1"].createDimension("depth",1046)


# In[ ]:


dsnet.createVariable("transducer1/time",'f8',("time"))
dsnet.createVariable("transducer1/depth",'f8',("depth"))


# In[ ]:


dsnet.createVariable("transducer1/echo1",'f8',("depth","time"))


# In[ ]:


dsnet.close()


# In[ ]:


import xarray as xr


# In[ ]:


ds = xr.open_dataset("../test3.nc")


# In[ ]:


ds


# In[ ]:


ds_grp = xr.open_dataset("../test3.nc",group='transducer1')
ds_grp


# In[ ]:


ds = xr.open_mfdataset("../test2.nc")


# In[ ]:


ds


# In[ ]:


ds_grp = xr.open_mfdataset("../test2.nc",group='transducer1')


# In[ ]:


ds_grp


# In[ ]:


ds_grp_var = xr.open_dataset("test2.nc",group="transducer1")


# In[ ]:


ds_grp_var

