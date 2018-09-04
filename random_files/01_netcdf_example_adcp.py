
# coding: utf-8

# ## Working with OOI NetCDF Data
# 
# In this example we will learn how to programatically download and work with OOI NetCDF data from within the notebook. We will use data from the 3D Thermistory Array deployed in the ASHES Vent field at Axial Seamount for this example, but the mechanics apply to all datasets that are processed through the OOI Cyberinfrastructure (CI) system. You wil learn:
# 
# * how to find the data you are looking for
# * how to use the machine to machine API to request data
# * how to load the NetCDF data into your notebook, once the data request has completed
# * how to explore and plot data
# 
# The difference between a NetCDF and JSON data request is that NetCDF files are served asynchronously and delivered to a THREDDS server, while the JSON data response is synchronous (instantaneous) and served as a JSON object in the GET response. NetCDF data is undecimated (full data set), while the JSON response is decimated down to a maximum of 20,000 data points.
# 
# Login in at https://ooinet.oceanobservatories.org/ and obtain your <b>API username and API token</b> under your profile (top right corner).
# 
# A great resource for data wrangling and exploration in python can be found at https://chrisalbon.com/. Tip: add "albon" to your search in google when trying to find a common technique and chances are Chris Albon has made a post on how to do it.

# In[3]:


# username =''
# token = ''
username = 'OOIAPI-SQI3CFA9Y1D6NO'
token = 'UTM58WKFYG39WC'


# Optionally, you can handle authentication outside the notebook by setting up a .netrc file in your home directory and loading it with your bash profile. Open your terminal
# ```
# $ touch .netrc
# $ chmod 700 .netrc
# $ vim .netrc
# 
# ```
# Add the following your your .netrc file:
# 
# ```
# machine ooinet.oceanobservatories.org
# login OOIAPI-TEMPD1SPK4K0X
# password TEMPCXL48ET2XT
# ```
# 
# Use your username and token. Save the file and uncomment the following cell.

# In[2]:


import netrc
netrc = netrc.netrc()
remoteHostName = "ooinet.oceanobservatories.org"
info = netrc.authenticators(remoteHostName)
username = info[0]
token = info[2]


# ### Part One: Finding and requesting the data.

# In[4]:


import requests
import time


# The ingredients being used to build the data_request_url can be found here. For this example, we will use the data from the 3D Thermistor Array (TMPSF)
# http://ooi.visualocean.net/instruments/view/RS03ASHS-MJ03B-07-TMPSFA301
# 
# ![RS03ASHS-MJ03B-07-TMPSFA301](https://github.com/friedrichknuth/ooi_data_analysis/raw/master/qc_db_images/RS03ASHS-MJ03B-07-TMPSFA301.png)

# In[54]:


# subsite = 'CE04OSBP'
# node = 'LJ01C'
# sensor = '05-ADCPSI103'
# method = 'streamed'
# stream = 'adcp_velocity_beam'
# beginDT = '2017-08-21T00:00:00.000Z' #begin of first deployement
# # endDT = None
# endDT = '2017-08-22T23:59:59.000Z' #begin of first deployement

# subsite = 'CE06ISSM'
# node = 'MFD35'
# sensor = '04-ADCPTM000'
# method = 'recovered_host'
# stream = 'adcp_velocity_earth'
# beginDT = '2016-11-01T00:00:00.000Z' #begin of first deployement
# endDT = '2016-11-02T00:00:00.000Z'

subsite = 'CE06ISSM'
node = 'MFD37'
sensor = '07-ZPLSCC000'
method = 'telemetered'
stream = 'zplsc_c_instrument'
beginDT = '2016-11-01T00:00:00.000Z' #begin of first deployement
endDT = '2016-11-02T00:00:00.000Z'

# subsite = 'RS03ASHS'
# node = 'MJ03B'
# sensor = '07-TMPSFA301'
# method = 'streamed'
# stream = 'tmpsf_sample'
# beginDT = '2014-09-27T01:01:01.000Z' #begin of first deployement
# endDT = None

# subsite = 'RS03ASHS'
# node = 'MJ03B'
# sensor = '07-TMPSFA301'
# method = 'streamed'
# stream = 'tmpsf_sample'
# beginDT = '2014-09-27T01:01:01.000Z' #begin of first deployement
# endDT = None


# Send the data request.

# In[61]:


base_url = 'https://ooinet.oceanobservatories.org/api/m2m/12576/sensor/inv/'

data_request_url ='/'.join((base_url,subsite,node,sensor,method,stream))
params = {
    'beginDT':beginDT,
    'endDT':endDT,   
}
r = requests.get(data_request_url, params=params, auth=(username, token))
data = r.json()


# The first url in the response is the location on THREDDS where the data is being served. We will get back to using the THREDDS location later.

# In[62]:


print(data['allURLs'][0])


# The second url in the response is the regular APACHE server location for the data.

# In[63]:


data


# We will use this second location to programatically check for a status.txt file to be written, containing the text 'request completed'. This indicates that the request is completed and the system has finished writing out the data to this location. This step may take a few minutes.

# In[9]:


get_ipython().run_cell_magic('time', '', "check_complete = data['allURLs'][1] + '/status.txt'\nfor i in range(1800): \n    r = requests.get(check_complete)\n    if r.status_code == requests.codes.ok:\n        print('request completed')\n        break\n    else:\n        time.sleep(1)")


# ## Part Two: Loading the data into the notebook.

# In[66]:


import re
import xarray as xr
import pandas as pd
import os


# Next we will parse the html at the location where the files are being delivered to get the list of the NetCDF files written to THREDDS. Note that seperate NetCDF files are created at 500 mb intervals and when there is a new deployment.

# In[67]:


url = data['allURLs'][0]
# url = 'https://opendap.oceanobservatories.org/thredds/catalog/ooi/ooidatateam@gmail.com/20180221T030103-RS03ASHS-MJ03B-07-TMPSFA301-streamed-tmpsf_sample/catalog.html'
tds_url = 'https://opendap.oceanobservatories.org/thredds/dodsC'
datasets = requests.get(url).text
urls = re.findall(r'href=[\'"]?([^\'" >]+)', datasets)
x = re.findall(r'(ooi/.*?.nc)', datasets)
for i in x:
    if i.endswith('.nc') == False:
        x.remove(i)
for i in x:
    try:
        float(i[-4])
    except:
        x.remove(i)
datasets = [os.path.join(tds_url, i) for i in x]


# In[68]:


datasets


# In[69]:


ds = xr.open_mfdataset(datasets)
ds


# In[78]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[80]:


ds['driver_timestamp'].plot(marker='.')


# In[92]:


ds['zplsc_c_values_channel_1'].T.plot()


# In[100]:


ds['zplsc_c_values_channel_1'].lat


# Use xarray to open all netcdf files as a single xarray datase, swap the dimention from obs to time and and examine the content.

# In[16]:


ds = xr.open_mfdataset(datasets)
ds = ds.swap_dims({'obs': 'time'})
ds = ds.chunk({'time': 100})
ds


# ## Part Three: Exploring the data.

# In[17]:


import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import numpy as np


# Use built in xarray plotting functions to inspect beam intensity data.

# In[30]:


ds_copy = ds.copy()


# In[38]:


fig,ax = plt.subplots(2,2,figsize=(16,8))
ds['echo_intensity_beam1'].T.plot(ax=ax[0,0])
ds['echo_intensity_beam2'].T.plot(ax=ax[0,1])
ds['echo_intensity_beam3'].T.plot(ax=ax[1,0])
ds['echo_intensity_beam4'].T.plot(ax=ax[1,1])
plt.show()


# We can tell that the peak temperature is increatsing, but this simple line plot does not reveal the internal data distribution. Let's convert to pandas dataframe and downsample from 1 Hz to 1/60 Hz. This step may take 5-10 minutes. More ram will be allocated during the workshop to expedite processing. If the step fails entirely for any reason, please send us a note on slack.

# In[10]:


get_ipython().run_cell_magic('time', '', "from dask.diagnostics import ProgressBar\nwith ProgressBar():\n    df = ds['temperature12'].to_dataframe()\n    df = df.resample('min').mean()")


# In[12]:


get_ipython().run_cell_magic('time', '', "plt.close()\nfig, ax = plt.subplots()\nfig.set_size_inches(16, 6)\ndf['temperature12'].plot(ax=ax)\ndf['temperature12'].resample('H').mean().plot(ax=ax)\ndf['temperature12'].resample('D').mean().plot(ax=ax)\nplt.show()")


# Now we are getting a better sense of the data. Let's convert time to ordinal, grab temperature values and re-examine using hexagonal bi-variate binning. Again, this step may take a few minutes, but should run faster during the workshop. If the step fails entirely for any reason, please send us a note on slack.

# In[13]:


get_ipython().run_cell_magic('time', '', 'time = []\ntime_pd = pd.to_datetime(ds.time.values.tolist())\nfor i in time_pd:\n    i = np.datetime64(i).astype(datetime.datetime)\n    time.append(dates.date2num(i)) ')


# In[14]:


temperature = ds['temperature12'].values.tolist()


# In[15]:


plt.close()
fig, ax = plt.subplots()
fig.set_size_inches(16, 6)

hb1 = ax.hexbin(time, temperature, bins='log', vmin=0.4, vmax=3, gridsize=(1100, 100), mincnt=1, cmap='Blues')
fig.colorbar(hb1)
ax.yaxis.grid(True)
ax.xaxis.grid(True)
# ax.set_xlim(datetime.datetime(2015, 12, 1, 0, 0),datetime.datetime(2016, 7, 25, 0, 0))
# ax.set_ylim(2,11)
years = dates.YearLocator()
months = dates.MonthLocator()
yearsFmt = dates.DateFormatter('\n\n\n%Y')
monthsFmt = dates.DateFormatter('%b')
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthsFmt)
ax.xaxis.set_minor_locator(years)
ax.xaxis.set_minor_formatter(yearsFmt)
plt.tight_layout()
plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
plt.ylabel('Temperature $^\circ$C')
plt.xlabel('Time')
plt.show()

