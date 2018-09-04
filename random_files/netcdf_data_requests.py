
# coding: utf-8

# ## Requesting OOI NetCDF Data
# 
# In this example we will learn how to programatically download and work with OOI NetCDF data from within the notebook. We will use data from the 3D Thermistory Array deployed in the ASHES Vent field at Axial Seamount for this example, but the mechanics apply to all datasets that are processed through the OOI Cyberinfrastructure (CI) system. You wil learn:
# 
# * how to find the data you are looking for
# * how to use the machine to machine API to request data
# 
# The difference between a NetCDF and JSON data request is that NetCDF files are served asynchronously and delivered to a THREDDS server, while the JSON data response is synchronous (instantaneous) and served as a JSON object in the GET response. NetCDF data is undecimated (full data set), while the JSON response is decimated down to a maximum of 20,000 data points.
# 
# Login in at https://ooinet.oceanobservatories.org/ and obtain your <b>API username and API token</b> under your profile (top right corner), or use the credentials provided below.
# 

# In[39]:


# username = 'OOIAPI-D8S960UXPK4K03'
# token = 'IXL48EQ2XY'
username = 'OOIAPI-SQI3CFA9Y1D6NO'
token = 'UTM58WKFYG39WC'


# ### Finding and requesting the data.

# In[40]:


import requests
import time


# The ingredients being used to build the data_request_url can be found here. For this example, we will use the data from Coastal Endurance Array, Washington Inshore Surface Mooring, Seafloor Multi-Function Node (MFN) Bio-acoustic Sonar:
# 
# http://ooi.visualocean.net/instruments/view/CE06ISSM-MFD37-07-ZPLSCC000

# In[58]:


subsite = 'CE06ISSM'
node = 'MFD37'
sensor = '07-ZPLSCC000'
method = 'telemetered'
stream = 'zplsc_c_instrument'
beginDT = '2018-01-01T08:01:01.000Z' #begin of first deployement
endDT = '2018-01-01T20:01:01.000Z'


# Send the data request.

# In[59]:


# base_url = 'https://ooinet.oceanobservatories.org/api/m2m/12576/sensor/inv/'

# data_request_url ='/'.join((base_url,subsite,node,sensor,method,stream))

data_request_url = 'https://ooinet.oceanobservatories.org/api/m2m/12576/sensor/inv/CE01ISSM/MFD37/07-ZPLSCC000/telemetered/zplsc_c_instrument/'

params = {
    'beginDT':beginDT,
    'endDT':endDT,   
}
r = requests.get(data_request_url, params=params, auth=(username, token))
data = r.json()


# In[60]:


data


# The first url in the response is the location on THREDDS where the data is being served. We will get back to using the THREDDS location later.

# In[54]:


print(data['allURLs'][0])


# The second url in the response is the regular APACHE server location for the data.

# In[6]:


print(data['allURLs'][1])


# We will use this second location to programatically check for a status.txt file to be written, containing the text 'request completed'. This indicates that the request is completed and the system has finished writing out the data to this location. This step may take a few minutes.

# In[ ]:


get_ipython().run_cell_magic('time', '', "check_complete = data['allURLs'][1] + '/status.txt'\nfor i in range(1800): \n    r = requests.get(check_complete)\n    if r.status_code == requests.codes.ok:\n        print('request completed')\n        break\n    else:\n        time.sleep(1)")


# Copy the thredds url (from `print(data['allURLs'][0])`) and paste it into the netcdf_data_plotting notebook to proceed.
