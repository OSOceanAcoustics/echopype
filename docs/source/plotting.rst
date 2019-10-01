Plotting with echopype
========================

Introduction
-------------
echopype proves an easy to use framework for plotting echograms using ``EchoData`` objects. Minimal code is required to see Sv, TS, or MVBS data plotted with select frequencies, but as echopype's plotting functionality are simply a thin wrapper around `xarray <http://xarray.pydata.org/en/stable/index.html>`_
plotting functions, which wrap `matplotlib <https://matplotlib.org/>`_  functions, plots can be customized using matplotlib's extensive features.

Plotting an Echogram
----------------------

AZFP Example
~~~~~~~~~~~~~~
This example will use the test datasets in the `echopype <https://github.com/OSOceanAcoustics/echopype>`_
GitHub repository.

First, create an ``EchoData`` object for the AZFP data

.. code-block:: console

    import matplotlib.pyplot as plt
    import echopype as ep
    azfp_xml_path = './echopype/test_data/azfp/17041823.XML'
    azfp_01a_path = './echopype/test_data/azfp/17082117.01A'

    # Convert to .nc file
    tmp_convert = ep.convert.Convert(azfp_01a_path, azfp_xml_path)
    tmp_convert.raw2nc()

    # Make EchoData object
    tmp_echo = ep.model.EchoData(tmp_convert.nc_path)

Now, calibrate the raw data to acquire xarray Datasets to plot

.. code-block:: console

    tmp_echo.calibrate(save=False)       # Sv data
    tmp_echo.calibrate_ts(save=False)    # TS data
    tmp_echo.get_MVBS()                  # MVBS data

Plotting a Single Channel
++++++++++++++++++++++++++++

For plotting a single frequency, plot the data using ``.plot()`` on the ``EchoData`` object and specify the frequency to be plotted (in Hz). This example data set contains data collected in bursts. So times where the instrument did not take measurements are blank.

.. code-block:: console

    tmp_plot.plot('Sv', frequency=38000, cmap='jet')
    plt.show()

IMAGE

This data can also be represented without the white bars.

.. code-block:: console

    tmp_plot.plot('Sv', frequency=38000, infer_burst=True, cmap='jet')
    plt.show()

IMAGE

Or, by having the x-axis represent the ping number as opped to the ping time.

.. code-block:: console

    tmp_plot.plot('Sv', frequency=38000, plot_ping_number=True, cmap='jet')
    plt.show()

IMAGE

Plotting Multiple Channels
+++++++++++++++++++++++++++++

In order to plot multiple frequencies, specify frequency as a ``set`` of frequencies. echopype will plot every frequency in subplots using xarray's ``FacetGrid`` functionality.

.. code-block:: console

    tmp_plot.plot('Sv', frequency={38000, 200000}, cmap='jet')
    plt.show()

IMAGE

Or, simply leave out frequency to plot all channels.

.. code-block:: console

    tmp_plot.plot('Sv', cmap='jet')
    plt.show()

IMAGE

Other Plotting Routines
+++++++++++++++++++++++++++
Having a data structure built off of xarray and numpy means that users are not limited to echopype's plotting methods. Users looking for additional features can use matplotlib's ``pcolormesh`` without wrappers, or other plotting packages such as `Bokeh <https://bokeh.pydata.org/en/latest/>`_ or `hvplot <https://hvplot.pyviz.org/>`_.

Here is an example using hvplot which includes a nifty frequency slider using the `tmp_echo` defined earlier

.. code-block:: console

    import hvplot.xarray
    tmp_echo.Sv.hvplot(y='range_bin', x='ping_time', cmap='jet',width=500, height=400)

INTERACTIVE IMAGE
