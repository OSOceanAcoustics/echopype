Plotting with echopype
========================

Introduction
-------------
echopype proves an easy to use framework for plotting echograms using ``Process`` objects. Minimal code is required to see Sv, TS, or MVBS data plotted with select frequencies, but as echopype's plotting functionality are simply a thin wrapper around `xarray <http://xarray.pydata.org/en/stable/index.html>`_
plotting functions, which wrap `matplotlib <https://matplotlib.org/>`_  functions, plots can be customized using matplotlib's extensive features.

Plotting an Echogram
----------------------
These examples will use the test datasets in the `echopype <https://github.com/OSOceanAcoustics/echopype>`_
GitHub repository.

AZFP Example
~~~~~~~~~~~~~~
This example will use the test datasets in the `echopype <https://github.com/OSOceanAcoustics/echopype>`_
GitHub repository.

First, create a ``Process`` object for the AZFP data

.. code-block:: console

    import matplotlib.pyplot as plt
    import echopype as ep
    azfp_xml_path = './echopype/test_data/azfp/17041823.XML'
    azfp_01a_path = './echopype/test_data/azfp/17082117.01A'

    # Convert to .nc file
    tmp_convert = ep.convert.Convert(azfp_01a_path, azfp_xml_path)
    tmp_convert.raw2nc()

    # Make Process object
    tmp_echo = ep.process.Process(tmp_convert.nc_path)

Then calibrate the raw data to acquire xarray Datasets to plot

.. code-block:: console

    tmp_echo.calibrate(save=False)       # Sv data
    tmp_echo.calibrate_ts(save=False)    # TS data
    tmp_echo.get_MVBS()                  # MVBS data

Now, make an `EchoGram` object for plotting

.. code-block:: console

    tmp_plot = EchoGram(tmp_echo)

Plotting a Single Channel
++++++++++++++++++++++++++++

For plotting a single frequency, plot the data using ``.plot()`` on the ``Process`` object and specify the frequency to be plotted (in Hz). This example data set contains data collected in bursts. So times where the instrument did not take measurements are blank.

.. code-block:: console

    tmp_plot.plot('Sv', frequency=38000, cmap='jet')
    plt.show()

.. image:: images/azfp_single_no_infer.png

This data can also be represented without the white bars.

.. code-block:: console

    tmp_plot.plot('Sv', frequency=38000, infer_burst=True, cmap='jet')
    plt.show()

.. image:: images/azfp_single_infer.png

Or, by having the x-axis represent the ping number as opped to the ping time.

.. code-block:: console

    tmp_plot.plot('Sv', frequency=38000, plot_ping_number=True, cmap='jet')
    plt.show()

.. image:: images/azfp_single_number.png

Plotting Multiple Channels
+++++++++++++++++++++++++++++

In order to plot multiple frequencies, specify frequency as a ``set`` of frequencies. echopype will plot every frequency in subplots using xarray's ``FacetGrid`` functionality.

.. code-block:: console

    tmp_plot.plot('Sv', frequency={38000, 200000}, cmap='jet')
    plt.show()

.. image:: images/azfp_multi_no_infer.png

Or, simply leave out frequency to plot all channels.

.. code-block:: console

    tmp_plot.plot('Sv', cmap='jet')
    plt.show()

.. image:: images/azfp_all_no_infer.png


EK60 Example
~~~~~~~~~~~~~~

Because EK60 does not collect data in bursts, plotting is even simpler than for the AZFP.

To start, get an ``EchoData`` object for the EK60 data.

.. code-block:: console

    import matplotlib.pyplot as plt
    import echopype as ep
    ek60_raw_path = './echopype/test_data/ek60/DY1801_EK60-D20180211-T164025.raw'

    # Convert to .nc file
    tmp_convert = ep.convert.Convert(ek60_raw_path)
    tmp_convert.raw2nc()

    # Make EchoData object
    tmp_echo = ep.model.EchoData(tmp_convert.nc_path)

Then calibrate the raw data to acquire xarray Datasets to plot

.. code-block:: console

    tmp_echo.calibrate(save=False)       # Sv data
    tmp_echo.calibrate_ts(save=False)    # TS data
    tmp_echo.get_MVBS()                  # MVBS data

Now, make an ``EchoGram`` object for plotting

.. code-block:: console

    tmp_plot = EchoGram(tmp_echo)

Plotting a Single Channel
++++++++++++++++++++++++++++

To plot a single frequency, call ``.plot()`` on the ``EchoGram`` object and specify the frequency to be plotted (in Hz).

.. code-block:: console

    tmp_plot.plot('Sv', frequency=38000, cmap='jet')
    plt.show()

.. image:: images/ek60_single.png



Plotting Multiple Channels
+++++++++++++++++++++++++++++

As for AZFP, plot multiple channels by creating a ``set` of the desired frequencies

.. code-block:: console

    tmp_plot.plot('Sv', frequency={38000, 200000}, cmap='jet')
    plt.show()

.. image:: images/ek60_multi.png

And to see all availible channels, simply leave out the ``frequency`` argument

.. code-block:: console

    tmp_plot.plot('Sv', cmap='jet')
    plt.show()

.. image:: images/ek60_all.png

Other Plotting Routines
~~~~~~~~~~~~~~~~~~~~~~~~~
Having a data structure built off of xarray and numpy means that users are not limited to echopype's plotting methods. Users looking for additional features can use matplotlib's ``pcolormesh`` without wrappers, or other plotting packages such as `Bokeh <https://bokeh.pydata.org/en/latest/>`_ or `hvplot <https://hvplot.pyviz.org/>`_.

Here is an example using hvplot which includes a nifty frequency slider using the `tmp_echo` defined earlier

.. code-block:: console

    import hvplot.xarray
    tmp_echo.Sv.hvplot(y='range_bin', x='ping_time', cmap='jet',width=500, height=400)

.. image:: images/azfp_hvplot.PNG