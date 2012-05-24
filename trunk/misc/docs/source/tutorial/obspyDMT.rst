=========================================================================
ObsPyDMT: Retrieving, Processing and Management of Large Seismic Datasets
=========================================================================

Welcome!

obspyDMT_ (ObsPy Data Management Tool) is a command line tool for retrieving, processing and management of large seismic datasets. This tool was developed to deal with the following tasks in a fully automatic way:

1. Retrieval of waveforms (MSEED or SAC), response files and metadata from IRIS_ and ArcLink_ archives. This could be done for single or large requests.
2. Extracting the information of all the events via user-defined options (time span, magnitude, depth and event location).
3. Supports the event-based and continuous requests (refer to the "Event-Based Request" and "Continuous Request" sections in this tutorial).
4. Updating the existing folders (waveform, response and metadata).
5. Applying "Instrument Correction" to a large dataset. (correction for Displacement, Velocity and Acceleration)
6. Plotting tools: Currently you are able to plot all the events and/or station locations, Ray coverage (event-station pair) and epicentral-distance plots for all archived waveforms.


This tutorial has been divided into the following sections: 

1.  "Let's Get Started": download obspyDMT code and check your local machine for required dependencies.
2.  "Option Types": there are two types of options for obspyDMT which are explained in this section.
3.  "Event-Info Request": if you are looking for some events and you want to get info about them.
4.  "Event-Based Request": retrieve the waveforms, response files and meta-data of all the requested stations for all the events found in the archive.
5.  "Continuous Request": retrieve the waveforms, response files and meta-data of all the requested stations for all the specified time span.
6.  "Geographical Restriction": if you are interested in the events happened in a specific geographical coordinate and/or retrieving the data from the stations in a specific circular or rectangular bounding area.
7.  "Instrument Correction": instrument correction for displacement, velocity and acceleration.
8.  "Update": if you want to continue an interrupted request or complete your existing folder.
9.  "Plot": for an existing folder, you could plot all the events and/or all the stations, ray path for event-station pairs and epicentral-distance/time for the waveforms.
10. "Folder Structure": the way that obspyDMT organize your retrieved and processed data.
11. "Available Options": all options currently available in obspyDMT have been shown in the form of a table.

-----------------
Let's Get Started
-----------------

It won't take too long! To get the code and check the dependencies required for running the code properly, please:

1. download the source code from obspyDMT_ or copy this line to your shell:

::

	$ svn checkout https://svn.obspy.org/trunk/apps/obspyDMT $HOME/obspyDMT

2. go to your obspyDMT directory and run the following commands:

::

	$ cd $HOME/obspyDMT
	$ ./obspyDMT.py --check

------------
Option Types
------------

This is the general way to define your inputs for obspyDMT:

::

	$ ./obspyDMT.py --option-1 'value' --option-2

* option-1 (with value): this value will be stored and be used in the program as an input.
* option-2 (without value): if you enter an option-2 in your request, the required task will be performed. For example, if you enter "--check" (refer to "Let's Get it Started" section), the program will check all the dependencies required for running the code properly.

If you want info about all the available options, you should type:

::

    $ ./obspyDMT.py --help 

or refer to the "Available Options" section in this tutorial. In this section, the options marked with "*" are the first option type (option-1), and the options marked with "**" are the second type (option-2).

ONE GOOD THING: the order of options is commutative!

------------------
Event-Info Request
------------------

In this type of request, obspyDMT will search for all the available events based on the options specified by the user, then print the results and create an event catalogue.

The following lines show how to send an "Event-Info Request" with ObsPyDMT and present some examples.

The general way to define an "Event-Info Request" is:

::

	$ ./obspyDMT.py --iris 'N' --arc 'N' --option-1 'value' --option-2

The flags "--iris 'N'" and "--arc 'N'", force the code to skip the process of retrieving the waveforms, response files and metadata from IRIS and ArcLink since we are only interested in the event info here.
For details on option-1 and option-2 please refer to "Option Types" section.

Example 1: run with the default values:

::

    $ ./obspyDMT.py --iris 'N' --arc 'N'

When the job starts, a folder called "obspyDMT-data" will be created in "$HOME/obspyDMT". This folder will contain the info about the events set in the default request (Please refer to the "Folder Structure" section)

Example 2: by adding flags to the above command, one can change the default values and add/remove functionalities of the code. As an example, the following command shows how to get the info of all the events with magnitude more than Mw 6.0 occured after 2011-03-01 and before 2012-03-01:

::
	
	$ ./obspyDMT.py --min_mag '6.0' --min_date '2011-03-01' --max_date '2012-03-01'

-------------------
Event-Based Request
-------------------

In this type of request, the following steps will be done automatically:

1. Search for all available events based on the options specified by the user.
2. Check the availability of the requested stations for each event.
3. Start to retrieve the waveforms and/or response files for each event and for all available stations. (default: waveforms, response files and metadata will be retrieved.)
4. Instrument Correction to all saved waveforms based on the specified options.

The following lines show how to send an "Event-Based Request" with ObsPyDMT and present two short examples.

The general way to define an "Event-Based Request" is:

::

	$ ./obspyDMT.py --option-1 'value' --option-2

For details on option-1 and option-2 please refer to "Option Types" section.

Example 1: to test the code with the defualt values run:

::

    $ ./obspyDMT.py --test '20'

if you take away the option "--test '20'", the default values could result in a huge amount of requests. This option set the code to send "20" requests to IRIS and ArcLink which is suitable for testing.

When the job starts, a folder called "obspyDMT-data" will be created in "$HOME/obspyDMT" and the data will be organized there. (Please refer to the "Folder Structure" section)

Example 2: by adding flags to the above command, one can change the default values and add/remove functionalities of the code. As an example, the following commands show how to get all the waveforms, response files and metadata of "BHZ" channels available in "TA" network for the great Tohoku-oki earthquake of magnitude Mw 9.0:

::

    $ ./obspyDMT.py --min_mag '8.9' --min_date '2011-03-01' --identity 'TA.*.*.BHZ'

or instead of using "identity" option:

::

    $ ./obspyDMT.py --min_mag '8.9' --min_date '2011-03-01' --net 'TA' --cha 'BHZ'

In the case that you know from which data provider you want to retrieve the data, it is better to exclude the non-related ones. For instance, in this example since we know that "TA" network is within IRIS, it is more wise to exclude ArcLink by:

::

    $ ./obspyDMT.py --min_mag '8.9' --min_date '2011-03-01' --identity 'TA.*.*.BHZ' --arc 'N'

Example 3: for downloading the same event and station identity but "SAC" as the desired waveform format instead of "MSEED" (default):

::

    $ ./obspyDMT.py --min_mag '8.9' --min_date '2011-03-01' --identity 'TA.*.*.BHZ' --arc 'N' --SAC

When obspyDMT wants to save the waveforms in "SAC" format, it will fill out the station location (stla and stlo), station elevation (stel), station depth (stdp), event location (evla and evlo), event depth (evdp) and event magnitude (mag) in the SAC headers.

Example 4: for downloading just the raw waveforms without response file and instrument correction:

::

    $ ./obspyDMT.py --min_mag '8.9' --min_date '2011-03-01' --identity 'TA.Z*.*.BHZ' --arc 'N' --SAC --response 'N' --ic_no

Example 5: the default values for the preset (how close the time series data (waveform) will be cropped before the origin time of the event) and the offset (how close the time series data (waveform) will be cropped after the origin time of the event) are 0 and 1800 seconds. You could change them by adding the following flags:

::

	$ ./obspyDMT.py --preset time_before --offset time_after --option-1 value --option-2 

------------------
Continuous Request
------------------

In this type of request, the following steps will be done automatically:

1. Get the time span from input and in case of the large time span, divide it into small intervals.
2. Check the availability of the requested stations for each interval.
3. Start to retrieve the waveforms and/or response files for each interval and for all the available stations. (default: waveforms, response files and metadata will be retrieved.)
4. Instrument Correction to all saved waveforms based on the specified options.
5. Merging the small retrieved time intervals to get the original input time span and save the final product.

The following lines show how to send a "Continuous Request" with ObsPyDMT and present two short examples.

The general way to define a "Continuous Request" is:

::

	$ ./obspyDMT.py --continuous --option-1 value --option-2

For details on option-1 and option-2 please refer to "Option Types" section.

Example 1: to test the code with the defualt values run:

::

    $ ./obspyDMT.py --continuous --test '20'

if you take away the option "--test '20'", the default values could result in a huge amount of requests. This option set the code to send "20" requests to IRIS and ArcLink which is suitable for testing.

When the job starts, a folder called "obspyDMT-data" will be created in "$HOME/obspyDMT" and the data will be organized there. (Please refer to the "Folder Structure" section)

Example 2: by adding flags to the above command, one can change the default values and add/remove functionalities of the code. As an example, the following command lines show how to get all the waveforms, response files and metadata of the "BHZ" channels available in "TA" network for the specified time span:

::

    $ ./obspyDMT.py --continuous --identity 'TA.*.*.BHZ' --min_date '2011-01-01' --max_date '2011-02-01'

or instead of using "identity" option:

::

    $ ./obspyDMT.py --continuous --net 'TA' --cha 'BHZ' --min_date '2011-01-01' --max_date '2011-02-01'

In the case that you know from which data provider you want to retrieve the data, it is better to exclude the non-related ones. For instance, in this example since we know that "TA" network is within IRIS, it is more wise to exclude ArcLink by:

::

    $ ./obspyDMT.py --continuous --identity 'TA.*.*.BHZ' --arc 'N' --min_date '2011-01-01' --max_date '2011-02-01'

Example 3: for downloading the same time span and station identity but "SAC" as the desired waveform format instead of "MSEED" (default):

::

    $ ./obspyDMT.py --continuous --identity 'TA.*.*.BHZ' --arc 'N' --min_date '2011-01-01' --max_date '2011-02-01' --SAC

When obspyDMT wants to save the waveforms in "SAC" format, it will fill out the station location (stla and stlo), station elevation (stel) and station depth (stdp) in the SAC headers.

Example 4: for downloading just the raw waveforms without response file and instrument correction:

::

    $ ./obspyDMT.py --continuous --identity 'TA.*.*.BHZ' --arc 'N' --min_date '2011-01-01' --max_date '2011-02-01' --SAC --response 'N' --ic_no

------------------------
Geographical Restriction
------------------------

If you are interested in the events happened in a specific geographical coordinate and/or retrieving the data from the stations in a specific circular or rectangular bounding area, you are in the right section! Here, we have two examples:

Example 1: to extract the info of all the events occured in 2010 in a rectangular area (lon1=44.38E lon2=63.41E lat1=24.21N lat2=40.01N) with magnitude more than 3.0 and maximum depth of 80 km:

::

    $ ./obspyDMT.py --iris 'N' --arc 'N' --min_mag '3.0' --max_depth '-80.0' --min_date '2010-01-01' --max_date '2011-01-01' --event_rect '44.38/63.41/24.21/40.01'

Example 2: to get all the waveforms, response files and metadata of "BHZ" channels available in a specified rectangular bounding area (lon1=125.0W lon2=70.0W lat1=25N lat2=45N) for the great Tohoku-oki earthquake of magnitude Mw 9.0, the command line will be:

::

    $ ./obspyDMT.py --min_mag '8.9' --min_date '2011-03-01' --cha 'BHZ' --station_rect '125.0/70.0/25.0/45.0'

---------------------
Instrument Correction
---------------------

When obspyDMT retrieves waveforms and their response files, by default it applies the instrument correction to the waveform with displacement as the correction unit. To change the correction unit to Velocity or Acceleration:

::

	$ ./obspyDMT.py --corr_unit 'VEL' --option-1 'value' --option-2
	$ ./obspyDMT.py --corr_unit 'ACC' --option-1 'value' --option-2

where option-1 and option-2 are the ones related to your requests as was shown in the previous sections.

Please note that all the commands presented in this section could be applied to "Continuous Request" as well but with slightly changes (refer to the "Continuous Request" section).

Before applying the instrument correction, a bandpass filter will be applied to the data with this default values: '(0.008, 0.012, 3.0, 4.0)'. If you want to apply another band pass filter:

::

	$ ./obspyDMT.py --pre_filt '(f1,f2,f3,f4)' --option-1 value --option-2

where (f1,f2,f3,f4) are the four corner frequencies of a cosine taper, one between f2 and f3 and tapers to zero for f1 < f < f2 and f3 < f < f4.

If you do not need the pre filter:

::

	$ ./obspyDMT.py --pre_filt 'None' --option-1 value --option-2

You could idle the instrument correction functionallity by:

::

	$ ./obspyDMT.py --ic_no --option-1 value --option-2

In case that you want to apply instrument correction to an existing folder:

::

	$ ./obspyDMT.py --ic_all 'address' --corr_unit unit

here "address" is the path where your nots-corrected waveforms are stored.
as mentioned above, "unit" is the unit that you want to correct the waveforms to. It could be "DIS" (default), "VEL" or "ACC".

To make it more clear, let's take a look at an example with following 2 steps:

Step 1: to get all the waveforms, response files and metadata of "BHZ" channels available in "TA" network for the great Tohoku-oki earthquake of magnitude Mw 9.0 you type:

::

    $ ./obspyDMT.py --min_mag '8.9' --min_date '2011-03-01' --identity 'TA.*.*.BHZ' --arc 'N'

Step 2: to correct the raw waveforms for velocity already stored in the default path ("./obspyDMT-data"):

::

    $ ./obspyDMT.py --ic_all './obspyDMT-data' --corr_unit 'VEL'

------
Update
------

If you want to continue an interrupted request or complete your existing folder, you could use the updating option. The general ways to update an existing folder (located in "address") for IRIS stations, ArcLink stations or both are:

::

	$ ./obspyDMT.py --iris_update 'address' --option-1 value --option-2
	$ ./obspyDMT.py --arc_update 'address' --option-1 value --option-2
	$ ./obspyDMT.py --update_all 'address' --option-1 value --option-2

Please note that all the commands presented in this section could be applied to "Continuous Request" as well but with slightly changes (refer to the "Continuous Request" section).

Example 1: first, lets retrieve all the waveforms, response files and metadata of "BHZ" channels available in "TA" network for the great Tohoku-oki earthquake of magnitude Mw 9.0, the command line will be:

::

    $ ./obspyDMT.py --min_mag '8.9' --min_date '2011-03-01' --identity 'TA.*.*.BHZ' --arc 'N'

now, we want to update the saved folder for "BHE" channels:

	$ ./obspyDMT.py --update_all './obspyDMT-data' --identity 'TA.*.*.BHE'

----
Plot
----

For an existing folder, you could plot all the events and/or all the stations, ray path for event-station pairs and epicentral-distance/time for the waveforms.

The general syntax for plotting tools is: 

::

    $ ./obspyDMT.py --plot_option 'address'

that "--plot_option" could be "--plot_ev" for events, "--plot_sta" for stations, "--plot_se" for stations and events, "--plot_ray" for event-station pairs and "--plot_epi" for epicentral-distance/time. 

All the examples showed in this section are based on the folder created by the following request:

::

    $ ./obspyDMT.py --min_mag '8.9' --min_date '2011-03-01' --identity 'TA.*.*.BHZ' --arc 'N'

Example 1: let's plot both stations and events available in the folder:

::

    $ ./obspyDMT.py --plot_se './obspyDMT-data'

the default format is "png", but assume that we want "pdf" for our figures, then:

::

    $ ./obspyDMT.py --plot_se './obspyDMT-data' --plot_format 'pdf'

Example 2: in this example, we want to plot the ray path for event-station pairs but save the result in "$HOME/Desktop":

::

    $ ./obspyDMT.py --plot_ray './obspyDMT-data' --plot_format 'pdf' --plot_save '$HOME/Desktop'

----------------
Folder Structure
----------------

Here, we will talk more about how obspyDMT organizes your retrieved and processed data in your local machine. Basically, when you want to run the code, you could specify a directory in which all the data will be organized:

::

    $ ./obspyDMT.py --datapath './mydata'

obspyDMT will create the folder ("mydata") in the desired address and then start to create folders and files during retrieving and processing as it has been shown in the table below: 

"Under Construction"

-----------------
Available Options
-----------------

All the options currently available in obspyDMT are shown in the table below. Additionally, they could be seen by:

In the description part, options have been marked by (*) or (**) which are:

(*): option type 1 (with value)
(**): option type 2 (without value)

Please refer to the "Option Types" section for more info about type 1 and type 2

+-----------------------+-----------------------+---+-----------------------+-----------------------+
| options               | description           |   | options               | description           |
+=======================+=======================+===+=======================+=======================+
| --help                | show all the available|   | --test                | test the program for  |
|                       | flags with a short    |   |                       | the desired number of |
|                       | description for each  |   |                       | requests, eg:         |
|                       | and exit (**)         |   |                       | '--test 10' will test |
|                       |                       |   |                       | the program for 10    |
|                       |                       |   |                       | requests.             |
|                       |                       |   |                       | [Default: 'N'] (*)    |
+-----------------------+-----------------------+---+-----------------------+-----------------------+
| --version             | show the obspyDMT     |   | --iris_update         | update the specified  |
|                       | version and exit (**) |   |                       | folder for IRIS,      |
|                       |                       |   |                       | syntax:               |
|                       |                       |   |                       | --iris_update         |
|                       |                       |   |                       | address_of_the        |
|                       |                       |   |                       | _target_folder.       |
|                       |                       |   |                       | [Default: 'N'] (*)    |
+-----------------------+-----------------------+---+-----------------------+-----------------------+
| --check               | check all the         |   | --arc_update          | update the specified  |
|                       | dependencies and      |   |                       | folder for ArcLink,   |
|                       | their installed       |   |                       | syntax:               |
|                       | versions on the       |   |                       | --arc_update          |
|                       | local machine         |   |                       | address_of_the        |
|                       | and exit (**)         |   |                       | _target_folder.       |
|                       |                       |   |                       | [Default: 'N'] (*)    |
+-----------------------+-----------------------+---+-----------------------+-----------------------+
| --type                | type of the input     |   | --update_all          | update the specified  |
|                       | ('command' or 'file') |   |                       | folder for both IRIS  |
|                       | to be read            |   |                       | and ArcLink,          |
|                       | by obspyDMT. Please   |   |                       | syntax: --update_all  |
|                       | note that for         |   |                       | address_of_the        |
|                       | "--type 'file'" an    |   |                       | _target_folder.       |
|                       | external file         |   |                       | [Default: 'N'] (*)    |
|                       | ('INPUT.cfg') should  |   |                       |                       |
|                       | exist in the same     |   |                       |                       |
|                       | directory as          |   |                       |                       |
|                       | obspyDMT.py           |   |                       |                       |
|                       | [Default: command] (*)|   |                       |                       |
+-----------------------+-----------------------+---+-----------------------+-----------------------+
| --reset               | if the datapath is    |   | --iris_ic             | apply instrument      |
|                       | found deleting it     |   |                       | correction to the     |
|                       | before running        |   |                       | specified folder for  |
|                       | obspyDMT. (**)        |   |                       | the downloaded        |
|                       |                       |   |                       | waveforms from        |
|                       |                       |   |                       | IRIS, syntax:         |
|                       |                       |   |                       | --iris_ic address_of  |
|                       |                       |   |                       | _the_target_folder.   |
|                       |                       |   |                       | [Default: 'N'] (*)    |
+-----------------------+-----------------------+---+-----------------------+-----------------------+
| --datapath            | the path where        |   | --arc_ic              | apply instrument      |
|                       | obspyDMT will store   |   |                       | correction to the     |
|                       | the data [Default:    |   |                       | specified folder for  |
|                       | './obspyDMT-data'] (*)|   |                       | the downloaded        |
|                       |                       |   |                       | waveforms from        |
|                       |                       |   |                       | ArcLink, syntax:      |
|                       |                       |   |                       | --arc_ic address_of   |
|                       |                       |   |                       | _the_target_folder.   |
|                       |                       |   |                       | [Default: 'N'] (*)    |
+-----------------------+-----------------------+---+-----------------------+-----------------------+
| --min_date            | start time, syntax:   |   | --iris_ic_auto        | apply instrument      |
|                       | Y-M-D-H-M-S (eg:      |   |                       | correction            |
|                       | '2010-01-01-00-00-00')|   |                       | automatically after   |
|                       | or just Y-M-D         |   |                       | downloading the       |
|                       | [Default: 10 days ago]|   |                       | waveforms from IRIS.  |
|                       | (*)                   |   |                       | [Default: 'Y'] (*)    |
+-----------------------+-----------------------+---+-----------------------+-----------------------+
| --max_date            | end time, syntax:     |   | --arc_ic_auto         | apply instrument      |
|                       | Y-M-D-H-M-S (eg:      |   |                       | correction            |
|                       | '2011-01-01-00-00-00')|   |                       | automatically after   |
|                       | or just Y-M-D         |   |                       | downloading the       |
|                       | [Default: 5 days ago] |   |                       | waveforms from        |
|                       | (*)                   |   |                       | ArcLink.              |
|                       |                       |   |                       | [Default: 'Y'] (*)    |
+-----------------------+-----------------------+---+-----------------------+-----------------------+
| --min_mag             | minimum magnitude.    |   | --ic_all              | apply instrument      |
|                       | [Default: 5.5]        |   |                       | correction to the     |
|                       | (*)                   |   |                       | specified folder      |
|                       |                       |   |                       | for all the waveforms |
|                       |                       |   |                       | (IRIS and ArcLink),   |
|                       |                       |   |                       | syntax: --ic_all      |
|                       |                       |   |                       | address_of_the        |
|                       |                       |   |                       | _target_folder.       |
|                       |                       |   |                       | [Default: 'N'] (*)    |
+-----------------------+-----------------------+---+-----------------------+-----------------------+
| --max_mag             | maximum magnitude.    |   | --ic_no               | do not apply          |
|                       | [Default: 9.9]        |   |                       | instrument correction |
|                       | (*)                   |   |                       | automatically.        |
|                       |                       |   |                       | This is equivalent    |
|                       |                       |   |                       | to: "--iris_ic_auto N |
|                       |                       |   |                       | --arc_ic_auto N" (**) |
+-----------------------+-----------------------+---+-----------------------+-----------------------+
| --min_depth           | minimum depth.        |   | --pre_filt            | apply a bandpass      |
|                       | [Default: +10.0       |   |                       | filter to the data    |                                          
|                       | (above the surface!)] |   |                       | trace before          |               
|                       | (*)                   |   |                       | deconvolution         |
|                       |                       |   |                       | ('None' if you do not |
|                       |                       |   |                       | need pre_filter),     | 
|                       |                       |   |                       | syntax:               |
|                       |                       |   |                       | '(f1,f2,f3,f4)' which |
|                       |                       |   |                       | are the four corner   |
|                       |                       |   |                       | frequencies of a      |
|                       |                       |   |                       | cosine taper, one     |
|                       |                       |   |                       | between f2 and f3     |
|                       |                       |   |                       | and tapers to zero    |
|                       |                       |   |                       | for f1 < f < f2 and   |
|                       |                       |   |                       | f3 < f < f4.          |
|                       |                       |   |                       | [Default:             |
|                       |                       |   |                       | '(0.008, 0.012, 3.0,  |
|                       |                       |   |                       | 4.0)'] (*)            |
+-----------------------+-----------------------+---+-----------------------+-----------------------+
| --max_depth           | maximum depth.        |   | --corr_unit           | correct the raw       |
|                       | [Default: -6000.0]    |   |                       | waveforms for DIS (m),| 
|                       | (*)                   |   |                       | VEL (m/s) or          |
|                       |                       |   |                       | ACC (m/s^2).          |
|                       |                       |   |                       | [Default: DIS] (*)    |
+-----------------------+-----------------------+---+-----------------------+-----------------------+
| --event_rect          | search for all the    |   | --zip_w               | compress the          |
|                       | events within the     |   |                       | raw-waveform files    |                                            
|                       | defined rectangle,    |   |                       | after applying        |                                         
|                       | GMT syntax:           |   |                       | instrument correction.|                                         
|                       | <lonmin>/<lonmax>/    |   |                       | (**)                  |                            
|                       | <latmin>/<latmax>     |   |                       |                       |                            
|                       | [Default:             |   |                       |                       |                    
|                       | -180.0/+180.0         |   |                       |                       |                       
|                       | /-90.0/+90.0] (*)     |   |                       |                       |   
+-----------------------+-----------------------+---+-----------------------+-----------------------+
| --max_result          | maximum number of     |   | --zip_r               | compress the response |
|                       | events to be          |   |                       | files after applying  |                                         
|                       | requested.            |   |                       | instrument correction.|                                        
|                       | [Default: 2500] (*)   |   |                       | (**)                  |   
+-----------------------+-----------------------+---+-----------------------+-----------------------+
| --get_events          | event-based request   |   | --iris_merge          | merge the IRIS        |
|                       | (please refer to      |   |                       | waveforms in the      |                                         
|                       | the tutorial).        |   |                       | specified folder,     |                                        
|                       | [Default: 'Y'] (*)    |   |                       | syntax: --iris_merge  |                
|                       |                       |   |                       | address_of_the        |
|                       |                       |   |                       | _target_folder.       |
|                       |                       |   |                       | [Default: 'N'] (*)    |
+-----------------------+-----------------------+---+-----------------------+-----------------------+
| --continuous          | continuous request    |   | --arc_merge           | merge the ArcLink     |
|                       | (please refer to the  |   |                       | waveforms in the      |                                             
|                       | tutorial). (**)       |   |                       | specified folder,     |         
|                       |                       |   |                       | syntax: --arc_merge   |
|                       |                       |   |                       | address_of_the        |
|                       |                       |   |                       | _target_folder.       |
|                       |                       |   |                       | [Default: 'N'] (*)    |
+-----------------------+-----------------------+---+-----------------------+-----------------------+
| --interval            | time interval for     |   | --iris_merge_auto     | merge automatically   |
|                       | dividing the          |   |                       | after downloading     |                                      
|                       | continuous request.   |   |                       | the waveforms from    |                                             
|                       | [Default: 86400 sec   |   |                       | IRIS.                 |                                 
|                       | (1 day)] (*)          |   |                       | [Default: 'Y'] (*)    |           
+-----------------------+-----------------------+---+-----------------------+-----------------------+
| --iris_bulk           | using the IRIS        |   | --arc_merge_auto      | merge automatically   |
|                       | bulkdataselect        |   |                       | after downloading     |                                        
|                       | Web service.          |   |                       | the waveforms         |                                  
|                       | Since this method     |   |                       | from ArcLink.         |                                       
|                       | returns multiple      |   |                       | [Default: 'Y'] (*)    |                                        
|                       | channels of time      |   |                       |                       |                           
|                       | series data for       |   |                       |                       |                          
|                       | specified time ranges |   |                       |                       |                                
|                       | in one request,       |   |                       |                       |                          
|                       | it speeds up the      |   |                       |                       |                           
|                       | waveform retrieving   |   |                       |                       |                              
|                       | approximately by      |   |                       |                       |                           
|                       | a factor of two.      |   |                       |                       |                           
|                       | [RECOMMENDED] (**)    |   |                       |                       | 
+-----------------------+-----------------------+---+-----------------------+-----------------------+
| --waveform            | retrieve the waveform.|   | --merge_all           | merge all waveforms   |
|                       | [Default: 'Y'] (*)    |   |                       | (IRIS and ArcLink) in |
|                       |                       |   |                       | the specified folder, |
|                       |                       |   |                       | syntax: --merge_all   |
|                       |                       |   |                       | address_of_the        |
|                       |                       |   |                       | _target_folder.       |
|                       |                       |   |                       | [Default: 'N'] (*)    |
+-----------------------+-----------------------+---+-----------------------+-----------------------+
| --response            | retrieve the response |   | --merge_no            | do not merge          |
|                       | file. [Default: 'Y']  |   |                       | automatically. This is| 
|                       | (*)                   |   |                       | equivalent to:        |
|                       |                       |   |                       | "--iris_merge_auto N  |
|                       |                       |   |                       | --arc_merge_auto N"   |
|                       |                       |   |                       | (**)                  |
+-----------------------+-----------------------+---+-----------------------+-----------------------+
| --iris                | send request          |   | --merge_type          | merge 'raw' or        |
|                       | (waveform/response)   |   |                       | 'corrected' waveforms.|                                                  
|                       | to IRIS.              |   |                       | [Default: 'raw']      |                                  
|                       | [Default: 'Y'] (*)    |   |                       | (*)                   | 
+-----------------------+-----------------------+---+-----------------------+-----------------------+
| --arc                 | send request          |   | --plot_iris           | plot waveforms        |
|                       | (waveform/response)   |   |                       | downloaded from IRIS. |                                                 
|                       | to ArcLink.           |   |                       | (*)                   |                      
|                       | [Default: 'Y'] (*)    |   |                       |                       | 
+-----------------------+-----------------------+---+-----------------------+-----------------------+
| --SAC                 | SAC format for saving |   | --plot_arc            | plot waveforms        |
|                       | the waveforms. Station|   |                       | downloaded from       |                                              
|                       | location (stla and    |   |                       | ArcLink. (*)          |                                    
|                       | stlo), station        |   |                       |                       |                         
|                       | elevation (stel),     |   |                       |                       |                            
|                       | station depth (stdp), |   |                       |                       |                                
|                       | event location (evla  |   |                       |                       |                               
|                       | and evlo), event depth|   |                       |                       |                                 
|                       | (evdp) and event      |   |                       |                       |                           
|                       | magnitude (mag) will  |   |                       |                       |                               
|                       | be stored in the SAC  |   |                       |                       |                               
|                       | headers.              |   |                       |                       |                   
|                       | [Default: MSEED] (**) |   |                       |                       | 
+-----------------------+-----------------------+---+-----------------------+-----------------------+
| --time_iris           | generate a data-time  |   | --plot_all            | plot all waveforms    |
|                       | file for an IRIS      |   |                       | (IRIS and ArcLink).   |                                            
|                       | request. This file    |   |                       | [Default: 'Y'] (*)    |                                          
|                       | shows the required    |   |                       |                       |                             
|                       | time for each request |   |                       |                       |                                
|                       | and the stored data   |   |                       |                       |                              
|                       | in the folder. (**)   |   |                       |                       |
+-----------------------+-----------------------+---+-----------------------+-----------------------+
| --time_arc            | generate a data-time  |   | --plot_type           | plot 'raw' or         |
|                       | file for an ArcLink   |   |                       | 'corrected' waveforms.|                                                  
|                       | request. This file    |   |                       | [Default: 'raw'] (*)  |                                                
|                       | shows the required    |   |                       |                       |                             
|                       | time for each request |   |                       |                       |                                
|                       | and the stored data   |   |                       |                       |                              
|                       | in the folder. (**)   |   |                       |                       |
+-----------------------+-----------------------+---+-----------------------+-----------------------+
| --preset              | time parameter in     |   | --plot_ev             | plot all the events   |
|                       | seconds which         |   |                       | found in the specified|                                            
|                       | determines how close  |   |                       | folder, syntax:       |                                            
|                       | the time series data  |   |                       | --plot_ev address_of  |                                                 
|                       | (waveform) will be    |   |                       | _the_target_folder.   |                                             
|                       | cropped before the    |   |                       | [Default: 'N'] (*)    |
|                       | origin time of the    |   |                       |                       |
|                       | event.                |   |                       |                       |
|                       | [Default: 0.0 seconds.|   |                       |                       |
|                       | ] (*)                 |   |                       |                       |
+-----------------------+-----------------------+---+-----------------------+-----------------------+
| --offset              | time parameter in     |   | --plot_sta            | plot all the stations |
|                       | seconds which         |   |                       | found in the specified|                                            
|                       | determines how close  |   |                       | folder, syntax:       |                                            
|                       | the time series data  |   |                       | --plot_sta address_of |                                                  
|                       | (waveform) will be    |   |                       | _the_target_folder.   |                                             
|                       | cropped after the     |   |                       | [Default: 'N'] (*)    |                                         
|                       | origin time of the    |   |                       |                       |                             
|                       | event.                |   |                       |                       |                 
|                       | [Default:             |   |                       |                       |                   
|                       | 1800.0 seconds.] (*)  |   |                       |                       |
+-----------------------+-----------------------+---+-----------------------+-----------------------+
| --identity            | identity code         |   | --plot_se             | plot both all the     |
|                       | restriction, syntax:  |   |                       | stations and all the  |                                                 
|                       | net.sta.loc.cha       |   |                       | events found in the   |                                           
|                       | (eg: TA.*.*.BHZ to    |   |                       | specified folder,     |                                            
|                       | search for all BHZ    |   |                       | syntax: --plot_se     |                                            
|                       | channels in           |   |                       | address_of_the_target |                                         
|                       | TA network).          |   |                       | _folder.              |                                         
|                       | [Default: *.*.*.*] (*)|   |                       | [Default: 'N'] (*)    | 
+-----------------------+-----------------------+---+-----------------------+-----------------------+
| --net                 | network code.         |   | --plot_ray            | plot the ray coverage |
|                       | [Default: "*"] (*)    |   |                       | for all the           |
|                       |                       |   |                       | station-event pairs   |
|                       |                       |   |                       | found in the specified| 
|                       |                       |   |                       | folder, syntax:       |
|                       |                       |   |                       | --plot_ray address    |
|                       |                       |   |                       | _of_the_target_folder.|
|                       |                       |   |                       | [Default: 'N'] (*)    |
+-----------------------+-----------------------+---+-----------------------+-----------------------+
| --sta                 | station code.         |   | --plot_epi            | plot "epicentral      |
|                       | [Default: "*"] (*)    |   |                       | distance-time" for all| 
|                       |                       |   |                       | the waveforms found in| 
|                       |                       |   |                       | the specified folder, |
|                       |                       |   |                       | syntax: --plot_epi    |
|                       |                       |   |                       | address_of_the_target |
|                       |                       |   |                       | _folder.              |
|                       |                       |   |                       | [Default: 'N'] (*)    |
+-----------------------+-----------------------+---+-----------------------+-----------------------+
| --loc                 | location code.        |   | --min_epi             | plot "epicentral      |
|                       | [Default: "*"] (*)    |   |                       | distance-time"        |
|                       |                       |   |                       | (refer to             |
|                       |                       |   |                       | '--plot_epi') for all |
|                       |                       |   |                       | the waveforms with    |
|                       |                       |   |                       | epicentral-distance >=| 
|                       |                       |   |                       | min_epi.              |
|                       |                       |   |                       | [Default: 0.0] (*)    |
+-----------------------+-----------------------+---+-----------------------+-----------------------+
| --cha                 | channel code.         |   | --max_epi             | plot "epicentral      |
|                       | [Default: "*"] (*)    |   |                       | distance-time"        |
|                       |                       |   |                       | (refer to             |
|                       |                       |   |                       | '--plot_epi') for all |
|                       |                       |   |                       | the waveforms with    |
|                       |                       |   |                       | epicentral-distance <=| 
|                       |                       |   |                       | max_epi.              |
|                       |                       |   |                       | [Default: 180.0] (*)  |
+-----------------------+-----------------------+---+-----------------------+-----------------------+
| --station_rect        | search for all the    |   | --plot_save           | the path where        |
|                       | stations within the   |   |                       | obspyDMT will store   |                                               
|                       | defined rectangle,    |   |                       | the plots             |                                    
|                       | GMT syntax:           |   |                       | [Default: '.'         |                                 
|                       | <lonmin>/<lonmax>/    |   |                       | (the same directory   |                                             
|                       | <latmin>/<latmax>.    |   |                       | as obspyDMT.py)] (*)  |                                                
|                       | May not be used       |   |                       |                       |                          
|                       | together with circular|   |                       |                       |                                 
|                       | bounding box station  |   |                       |                       |                               
|                       | restrictions          |   |                       |                       |                       
|                       | (station_circle)      |   |                       |                       |                           
|                       | [Default:             |   |                       |                       |                    
|                       | -180.0/+180.0/        |   |                       |                       |                        
|                       | -90.0/+90.0] (*)      |   |                       |                       |
+-----------------------+-----------------------+---+-----------------------+-----------------------+
| --station_circle      | search for all the    |   | --plot_format         | format of the plots   |
|                       | stations within the   |   |                       | saved on the local    |                                              
|                       | defined circle,       |   |                       | machine               |                               
|                       | syntax:               |   |                       | [Default: 'png'] (*)  |                                     
|                       | <lon>/<lat>/          |   |                       |                       |                       
|                       | <rmin>/<rmax>.        |   |                       |                       |                          
|                       | May not be used       |   |                       |                       |                           
|                       | together with         |   |                       |                       |                         
|                       | rectangular bounding  |   |                       |                       |                                
|                       | box station           |   |                       |                       |                       
|                       | restrictions          |   |                       |                       |                        
|                       | (station_rect). (*)   |   |                       |                       |    
+-----------------------+-----------------------+---+-----------------------+-----------------------+
| --email               | send an email to the  |   |                       |                       |          
|                       | specified             |   |                       |                       |          
|                       | email-address after   |   |                       |                       |          
|                       | completing the job,   |   |                       |                       |          
|                       | syntax:               |   |                       |                       |          
|                       | --email email_address.|   |                       |                       |          
|                       | [Default: 'N'] (*)    |   |                       |                       |          
+-----------------------+-----------------------+---+-----------------------+-----------------------+

.. _obspyDMT: http://obspy.org/browser/obspy/trunk/apps/obspyDMT/obspyDMT.py
.. _IRIS: http://www.iris.edu/ws/
.. _ArcLink: http://www.webdc.eu/arclink/query?sesskey=aa048d3c
