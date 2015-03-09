Authors
========

Akhilesh Gadde
Sai Santhosh Vaidyam Anandan
Satya Hara Pavan Kumar Maguluri
Sowmiya Narayan Srinath
Sri Sruthi Mandadapu

Pre-requisites
===============

The following python packages need to be installed to run the application:

opencv-python-2.4.10
matplotlib-1.4.2
numpy-1.9.1
pandas-0.15.0
Pillow-2.6.1
pip-1.5.6
pyparsing-2.0.3
python-dateutil-2.2
pytz-2014.7
scikit-learn-0.15.2
scipy-0.14.0
six-1.8.0
python-tkinter
python-setuptools

Most of these may be installed with anaconda and may become available after 
loading the module:
$ module load anaconda/2.1.0 

Execution
==========

After loading the modules, the application can be launched as follows:

$ python featureAnalytics.py

It may take some time for the feature calculation to complete and open the GUI.
Input boxes have been provided to accept the users choice of clustering algorithm, number 
of clusters and initial inputs epsilon and Minimum points (for DBSCAN).
All selected features must be filtered with required values (using the < and > text boxes).
For unfiltered selection, < 1000 and > 0 is a good value for any of the features.
Click on "Generate the cluster" after selecting what features to display in the graph.
The masked slide image is saved to the same location as the application under the name
"maskCluster.jpg"

Limitations
============
We do not currently support a Save file feature for the user to select the location of the
saved image.
Erroneous input simply results in no cluster graph/image being generated. Please check the
filter values and ensure some nuclei are in the input range if no graph is generated.
Due to some limitations of python-tkinter, the application is known to not terminate safely
on some systems after closing the main GUI window. You may need to kill the python process
manually in such cases.
