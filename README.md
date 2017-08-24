# OpenEnded

This repository contains the Python2 code for the UnADevs Method: Unbounded Unsupervised Activity Discovery using the Temporal Behaviour Assumption.

The method is based on online clustering, and additinally includes the temporal information of the occuring activities. Therefore, it is able to discover clusters of repating/periodic activities as the occur and additionally keeps track of the time interval of the discovered cluster. This way, the system may prompt the user about the discovered activity with the appropriate time interval and ask for feedback.

The detailed description of the algorithm is in the paper "Unsupervised Online Activity Discovery Using Temporal Behaviour Assumption". [1]

More information about the projet: http://www.sussex.ac.uk/strc/research/wearable/research-ll

<li>Instructions on how to run the model are included in the <b>JSI-release notebook</b>. It is an  example code run on a Subject 1 from the JSI-ADL dataset: https://github.com/sussexwearlab/OpenEnded/blob/master/JSI-release.ipynb</li>

The algorithm that performs the clustering is in a separate Python file, which is called in the <b>JSI-release notebook</b>: <b>"Online_temporal_clustering_release.py"</b>.

<b>"Utilities_JSI_release.py"</b> is the file that contains the modules that do validation and visualization of the data.

The folder <b>"data_JSI"</b> contains the feature vectors from the 10 subjects from the JSI-ADL dataset

<b>"Preprocessing"</b> folder contains an example on how to create the feature vectors from an example raw data file. 

<b>Please cite:</b><br>
[1] H. Gjoreski, D. Roggen. Unsupervised Online Activity Discovery Using Temporal Behaviour Assumption. In: 21th International Symposium Wearable Computers (ISWC) 2017, 11-15 September 2017, Maui, Hawaii, USA
