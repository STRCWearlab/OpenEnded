The files in this folder correspond to the JSI-ADL dataset [1][2].
Each file contains the extracted features from the wrist-worn 3 axial accelerometer.
To segment the data and extract the features, a sliding window is used with size 2 seconds, and 1 second overlap.

The features are extracted in the following order for each of the axes (X, Y, Z) and for the length of the acceleration vector:
 - mean, standard deviation, energy, median, root mean square, integral,  kurtosis, skewness

The format is the following:
[timestamp, AccX_mean, AccX_std, AccX_energy, AccX_median, AccX_rms, AccX_integral, ..., Activity]

The Activities are the following:
• 'standing': 0
• 'lying_excercising': 1
• 'walking': 2
• 'cycling': 3
• 'working_pc': 4 • 'sitting': 5
• 'transition': 6
• 'shovelling': 7
• 'washing_dishes': 8
• 'running': 9
• 'allfours_move': 10
• 'lying_back': 11
• 'allfours': 12
• 'kneeling': 13
• 'scrubbing_floor': 14

[1] H. Gjoreski et al. “Context-based ensemble method for human energy expenditure estimation”. Appl. Soft Comput., vol. 37, pp. 960–970, 2015.
[2] B. Cvetković, R. Milić, and M. Luštrek, “Estimating Energy Expenditure with Multiple Models using Different Wearable Sensors,” IEEE J. Biomed. Heal. informatics, vol. 20, 2016.