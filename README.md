# TravelRecommendation

This is code for research in 'Personalized and Timed Travel Path Recommendation Based on GPS Data Using Deep Learning'.
Both .py files and .ipynb files are included, but using .ipynb is recommended since it is convenient to see the result of the code.

## Clustering
- Requirements: Installation of Python, Numpy, Matplotlib, Scikit-learn, SciPy, Pandas
- Edit_Distance_Hierarchical_Clustering.ipynb
	- Calculates the edit distance between string sequences and outputs hierarchical clustering result using edit distance
- DBSCAN_cosine_sim.ipynb / K-Means_cosine_sim.ipynb / Hierarchical_cosine_sim.ipynb
	- Each code can be used to calculate cosine similarity between string sequences and output the results of applying DBSCAN, K-Means, and Hierarchical clustering.
	- Also, a graph visualizing the silhouette coefficient of the clustering result can be generated. 
- Clustering_test.ipynb
	- Applies K-Means clustering to numerical sequences
- RelaxedLCS.ipynb
	- Calculates relaxed version of longest common subsequence of string sequences

## Data Processing
- Requirements: Installation of Python, Numpy, Scikit-mobility, Haversine, Geopandas, Pandas, Matplotlib, Shapely.geometry
- calculate_coord.py
	- Calculates distance between two coordinates
- noise_filtering.py
	- Removes GPS points that are considered as noise
- read_geolife.py
	- Reads geolife data and process it into the necessary form
- stop_detection.py
	- Detects stay points using scikit-mobility
- trj_separation.py
	- Separates trajectories that have interval more than 3 hours
- within_beijing.py
	- Removes coordinates that are not in beijing

## LSTM
- Requirements: Installation of Python, Tensorflow, Keras, Numpy, Matplotlib, Pandas
- RNN_test.ipynb
	- Test file
- seq2seq_initial_wholeSequence_clst1.ipynb
	- seq2seq model trained with cluster1 data
- seq2seq_initial_wholeSequence_clst2.ipynb
	- seq2seq model trained with cluster2 data

## Reverse Geocoding
- Requirements: Installation of Python, Pandas, googlemaps, geopy
- Reverse Geocoding using Google Maps (for Github).ipynb
	- Obtains the actual address of the place with the latitude and longitude information by using the API provide by Google Maps
- Reverse Geocoding using geopy.ipynb
	- Obtains the actual address of the place with the latitude and longitude information by using Geopy

## Sequence Analysis
- Requirements: Installation of Python, Numpy, Matplotlib, Pandas
- seq_data_analysis.py
	- Analyzes sequences by various standards, such as often visited locations
- seq_time_analysis.py
	- Analyzes sequences by staying time
- seq_type_analysis.py
	- Analyzes sequences by location types

## Sequence Generation
- Requirements: Installation of Python, Pandas, Numpy, Haversine, Datetime
- Sequence Generation _ 통합.ipynb
	- It is a code that converts a sequence of GPS information into a single string sequence using data processed using stop detection.
	- After removing data that will not be used with threshold, each location is given a unique number and converted to string sequences including the moving state using raw data.
- sequence_trimming.ipynb
	- Extracts a trimmed sequence that removes duplication from the converted string sequence
- seq_into_int_vectors.ipynb
	- Converts a string sequence into an integer sequence by giving an integer to every place that is given a unique number 
