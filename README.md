# CS6375ML-K-means

The package used and installation instructions
==============================================
1. numpy

	Open the terminal and input: pip3 install numpy
2. matplotlib

	Open the terminal and input: pip3 install matplotlib
3. pandas

	Open the terminal and input: pip3 install pandas
4. json

Development environment
=======================
Python 3.7.2

Algorithm and compile
=====================
Description: 
	
We wrote this program by python3 and you can input json file and initialseeds.txt. The program used k-means algorithm and jaccard distance to cluster tweets and output the clustering results and the overall SSE value.

How we designed the program:
1. Load two files and change them to list.
2. Calculate the distance between every data and every centroid and store the distance to the list. Then select the smallest distance and distribute the data to the closest centroid.
3. For every centroid, update the centroid by using these data belong to this centroid.
4. Calculate the SSE.
5. Repeat step 2 to step 4 until the change between two SSE is small enough.
