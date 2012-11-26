-- All rights reserved
-- Authors: Mostafa Mahdieh, Marjan GhazviniNejad

== Directory Structure
dtset : standard datasets
dtset_pose : datasets for the pose estimation application
java : java code for creating a k-geodesically-nearest-neighbours graph using an algorithm of
      running time O(n^2 k)
files in root: matlab code of the Isograph algorithm and experiments related to the algorithm

== MATLAB Code
Isograph.m: The main code that loads the standard datasets and runs the isograph algorithm on them
and compares the graph with graphs built by kNN

PoseEstimation.m: The main code that loads the pose estimation datasets and runs the isograph algorithm on them
and compares the graph with graphs built by kNN

IsographReweight.m: The Isograph reweighting algorithm

optimizeSigma.m: The Marginal Liklihood optimization algorithm


