distype = 1; % 1 is for euclidian distance and 2 is for tf-id distance
pc = 35; % number of pca dimensions

out = fopen('pose_estimation_1.txt', 'w');
N = [];
for setid = [9, 10, 33, 40, 63, 102, 125, 1401, 1417, 1424]
	load(sprintf('dtset_pose/%d.mat', setid));
	Dt = X;
	labels = Y;
	N = [N, size(Dt,1)];  % number of instances
end
mean(N);
