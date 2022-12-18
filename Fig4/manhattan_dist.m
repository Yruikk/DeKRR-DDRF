function D = manhattan_dist(X,Y)
% X and Y are m-by-n matrices
% m denotes the number of samples
% n denotes the dimension of samples
[mx,nx] = size(X);
[my,ny] = size(Y);
X = reshape(X,[mx,1,nx]);
Y = reshape(Y,[1,my,ny]);
D = abs(X-Y);
D = sum(D,3);
end

