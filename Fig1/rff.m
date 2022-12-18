function [z,omega,D,b] = rff(data,D,kernel,sigma)
%RFF
%   Rahimi A, Recht B. Random features for large-scale kernel machines[J].
%   Advances in neural information processing systems, 2007, 20.
%   Input: data is a N-by-d matrix.
%   Output: z is a D-by-N matrix, and omega is a d-by-D matrix.
if nargin < 1 || isempty(data)
    error('rff_NeedData');
elseif ~ismatrix(data)
    error('rff_BadData');
% elseif size(data,1) < D
%     warning('rff_BadD');
end

if nargin < 2
    D = round(size(data,1)/2);
    kernel = 'Gaussian';
    sigma = 1;
elseif nargin < 3
    kernel = 'Gaussian';
    sigma = 1;
elseif nargin < 4
    sigma = 1;
end

[~,n]=size(data);
if strcmp(kernel,'Gaussian')
    omega = (1/sigma)*randn(n,D);
elseif strcmp(kernel,'Laplacian')
    gamma = 1/sigma;
    omega = cauchyrnd(n,D,0,gamma);
elseif strcmp(kernel,'Cauchy')
    b_cauchy = 1/sigma;
    omega = laprnd(n,D,0,b_cauchy);
end

% z = sqrt(1/D)*[cos(omega'*data');sin(omega'*data')]; % Another way.
b = 2*pi*rand(D,1);
z = sqrt(2/D)*cos(omega'*data'+b);
end