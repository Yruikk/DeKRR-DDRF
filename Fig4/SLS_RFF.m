function [z_new,omega_new,s,b_new] = SLS_RFF(data,y,l,s,kernel,sigma)
%SLS_RFF
%   Liu F, Huang X, Chen Y, et al. Random fourier features via fast 
%   surrogate leverage weighted sampling[C]//Proceedings of the AAAI 
%   Conference on Artificial Intelligence. 2020, 34(04): 4844-4851.
%   Input: data = N x d
if nargin < 1 || isempty(data)
    error('SLS-RFF_NeedData');
elseif ~ismatrix(data)
    error('SLS-RFF_BadData');
elseif nargin < 2
    error('SLS-RFF_NeedLambda');
end

% if nargin < 3
%     D = round(size(data,1)/2);
%     kernel = 'Gaussian';
%     sigma = 1;
% elseif nargin < 4
%     kernel = 'Gaussian';
%     sigma = 1;
% elseif nargin < 5
%     sigma = 1;
% end

[m,~] = size(data);
[z,omega,~,~] = rff(data,l,kernel,sigma);
z = z';
P_mat = (y'*z).^2;
L = sum(P_mat);
P_mat = P_mat/L;
r = mnrnd(s,P_mat);
index = find(r~=0);
index_new =[];
for i=1:length(index)
    index_new = [index_new repmat(index(i),1,r(index(i)))]; %#ok<*AGROW>
end
omega_new = omega(:,index_new);
b_new = 2*pi*rand(s,1);
z_new = (sqrt(2/s)*cos(omega_new'*data'+b_new));
% [m,~] = size(data);
% [z,omega,~,~] = rff(data,l,kernel,sigma);
% z = z';
% P_mat = (y'*z).^2;
% L = sum(P_mat);
% P_mat = P_mat/L;
% P_mat = P_mat';
% [~,ind] = sort(P_mat,'descend');
% weight = sqrt(1./P_mat(ind(1:s),:));
% omega_new = omega(:,index_new);
% b_new = 2*pi*rand(s,1);
% z_new = (sqrt(2/s)*cos(omega_new'*data'+b_new));
end

