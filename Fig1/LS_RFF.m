function [z_new,omega_new,D_new,b_new] = LS_RFF(data,lambda,D,kernel,sigma)
%LS_RFF
%   Li Z, Ton J F, Oglic D, et al. Towards a unified analysis of random 
%   Fourier features[C]//International conference on machine learning. 
%   PMLR, 2019: 3905-3914.
%   Input: data = N x d
if nargin < 1 || isempty(data)
    error('lwrff_NeedData');
elseif ~ismatrix(data)
    error('lwrff_BadData');
elseif nargin < 2
    error('lwrff_NeedLambda');
end

if nargin < 3
    D = round(size(data,1)/2);
    kernel = 'Gaussian';
    sigma = 1;
elseif nargin < 4
    kernel = 'Gaussian';
    sigma = 1;
elseif nargin < 5
    sigma = 1;
end

[m,~] = size(data);
[z,omega,~,~] = rff(data,D,kernel,sigma);
z = z';
new_mat = D*(z'*z)*inv(z'*z+m*lambda*eye(D)); %#ok<MINV>
p = diag(new_mat);
l = trace(new_mat);
q = p/l;
D_new = ceil(l);
r = mnrnd(D_new,q);
index = find(r~=0);
index_new =[];
for i=1:length(index)
    index_new = [index_new repmat(index(i),1,r(index(i)))]; %#ok<*AGROW>
end
omega_new = omega(:,index_new);
b_new = 2*pi*rand(D_new,1);
z_new = (sqrt(2/D_new)*cos(omega_new'*data'+b_new));
% [m,~] = size(data);
% [z,omega,~,~] = rff(data,D,kernel,sigma);
% z = z';
% new_mat = z'*z*inv(z'*z+m*lambda*eye(D)); %#ok<MINV>
% p = diag(new_mat);
% l = trace(new_mat);
% q = p/l;
% weight = sqrt(1./q);
% [~,ind] = sort(q,'descend');
% omega_new = omega(:,ind(1:D_new));
% weight_new = repmat(weight(ind(1:D_new),:),1,m);
% b_new = 2*pi*rand(D_new,1);
% z_new = weight_new.*(sqrt(2/D_new)*cos(omega_new'*data'+b_new));
end

