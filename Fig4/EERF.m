function [z_new,omega_new,M,b_new] = EERF(data,y,M_0,M,kernel,sigma)
%EERF 
%   Shahrampour S, Beirami A, Tarokh V. On data-dependent random features 
%   for improved generalization in supervised learning[C]//Proceedings of 
%   the AAAI Conference on Artificial Intelligence. 2018, 32(1).
%   Input: data is a N-by-d matrix, and y is a N-by-1 vector.
if nargin < 1 || isempty(data)
    error('EERF_NeedData');
elseif ~ismatrix(data)
    error('EERF_BadData');
elseif nargin < 2
    error('EERF_NeedLabel');
elseif M > M_0
    error('EERF_BadFeatureNum');
end

N = size(y,1);
[z,omega,~,~] = rff(data,M_0,kernel,sigma);
S_emp = abs(1/N*z*y);
[~,ind] = sort(S_emp,'descend');
omega_new = omega(:,ind(1:M));
b_new = 2*pi*rand(M,1);
z_new = sqrt(2/M)*cos(omega_new'*data'+b_new);
end

