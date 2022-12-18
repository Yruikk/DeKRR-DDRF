function [z_new,omega_new,D_new,b_new] = ELSS(data,lambda,M_0,kernel,sigma)
%ELSS
%   Shahrampour S, Kolouri S. On sampling random features from empirical 
%   leverage scores: Implementation and theoretical guarantees[J]. arXiv 
%   preprint arXiv:1903.08329, 2019.
%   Input: data = N x d
if nargin < 1 || isempty(data)
    error('ELSS_NeedData');
elseif ~ismatrix(data)
    error('ELSS_BadData');
elseif nargin < 2
    error('ELSS_NeedLambda');
end

if nargin < 3
    M_0 = round(size(data,1)/2);
    kernel = 'Gaussian';
    sigma = 1;
elseif nargin < 4
    kernel = 'Gaussian';
    sigma = 1;
elseif nargin < 5
    sigma = 1;
end

[m,~] = size(data);
[z,omega,~,~] = rff(data,M_0,kernel,sigma);
Q = (z*z')/(z*z'+lambda*eye(M_0));
if (condition)
    % future code here
end
end

