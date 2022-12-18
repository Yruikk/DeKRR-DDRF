function value = rmse(vec_1,vec_2)
%RMSE Root Mean Squared Error
%   p is defined as the prediction vector and a is defined as the actual vector.
%   MSE = sqrt{\frac{\sum_{i=1}^n {(p_i-a_i)^2}}{n}}.
if nargin < 1
    error('RMSE_NeedData');
elseif nargin < 2
    error('RMSE_NeedMoreData');
elseif size(vec_1,2) ~= 1 || size(vec_2,2) ~= 1
    error('RMSE_ShouldBeColumnVector');
elseif size(vec_1,1) ~= size(vec_2,1)
    error('RMSE_DimMismatch');
end

n = size(vec_1,1);
value = sqrt(1/n*norm(vec_1-vec_2,2)^2);
end