function value = mse(vec_1,vec_2)
%MSE Mean Squared Error
%   p is defined as the prediction vector and a is defined as the actual vector.
%   MSE = \frac{\sum_{i=1}^n {(p_i-a_i)^2}}{n}.
if nargin < 1
    error('MSE_NeedData');
elseif nargin < 2
    error('MSE_NeedMoreData');
elseif size(vec_1,2) ~= 1 || size(vec_2,2) ~= 1
    error('MSE_ShouldBeColumnVector');
elseif size(vec_1,1) ~= size(vec_2,1)
    error('MSE_DimMismatch');
end

n = size(vec_1,1);
value = (1/n)*norm(vec_1-vec_2,2)^2;
end

