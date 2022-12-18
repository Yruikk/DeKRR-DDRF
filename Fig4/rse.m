function value = rse(pred,target)
%RSE Relative Squared Error
%   p is defined as the prediction vector and a is defined as the actual vector.
%   RSE = \frac{\sum_i{p_i-a_i}^2}{\sum_i{mean(a)-a_i}^2}.
% if nargin < 1
%     error('RSE_NeedData');
% elseif nargin < 2
%     error('RSE_NeedMoreData');
% elseif size(pred,2) ~= 1 || size(target,2) ~= 1
%     error('RSE_ShouldBeColumnVector');
% elseif size(pred,1) ~= size(target,1)
%     error('RSE_DimMismatch');
% end

bar_target = mean(target);
if norm(target-bar_target,2) == 0
    error('RSE_BadTarget');
end
numerator = (norm(pred-target,2)^2);
denominator = (norm(target-bar_target,2)^2)/size(target,1);
value = (norm(pred-target,2)^2)/(norm(target-bar_target,2)^2);
end