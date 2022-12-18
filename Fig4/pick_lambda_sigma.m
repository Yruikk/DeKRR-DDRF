function [lambda_opt,sigma_opt] = ...
    pick_lambda_sigma(data,pms,lambda_list,sigma_list)
%PICK_LAMBDA_SIGMA
%   Because this step is used to set the problem, it is not necessary to 
%   select the parameters by crossing validation on the sub-training set 
%   and validation set.
train_grid = zeros(length(lambda_list),length(sigma_list));
test_grid = zeros(length(lambda_list),length(sigma_list));
for i=1:length(lambda_list)
    pms.lambda = lambda_list(i);
    for ii=1:length(sigma_list)
        pms.sigma = sigma_list(ii);
        [result_train,result_test] = KRR_centralized(data,pms);
        train_grid(i,ii) = result_train;
        test_grid(i,ii) = result_test;
    end
end
[~, position_min] = min(test_grid(:)); 
[lambda_ind,sigma_ind] = ind2sub(size(test_grid),position_min);
lambda_opt = lambda_list(lambda_ind);
sigma_opt = sigma_list(sigma_ind);
end

