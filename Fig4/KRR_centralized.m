function [result_train,result_test] = KRR_centralized(data,pms)
%KRR_CENTRALIZED
%   min 1/N*||alpha'*K-Y||_2^2+lambda*alpha'*K*alpha
%   Solution: alpha^\ast = [K+N*lambda*I]^{-1}*Y. 
lambda = pms.lambda;
sigma = pms.sigma;

X_train_cen = [];
Y_train_cen = [];
for j=1:pms.J
    X_train_cen = [X_train_cen;data.X_train{j}];
    Y_train_cen = [Y_train_cen;data.Y_train{j}];
end
N_train_all = sum(data.N_train);  
K_train_cen = kermat(X_train_cen,X_train_cen,'Gaussian',sigma);
% Check rank of gram matrix.
% rank_K = rank(K_train_cen);
alpha = (K_train_cen+N_train_all*lambda*eye(size(K_train_cen)))\Y_train_cen;
pred = K_train_cen*alpha;
if strcmp(pms.result_type,'mse')
    result_train = mse(pred,Y_train_cen);
elseif strcmp(pms.result_type,'rse')
    result_train = rse(pred,Y_train_cen);
end

X_test_cen = [];
Y_test_cen = [];
for j=1:pms.J
    X_test_cen = [X_test_cen;data.X_test{j}];
    Y_test_cen = [Y_test_cen;data.Y_test{j}];
end
% N_test_all = sum(data.N_test);
K_temp = kermat(X_test_cen,X_train_cen,'Gaussian',sigma);
pred = K_temp*alpha;
if strcmp(pms.result_type,'mse')
    result_test = mse(pred,Y_test_cen);
elseif strcmp(pms.result_type,'rse')
    result_test = rse(pred,Y_test_cen);
end
% fprintf('KRR_cen train=%.4e, test=%.4e,\n',result_train,result_test);
end

