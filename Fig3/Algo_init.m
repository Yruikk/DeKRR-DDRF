function [train_init,test_init,res_node_init,res_node,RF] ...
    = Algo_init(data,pms,D_j,RF_type)
%ALGO_INIT
lambda_j = cell(pms.J,1);
for j=1:pms.J
    lambda_j{j} = pms.lambda*sum(data.N_train)/(data.N_train(j)*pms.J);
end
sigma = pms.sigma;

X_train = data.X_train;
X_test = data.X_test;
Y_train = data.Y_train;
Y_test = data.Y_test;
N_train = data.N_train;
N_test = data.N_test;

theta = cell(pms.J,1);
pred = cell(pms.J,1);

res_node_init.train = zeros(pms.J,1);
res_node_init.test = zeros(pms.J,1);
res_node.train = zeros(pms.J,1);
res_node.test = zeros(pms.J,1);

z = cell(pms.J,1);
omega = cell(pms.J,1);
D = cell(pms.J,1);
b = cell(pms.J,1);
% Step 1. RF (Use the plain RFF or data-dependent RF).
for j=1:pms.J
    switch RF_type
        case 'RFF'
            [z{j},omega{j},D{j},b{j}] = ...
                rff(X_train{j},D_j{j},'Gaussian',sigma);
        case 'EERF'
            [z{j},omega{j},D{j},b{j}] = ...
                EERF(X_train{j},Y_train{j},round(pms.rate*D_j{j}),D_j{j},'Gaussian',sigma);
    end
    theta{j} = ((z{j}*z{j}')+N_train(j)*lambda_j{j}*eye(D{j}))\(z{j}*Y_train{j}); 
end
% Record init local loss.
for j=1:pms.J
    theta_temp = ((z{j}*z{j}')+N_train(j)*lambda_j{j}*eye(D{j}))\(z{j}*Y_train{j}); 
    pred_temp = z{j}'*theta_temp;
    res_node_init.train(j,1) = mse(pred_temp,Y_train{j});
    z_test = sqrt(2/D{j})*cos(omega{j}'*X_test{j}'+b{j});
    pred_temp = z_test'*theta_temp;
    res_node_init.test(j,1) = mse(pred_temp,Y_test{j});
end
% Record train loss.
pred_list = [];
target_list = [];
for j=1:pms.J
    pred{j} = z{j}'*theta{j};
    pred_list = [pred_list; pred{j}];
    target_list = [target_list; Y_train{j}];
end
if strcmp(pms.result_type,'mse')
    train_init = mse(pred_list,target_list);
elseif strcmp(pms.result_type,'rse')
    train_init = rse(pred_list,target_list);
end
% Record test loss.
pred_list = [];
target_list = [];
for j=1:pms.J
    z_test = sqrt(2/D{j})*cos(omega{j}'*X_test{j}'+b{j});
    pred{j} = z_test'*theta{j};
    pred_list = [pred_list; pred{j}];
    target_list = [target_list; Y_test{j}];
end
if strcmp(pms.result_type,'mse')
    test_init = mse(pred_list,target_list);
elseif strcmp(pms.result_type,'rse')
    test_init = rse(pred_list,target_list);
end

RF.param_z = z;
RF.param_omega = omega;
RF.param_D = D;
RF.param_b = b;

end

