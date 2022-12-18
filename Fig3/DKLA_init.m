function [train_init,test_init,res_node_init,res_node] ...
    = DKLA_init(data,pms,result_type)
% Version: 2022.06.14
lambda_j = cell(pms.J,1);
for j=1:pms.J
    lambda_j{j} = pms.lambda*sum(data.N_train)/(data.N_train(j)*pms.J);
end
D_RF = pms.D_j;
sigma = pms.sigma;

X_train = data.X_train;
X_test = data.X_test;
Y_train = data.Y_train;
Y_test = data.Y_test;
N_train = data.N_train;
N_test = data.N_test;

theta = zeros(D_RF,pms.J);
pred = cell(pms.J,1);

res_node_init.train = zeros(pms.J,1);
res_node_init.test = zeros(pms.J,1);
res_node.train = zeros(pms.J,1);
res_node.test = zeros(pms.J,1);

z = cell(pms.J,1);
omega = cell(pms.J,1);
D = cell(pms.J,1);
b = cell(pms.J,1);
% Step 1. RFF.
for j=1:pms.J
    if j==1
        [z{j},omega{j},D{j},b{j}] = rff(X_train{j},D_RF,'Gaussian',sigma);
    elseif j~=1
        omega{j} = omega{1};
        D{j} = D{1};
        b{j} = b{1};
        z{j} = sqrt(2/D{j})*cos(omega{j}'*X_train{j}'+b{j});
    end
    theta(:,j) = ((z{j}*z{j}')+N_train(j)*lambda_j{j}*eye(D{j}))\(z{j}*Y_train{j}); 
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
    pred{j} = z{j}'*theta(:,j);
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
    pred{j} = z_test'*theta(:,j);
    pred_list = [pred_list; pred{j}];
    target_list = [target_list; Y_test{j}];
end
if strcmp(pms.result_type,'mse')
    test_init = mse(pred_list,target_list);
elseif strcmp(pms.result_type,'rse')
    test_init = rse(pred_list,target_list);
end

end

