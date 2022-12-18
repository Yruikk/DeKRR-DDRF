function [theta,result_train,result_test,result_train_k,result_test_k,RF_new,res_node_init,res_node] ...
    = DKLA_RFF(data,pms,graph)
%DKLA_RFF
lambda_j = cell(pms.J,1);
for j=1:pms.J
    lambda_j{j} = pms.lambda*sum(data.N_train)/(data.N_train(j)*pms.J);
end
D_RF = pms.D_j;
sigma = pms.sigma;
rho = pms.rho_DKLA;
iter_max = pms.iter_max_DKLA;

nei = graph.nei;
card = graph.card;

X_train = data.X_train;
X_test = data.X_test;
Y_train = data.Y_train;
Y_test = data.Y_test;
N_train = data.N_train;
N_test = data.N_test;

N = sum(N_train);

theta = zeros(D_RF,pms.J);
theta_j = cell(pms.J,1);
theta_nei = cell(pms.J,1); 
pred = cell(pms.J,1);

result_train_k = zeros(iter_max+1,1);
result_test_k = zeros(iter_max+1,1);

res_node_init.train = zeros(pms.J,1);
res_node_init.test = zeros(pms.J,1);
res_node.train = zeros(pms.J,1);
res_node.test = zeros(pms.J,1);

z = cell(pms.J,1);
omega = cell(pms.J,1);
D = cell(pms.J,1);
b = cell(pms.J,1);

gamma = cell(pms.J,1);
inv_temp = cell(pms.J,1);
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
    gamma{j} = zeros(D{j},1);
    inv_temp{j} = inv(2/N*z{j}*z{j}'+(2*N_train(j)/N*lambda_j{j}+2*rho*card{j})*eye(D{j}));
end
for j=1:pms.J
    theta_j{j} = theta(:,j);
    theta_nei{j} = theta(:,nei{j});
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
    result_train_k(1,1) = mse(pred_list,target_list);
elseif strcmp(pms.result_type,'rse')
    result_train_k(1,1) = rse(pred_list,target_list);
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
    result_test_k(1,1) = mse(pred_list,target_list);
elseif strcmp(pms.result_type,'rse')
    result_test_k(1,1) = rse(pred_list,target_list);
end

% Step 2. Iteration for updating theta_j.
for iter=1:iter_max
    if mod(iter,200) == 0
        rho = rho*2;
        for j=1:pms.J
            inv_temp{j} = inv(2/N*z{j}*z{j}'+(2*N_train(j)/N*lambda_j{j}+2*rho*card{j})*eye(D{j}));
        end
    end
    % ADMM start.
    % Step 2.1. Comm 1st. Broadcast theta_j^k to node n and receive 
    %           theta_n^k from node j.(n \in \mathcal{N}_j)
    for j=1:pms.J
        theta_j{j} = theta(:,j);
        theta_nei{j} = theta(:,nei{j});
    end
    % Step 2.2. Update theta_j^k.
    for j=1:pms.J
        theta(:,j) = inv_temp{j}*(2/N*z{j}*Y_train{j} ...
            +rho*(card{j}*theta_j{j}+sum(theta_nei{j},2))-gamma{j});
    end
    % Step 2.3. Comm 2nd.
    for j=1:pms.J
        theta_j{j} = theta(:,j);
        theta_nei{j} = theta(:,nei{j});
    end
    % Step 2.4. Update gamma_j^k.
    for j=1:pms.J
        gamma{j} = gamma{j}+rho*sum(repmat(theta_j{j},1,card{j})-theta_nei{j},2);
    end
    % Record train loss.
    pred_list = [];
    target_list = [];
    for j=1:pms.J
        pred{j} = z{j}'*theta(:,j);
        
        res_node.train(j,1) = mse(pred{j},Y_train{j});
        
        pred_list = [pred_list; pred{j}];
        target_list = [target_list; Y_train{j}];
    end
    if strcmp(pms.result_type,'mse')
        result_train_k(iter+1,1) = mse(pred_list,target_list);
    elseif strcmp(pms.result_type,'rse')
        result_train_k(iter+1,1) = rse(pred_list,target_list);
    end
    % Record test loss.
    pred_list = [];
    target_list = [];
    for j=1:pms.J
        z_test = sqrt(2/D{j})*cos(omega{j}'*X_test{j}'+b{j});
        pred{j} = z_test'*theta(:,j);
        
        res_node.test(j,1) = mse(pred{j},Y_test{j});
        
        pred_list = [pred_list; pred{j}];
        target_list = [target_list; Y_test{j}];
    end
    if strcmp(pms.result_type,'mse')
        result_test_k(iter+1,1) = mse(pred_list,target_list);
    elseif strcmp(pms.result_type,'rse')
        result_test_k(iter+1,1) = rse(pred_list,target_list);
    end
end
% Record final train/test loss.
result_train = result_train_k(end,1);
result_test = result_test_k(end,1);
% Save RF for when needed.
RF_new.param_z = z;
RF_new.param_omega = omega;
RF_new.param_D = D;
RF_new.param_b = b;
end

