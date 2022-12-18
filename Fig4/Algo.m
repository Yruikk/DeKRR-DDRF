function [theta,result_train,result_test,result_train_k,result_test_k,res_node_init,res_node] ...
    = Algo(data,pms,graph,RF,weight)
% According to Decentralized Online Kernel Learning, only use gradient
% descent to solve the problem.
lambda_j = cell(pms.J,1);
for j=1:pms.J
    lambda_j{j} = pms.lambda*sum(data.N_train)/(data.N_train(j)*pms.J);
end

c_j = cell(pms.J,1);
for j=1:pms.J
    c_j{j} = weight(j)*pms.J*pms.c_Algo/graph.card{j};
end

iter_max = pms.iter_max_Algo;

nei = graph.nei;
card = graph.card;

X_train = data.X_train;
X_test = data.X_test;
Y_train = data.Y_train;
Y_test = data.Y_test;
N_train = data.N_train;
N_test = data.N_test;

theta = cell(pms.J,1);
pred = cell(pms.J,1);

result_train_k = zeros(iter_max+1,1);
result_test_k = zeros(iter_max+1,1);

res_node_init.train = zeros(pms.J,1);
res_node_init.test = zeros(pms.J,1);
res_node.train = zeros(pms.J,1);
res_node.test = zeros(pms.J,1);

z = RF.param_z; %cell(pms.J,1);
omega = RF.param_omega;
D = RF.param_D;
b = RF.param_b;

theta_temp = cell(pms.J,1);
inv_temp = cell(pms.J,1);
right_temp = cell(pms.J,1);
% Step 1. RFF.
for j=1:pms.J
    theta{j} = zeros(D{j},1);
end
% Note that j denotes agent itself, and i denotes agent j's one-hop neighbors.
% z{j} = Z_j(X_j), z_ij = Z_i(X_j), z_ji = Z_j(X_i), and z_ii = Z_i(X_i).
z_ij = cell(pms.J,1);
z_ji = cell(pms.J,1);
z_ii = cell(pms.J,1);
for j=1:pms.J
    z_ij{j} = cell(card{j},1);
    z_ji{j} = cell(card{j},1);
    z_ii{j} = cell(card{j},1);
    for i=1:card{j}
        z_ij{j}{i} = sqrt(2/D{nei{j}(i)})*cos(omega{nei{j}(i)}'*X_train{j}'+b{nei{j}(i)});
        z_ji{j}{i} = sqrt(2/D{j})*cos(omega{j}'*X_train{nei{j}(i)}'+b{j});
        z_ii{j}{i} = sqrt(2/D{nei{j}(i)})*cos(omega{nei{j}(i)}'*X_train{nei{j}(i)}'+b{nei{j}(i)});
    end
end
for j=1:pms.J
    inv_temp{j} = (1+c_j{j}*card{j})*z{j}*z{j}'+N_train(j)*lambda_j{j}*eye(D{j});
    right_temp{j} = cell(card{j},1);
    for i=1:card{j}
        inv_temp{j} = inv_temp{j}+c_j{nei{j}(i)}*z_ji{j}{i}*z_ji{j}{i}';
        right_temp{j}{i} = (c_j{j}*z{j}*z_ij{j}{i}'+c_j{nei{j}(i)}*z_ji{j}{i}*z_ii{j}{i}');
    end
    inv_temp{j} = inv_temp{j}+5*c_j{j}*z{j}*z{j}';
    inv_temp{j} = inv(inv_temp{j});
end
% Record init local loss.
for j=1:pms.J
    theta_tmp = ((z{j}*z{j}')+N_train(j)*lambda_j{j}*eye(D{j}))\(z{j}*Y_train{j}); 
    pred_temp = z{j}'*theta_tmp;
    res_node_init.train(j,1) = mse(pred_temp,Y_train{j});
    z_test = sqrt(2/D{j})*cos(omega{j}'*X_test{j}'+b{j});
    pred_temp = z_test'*theta_tmp;
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
    result_train_k(1,1) = mse(pred_list,target_list);
elseif strcmp(pms.result_type,'rse')
    result_train_k(1,1) = rse(pred_list,target_list);
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
    result_test_k(1,1) = mse(pred_list,target_list);
elseif strcmp(pms.result_type,'rse')
    result_test_k(1,1) = rse(pred_list,target_list);
end
% Step 2. Iteration for updating theta_j.
for iter=1:iter_max
    % Update theta_j.
    for j=1:pms.J
        theta_temp{j} = z{j}*Y_train{j};
        for i=1:card{j}
            theta_temp{j} = theta_temp{j}+right_temp{j}{i}*theta{nei{j}(i)};
        end
        theta_temp{j} = theta_temp{j}+5*c_j{j}*z{j}*z{j}'*theta{j};
    end
    for j=1:pms.J
        theta{j} = inv_temp{j}*theta_temp{j};
    end
    % Record train loss.
    pred_list = [];
    target_list = [];
    for j=1:pms.J
        pred{j} = z{j}'*theta{j};
        
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
        pred{j} = z_test'*theta{j};
        
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
end

