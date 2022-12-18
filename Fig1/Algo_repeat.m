function Algo_result = Algo_repeat(data,pms,graph,repeat_time)
Algo_train_list = zeros(repeat_time,1);
Algo_test_list = zeros(repeat_time,1);
Algo_node_train_init = zeros(pms.J,repeat_time); Algo_node_test_init = zeros(pms.J,repeat_time); 
Algo_node_train = zeros(pms.J,repeat_time); Algo_node_test = zeros(pms.J,repeat_time); 

c_list = [0.5 1 2 4 8];

weight = zeros(pms.J,1);
D_j = cell(pms.J,1);

% Assign D_j.
D_all = pms.J*pms.D_j;
for j=1:pms.J
%     weight(j,1) = sqrt(data.N_train(j));
    weight(j,1) = 1;
end
weight = weight./sum(weight);
for j=1:pms.J
    D_j{j} = round(weight(j,1)*D_all);
end
for j=1:pms.J
    weight(j) = 1/pms.J;
end

for i=1:repeat_time
    [~,~,~,~,RF_sub] = Algo_init(data,pms,D_j,'EERF');
    
    c_opt = pick_c(data,pms,graph,c_list,RF_sub.param_omega,RF_sub.param_D,RF_sub.param_b,weight);
    pms.c_Algo = c_opt;
    
    [~,result_train,result_test,~,~,result_node_init,result_node] ...
        = Algo(data,pms,graph,RF_sub,weight);
    Algo_train_list(i,1) = result_train;
    Algo_test_list(i,1) = result_test; 
    
    Algo_node_train_init(:,i) = result_node_init.train;
    Algo_node_test_init(:,i) = result_node_init.test;
    Algo_node_train(:,i) = result_node.train;
    Algo_node_test(:,i) = result_node.test;
end

result_Algo_train = mean(Algo_train_list);
result_Algo_test = mean(Algo_test_list);

Algo_result = [result_Algo_train result_Algo_test ...
    std(Algo_train_list) std(Algo_test_list)];
end