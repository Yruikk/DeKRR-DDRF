function result_init = InitResult(data,pms,repeat_time)
%INITRESULT
% DKLA
DKLA_train_init = zeros(repeat_time,1); DKLA_test_init = zeros(repeat_time,1);
DKLA_node_train_init = zeros(pms.J,repeat_time); DKLA_node_test_init = zeros(pms.J,repeat_time);
for i=1:repeat_time
    [train_init,test_init,result_node_init,result_node] ...
        = DKLA_init(data,pms);
    DKLA_train_init(i,1) = train_init; DKLA_test_init(i,1) = test_init;

    DKLA_node_train_init(:,i) = result_node_init.train;
    DKLA_node_test_init(:,i) = result_node_init.test;
end
result_DKLA_init = [mean(DKLA_train_init) mean(DKLA_test_init)];

result_DKLA_node_train_init = mean(DKLA_node_train_init,2);
result_DKLA_node_test_init = mean(DKLA_node_test_init,2);

DKLA_result_node_init = [result_DKLA_node_train_init result_DKLA_node_test_init  std(DKLA_node_train_init,0,2) std(DKLA_node_test_init,0,2)];
% Algo
Algo_train_init = zeros(repeat_time,1); Algo_test_init = zeros(repeat_time,1);
Algo_node_train_init = zeros(pms.J,repeat_time); Algo_node_test_init = zeros(pms.J,repeat_time);

D_all = pms.J*pms.D_j;
weight = zeros(pms.J,1);
for j=1:pms.J
    weight(j,1) = sqrt(data.N_train(j));
end
weight = weight./sum(weight);
D_j = cell(pms.J,1);
for j=1:pms.J
    D_j{j} = round(weight(j,1)*D_all);
end

for i=1:repeat_time
    [train_init,test_init,result_node_init,result_node,~] ...
        = Algo_init(data,pms,D_j,'EERF');
    Algo_train_init(i,1) = train_init; Algo_test_init(i,1) = test_init;

    Algo_node_train_init(:,i) = result_node_init.train;
    Algo_node_test_init(:,i) = result_node_init.test;
end
result_Algo_init = [mean(Algo_train_init) mean(Algo_test_init)];

result_Algo_node_train_init = mean(Algo_node_train_init,2);
result_Algo_node_test_init = mean(Algo_node_test_init,2);

Algo_result_node_init = [result_Algo_node_train_init result_Algo_node_test_init  std(Algo_node_train_init,0,2) std(Algo_node_test_init,0,2)];

result_init = [result_DKLA_init result_Algo_init];
end