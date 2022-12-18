function DKLA_result = DKLA_repeat(data,pms,graph,repeat_time) 
%DKLA_REPEAT
DKLA_train_list = zeros(repeat_time,1); DKLA_test_list = zeros(repeat_time,1);
DKLA_node_train_init = zeros(pms.J,repeat_time); DKLA_node_test_init = zeros(pms.J,repeat_time); 
DKLA_node_train = zeros(pms.J,repeat_time); DKLA_node_test = zeros(pms.J,repeat_time); 
for i=1:repeat_time
    [~,result_train,result_test,~,~,~,result_node_init,result_node] ...
        = DKLA_RFF(data,pms,graph);
    DKLA_train_list(i,1) = result_train; DKLA_test_list(i,1) = result_test;
    
    DKLA_node_train_init(:,i) = result_node_init.train;
    DKLA_node_test_init(:,i) = result_node_init.test;
    DKLA_node_train(:,i) = result_node.train;
    DKLA_node_test(:,i) = result_node.test;
end
result_DKLA_train = mean(DKLA_train_list);
result_DKLA_test = mean(DKLA_test_list);

DKLA_result = [result_DKLA_train result_DKLA_test ... 
               std(DKLA_train_list) std(DKLA_test_list)];
end