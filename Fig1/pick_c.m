function c_opt = pick_c(data,pms,graph,c_list,omega,D,b,weight)
%PICK_C
lambda_j = cell(pms.J,1);
for j=1:pms.J
    lambda_j{j} = pms.lambda*sum(data.N_train)/(data.N_train(j)*pms.J);
end

RF.param_omega = omega;
RF.param_D = D;
RF.param_b = b;

fold = 5;
cvIndices = cell(pms.J,1);
for j=1:pms.J
    cvIndices{j} = [];
    temp_N = [repmat(round(1/fold*data.N_train(j)),fold-1,1); ...
        data.N_train(j)-(fold-1)*round(1/fold*data.N_train(j))];
    for iter=1:fold
        cvIndices{j} = [cvIndices{j}; repmat(iter,temp_N(iter),1)];
    end
end
train_grid = zeros(length(c_list),1);
val_grid = zeros(length(c_list),1);
for i=1:length(c_list)
    pms.c_Algo = c_list(i);
    train_list = zeros(fold,1);
    val_list = zeros(fold,1);
    for iter=1:fold
        data_cv.N_train = zeros(pms.J,1);
        data_cv.N_test = zeros(pms.J,1);
        for j=1:pms.J
            testInd = (cvIndices{j}==iter);
            trainInd = ~testInd;
            data_cv.X_train{j} = data.X_train{j}(trainInd,:);
            data_cv.Y_train{j} = data.Y_train{j}(trainInd,:);
            data_cv.X_test{j} = data.X_train{j}(testInd,:);
            data_cv.Y_test{j} = data.Y_train{j}(testInd,:);
            data_cv.N_train(j) = sum(trainInd);
            data_cv.N_test(j) = sum(testInd);
        end
        [~,result_train,result_val,~,~] = Algo_cv(data_cv,pms,graph,RF,weight);            
        train_list(iter) = result_train;
        val_list(iter) = result_val;
    end
    train_grid(i,1) = mean(train_list);
    val_grid(i,1) = mean(val_list);
end
[~, position_min] = min(val_grid(:));
[c_ind,~] = ind2sub(size(val_grid),position_min);
c_opt = c_list(c_ind);
end

