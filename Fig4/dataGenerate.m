function data = dataGenerate(pms,data_type,filepath,normalize_type)
%DATAGENERATE
%   Version: 2022/06/18
%   1.产生各节点的原始data_j(X and Y)
%   2.中心式进行归一化
%   3.将归一化完的数据按照x：1-x的比例分成train和test，并分到各个节点上
N = zeros(pms.J,1);
N_train = zeros(pms.J,1); N_test = zeros(pms.J,1);
X = cell(pms.J,1);
raw_Y_train = cell(pms.J,1);raw_Y_test = cell(pms.J,1);
X_train = cell(pms.J,1); X_test = cell(pms.J,1);
Y_train = cell(pms.J,1); Y_test = cell(pms.J,1);

switch data_type
    case 'synthetic1'
        for j=1:pms.J
            N(j) = randi([2000,2100]);% 800,1200
            N_train(j) = floor(0.5*N(j));
            N_test(j) = N(j)-floor(0.5*N(j));
            X{j} = randn(N(j),pms.d);
            
            raw_Y_train{j} = zeros(N_train(j),1);
            raw_Y_test{j} = zeros(N_test(j),1);
            for i=1:N_train(j)
                raw_Y_train{j}(i,1) = sin(0.25*norm(X{j}(i,:),2)^2);
            end
            for i=1:N_test(j)
                raw_Y_test{j}(i,1) = sin(0.25*norm(X{j}(N_train(j)+i,:),2)^2);
            end
            SNR = 20;
            Y_train{j} = awgn(raw_Y_train{j},SNR,'measured');
            Y_test{j} = raw_Y_test{j};
        end
        X_all = [];
        Y_all = [];
        for j=1:pms.J
            X_all = [X_all;X{j}];
            Y_all = [Y_all;Y_train{j};Y_test{j}];
        end
    case 'synthetic2'
        for j=1:pms.J
            N(j) = randi([400,600]);
            N_train(j) = floor(0.7*N(j));
            N_test(j) = N(j)-floor(0.7*N(j));
            X{j} = rand(N(j),2);
            
            x = X{j}(1:N_train(j),1); y = X{j}(1:N_train(j),2);
            raw_Y_train{j} = 0.75*exp(-((9*x-2).^2+(9*y-2).^2)/4) ...
                +0.75*exp(-((9*x+1).^2)/49-(9*y+1)/10) ...
                +0.5*exp(-((9*x-7).^2+(9*y-3).^2)/4) ...
                -0.2*exp(-(9*x-4).^2-(9*y-7).^2);
            x = X{j}(N_train(j)+1:end,1); y = X{j}(N_train(j)+1:end,2);
            raw_Y_test{j} = 0.75*exp(-((9*x-2).^2+(9*y-2).^2)/4) ...
                +0.75*exp(-((9*x+1).^2)/49-(9*y+1)/10) ...
                +0.5*exp(-((9*x-7).^2+(9*y-3).^2)/4) ...
                -0.2*exp(-((9*x-4).^2)-((9*y-7).^2));
            SNR = 20;
            Y_train{j} = awgn(raw_Y_train{j},SNR,'measured');
            Y_test{j} = raw_Y_test{j};
        end
        X_all = [];
        Y_all = [];
        for j=1:pms.J
            X_all = [X_all;X{j}];
            Y_all = [Y_all;Y_train{j};Y_test{j}];
        end
    case 'real'
        raw_data = load(filepath);
        for j=1:pms.J
%                 N(j) = randi([800,1200]);
%                 N(j) = 1200;
            N(j) = 0.5*(200+400*(j-1));
                
            N_train(j) = floor(0.5*N(j));
            N_test(j) = N(j)-floor(0.5*N(j));
        end
        ind = randperm(size(raw_data.Y,1),sum(N));
        X_all = raw_data.X(ind,:);
        Y_all = raw_data.Y(ind,:);
    otherwise
end
% Normalization
if strcmp(normalize_type,'minmax')
    [X_all,~] = mapminmax(X_all',0,1);
    X_all = X_all';
    [Y_all,~] = mapminmax(Y_all',-1,1);
    Y_all = Y_all';
elseif strcmp(normalize_type,'mapstd')
    [X_all,~] = mapstd(X_all',0,1);
    X_all = X_all';
    [Y_all,~] = mapminmax(Y_all',-1,1);
    Y_all = Y_all';
end
% Scatter
if contains(data_type,'synthetic') %|| contains(filepath,'CPU') || contains(filepath,'Twitter')
    data_all = [X_all Y_all];
    [~,ind_sort] = sort(abs(data_all(:,end)),'descend');
    data_all_sorted = data_all(ind_sort,:);
    ind = 0;
    for j=1:pms.J
        data_j = data_all_sorted(ind+1:ind+N_train(j)+N_test(j),:);
        ind_shuffle = randperm(N_train(j)+N_test(j));
        data_j_shuffle = data_j(ind_shuffle,:);
        X_train{j} = data_j_shuffle(1:N_train(j),1:end-1);
        Y_train{j} = data_j_shuffle(1:N_train(j),end);
        X_test{j} = data_j_shuffle(N_train(j)+1:N_train(j)+N_test(j),1:end-1);
        Y_test{j} = data_j_shuffle(N_train(j)+1:N_train(j)+N_test(j),end);
        ind = ind+N_train(j)+N_test(j);
    end
else
    ind = 0;
    for j=1:pms.J
        X_train{j} = X_all(ind+1:ind+N_train(j),:);
        Y_train{j} = Y_all(ind+1:ind+N_train(j),:);
        X_test{j} = X_all(ind+N_train(j)+1:ind+N_train(j)+N_test(j),:);
        Y_test{j} = Y_all(ind+N_train(j)+1:ind+N_train(j)+N_test(j),:);
        ind = ind+N_train(j)+N_test(j);
    end
end

% Scatter_v1 Y
% if contains(data_type,'synthetic') || contains(filepath,'CPU') || contains(filepath,'Twitter')
%     iid_ratio = 0.2;
%     N_train_iid = zeros(pms.J,1);N_train_noniid = zeros(pms.J,1); 
%     N_test_iid = zeros(pms.J,1);N_test_noniid = zeros(pms.J,1);
%     for j=1:pms.J
%         N_train_iid(j) = floor((1-iid_ratio)*N_train(j));
%         N_train_noniid(j) = N_train(j) - floor((1-iid_ratio)*N_train(j));
%         N_test_iid(j) = floor((1-iid_ratio)*N_test(j));
%         N_test_noniid(j) = N_test(j) - floor((1-iid_ratio)*N_test(j));
%     end
%     data_all = [X_all Y_all];
%     data_all_noniid = data_all(1:sum(N_train_noniid)+sum(N_test_noniid),:);
%     data_all_part = data_all(sum(N_train_noniid)+sum(N_test_noniid)+1:end,:);
%     [~,ind_sort] = sort(abs(data_all_part(:,end)),'descend');
%     data_all_part_sorted = data_all_part(ind_sort,:);
%     ind = 0;
%     ind_noniid = 0;
%     for j=1:pms.J
%         data_j = data_all_part_sorted(ind+1:ind+N_train_iid(j)+N_test_iid(j),:);
%         ind_shuffle = randperm(N_train_iid(j)+N_test_iid(j));
%         data_j_shuffle = data_j(ind_shuffle,:);
%         X_train{j} = data_j_shuffle(1:N_train_iid(j),1:end-1);
%         X_train{j} = [X_train{j};data_all_noniid(ind_noniid+1:ind_noniid+N_train_noniid(j),1:end-1)];
%         Y_train{j} = data_j_shuffle(1:N_train_iid(j),end);
%         Y_train{j} = [Y_train{j};data_all_noniid(ind_noniid+1:ind_noniid+N_train_noniid(j),end)];
%         
%         X_test{j} = data_j_shuffle(N_train_iid(j)+1:N_train_iid(j)+N_test_iid(j),1:end-1);
%         X_test{j} = [X_test{j};data_all_noniid(ind_noniid+N_train_noniid(j)+1:ind_noniid+N_train_noniid(j)+N_test_noniid(j),1:end-1)];
%         Y_test{j} = data_j_shuffle(N_train_iid(j)+1:N_train_iid(j)+N_test_iid(j),end);
%         Y_test{j} = [Y_test{j};data_all_noniid(ind_noniid+N_train_noniid(j)+1:ind_noniid+N_train_noniid(j)+N_test_noniid(j),end)];
%         ind = ind+N_train_iid(j)+N_test_iid(j);
%         ind_noniid = ind_noniid+N_train_noniid(j)+N_test_noniid(j);
%     end
% else
%     ind = 0;
%     for j=1:pms.J
%         X_train{j} = X_all(ind+1:ind+N_train(j),:);
%         Y_train{j} = Y_all(ind+1:ind+N_train(j),:);
%         X_test{j} = X_all(ind+N_train(j)+1:ind+N_train(j)+N_test(j),:);
%         Y_test{j} = Y_all(ind+N_train(j)+1:ind+N_train(j)+N_test(j),:);
%         ind = ind+N_train(j)+N_test(j);
%     end
% end

% Scatter_v2 norm X
% if contains(data_type,'synthetic') || contains(filepath,'CPU') || contains(filepath,'Twitter')
%     iid_ratio = 0.2;
%     N_train_iid = zeros(pms.J,1);N_train_noniid = zeros(pms.J,1); 
%     N_test_iid = zeros(pms.J,1);N_test_noniid = zeros(pms.J,1);
%     for j=1:pms.J
%         N_train_iid(j) = floor((1-iid_ratio)*N_train(j));
%         N_train_noniid(j) = N_train(j) - floor((1-iid_ratio)*N_train(j));
%         N_test_iid(j) = floor((1-iid_ratio)*N_test(j));
%         N_test_noniid(j) = N_test(j) - floor((1-iid_ratio)*N_test(j));
%     end
%     data_all = [X_all Y_all];
%     data_all_noniid = data_all(1:sum(N_train_noniid)+sum(N_test_noniid),:);
%     data_all_part = data_all(sum(N_train_noniid)+sum(N_test_noniid)+1:end,:);
%     sorted_by_what = sqrt(sum(abs(data_all_part(:,1:end-1)).^2,2));
%     [~,ind_sort] = sort(sorted_by_what,'descend');
%     data_all_part_sorted = data_all_part(ind_sort,:);
%     ind = 0;
%     ind_noniid = 0;
%     for j=1:pms.J
%         data_j = data_all_part_sorted(ind+1:ind+N_train_iid(j)+N_test_iid(j),:);
%         ind_shuffle = randperm(N_train_iid(j)+N_test_iid(j));
%         data_j_shuffle = data_j(ind_shuffle,:);
%         X_train{j} = data_j_shuffle(1:N_train_iid(j),1:end-1);
%         X_train{j} = [X_train{j};data_all_noniid(ind_noniid+1:ind_noniid+N_train_noniid(j),1:end-1)];
%         Y_train{j} = data_j_shuffle(1:N_train_iid(j),end);
%         Y_train{j} = [Y_train{j};data_all_noniid(ind_noniid+1:ind_noniid+N_train_noniid(j),end)];
%         
%         X_test{j} = data_j_shuffle(N_train_iid(j)+1:N_train_iid(j)+N_test_iid(j),1:end-1);
%         X_test{j} = [X_test{j};data_all_noniid(ind_noniid+N_train_noniid(j)+1:ind_noniid+N_train_noniid(j)+N_test_noniid(j),1:end-1)];
%         Y_test{j} = data_j_shuffle(N_train_iid(j)+1:N_train_iid(j)+N_test_iid(j),end);
%         Y_test{j} = [Y_test{j};data_all_noniid(ind_noniid+N_train_noniid(j)+1:ind_noniid+N_train_noniid(j)+N_test_noniid(j),end)];
%         ind = ind+N_train_iid(j)+N_test_iid(j);
%         ind_noniid = ind_noniid+N_train_noniid(j)+N_test_noniid(j);
%     end
% else
%     ind = 0;
%     for j=1:pms.J
%         X_train{j} = X_all(ind+1:ind+N_train(j),:);
%         Y_train{j} = Y_all(ind+1:ind+N_train(j),:);
%         X_test{j} = X_all(ind+N_train(j)+1:ind+N_train(j)+N_test(j),:);
%         Y_test{j} = Y_all(ind+N_train(j)+1:ind+N_train(j)+N_test(j),:);
%         ind = ind+N_train(j)+N_test(j);
%     end
% end

% Scatter_Fed non-IID Y.
% data_all = [X_all Y_all];
% data_train_all =  data_all(1:sum(N_train),:);
% data_test_all = data_all(sum(N_train)+1:sum(N_train)+sum(N_test),:);
% 
% [~,ind_sort] = sort(abs(data_train_all(:,end)),'descend');
% data_train_sorted = data_train_all(ind_sort,:);
% ind_tr = 0;
% ind_te = 0;
% for j=1:pms.J
%     data_j = data_train_sorted(ind_tr+1:ind_tr+N_train(j),:);
%     ind_shuffle = randperm(N_train(j));
%     data_j_shuffle = data_j(ind_shuffle,:);
%     X_train{j} = data_j_shuffle(:,1:end-1);
%     Y_train{j} = data_j_shuffle(:,end);
%     ind_tr = ind_tr+N_train(j);
%     
%     X_test{j} = data_test_all(ind_te+1:ind_te+N_test(j),1:end-1);
%     Y_test{j} = data_test_all(ind_te+1:ind_te+N_test(j),end);
%     ind_te = ind_te+N_train(j);
% end

% Scatter_Fed non-IID norm X.
% data_all = [X_all Y_all];
% data_train_all =  data_all(1:sum(N_train),:);
% data_test_all = data_all(sum(N_train)+1:sum(N_train)+sum(N_test),:);
% 
% sort_by_what = sqrt(sum(abs(data_train_all(:,1:end-1)).^2,2));
% [~,ind_sort] = sort(sort_by_what,'descend');
% data_train_sorted = data_train_all(ind_sort,:);
% ind_tr = 0;
% ind_te = 0;
% for j=1:pms.J
%     data_j = data_train_sorted(ind_tr+1:ind_tr+N_train(j),:);
%     ind_shuffle = randperm(N_train(j));
%     data_j_shuffle = data_j(ind_shuffle,:);
%     X_train{j} = data_j_shuffle(:,1:end-1);
%     Y_train{j} = data_j_shuffle(:,end);
%     ind_tr = ind_tr+N_train(j);
%     
%     X_test{j} = data_test_all(ind_te+1:ind_te+N_test(j),1:end-1);
%     Y_test{j} = data_test_all(ind_te+1:ind_te+N_test(j),end);
%     ind_te = ind_te+N_train(j);
% end


for j=1:pms.J
    data.X_train{j} = X_train{j};
    data.Y_train{j} = Y_train{j};
    data.X_test{j} = X_test{j};
    data.Y_test{j} = Y_test{j};
end
data.N_train = N_train;
data.N_test = N_test;
data.data_type = data_type;
end

