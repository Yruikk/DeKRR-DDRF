function data = dataGenerate(pms,data_type,filepath,normalize_type)
%DATAGENERATE
N = zeros(pms.J,1);
N_train = zeros(pms.J,1); N_test = zeros(pms.J,1);
X_train = cell(pms.J,1); X_test = cell(pms.J,1);
Y_train = cell(pms.J,1); Y_test = cell(pms.J,1);

switch data_type
    case 'real'
        raw_data = load(filepath);
        N_all = size(raw_data.Y,1);
        for j=1:pms.J
            N(j) = floor(N_all/pms.J);
            
            N_train(j) = floor(0.5*N(j));
            N_test(j) = N(j)-floor(0.5*N(j));
        end
        ind = randperm(N_all,sum(N));
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
% Each node mixes a small part of the usual data, so that the nodes have 
% different data distributions but can still learn new information from 
% their neighbors.
iid_ratio = 0.2; % 0.75 for Energy and 0.2 for others.
N_train_iid = zeros(pms.J,1);N_train_noniid = zeros(pms.J,1);
N_test_iid = zeros(pms.J,1);N_test_noniid = zeros(pms.J,1);
for j=1:pms.J
    N_train_iid(j) = floor((1-iid_ratio)*N_train(j));
    N_train_noniid(j) = N_train(j) - floor((1-iid_ratio)*N_train(j));
    N_test_iid(j) = floor((1-iid_ratio)*N_test(j));
    N_test_noniid(j) = N_test(j) - floor((1-iid_ratio)*N_test(j));
end
data_all = [X_all Y_all];
data_all_noniid = data_all(1:sum(N_train_noniid)+sum(N_test_noniid),:);
data_all_part = data_all(sum(N_train_noniid)+sum(N_test_noniid)+1:end,:);
[~,ind_sort] = sort(abs(data_all_part(:,end)),'descend');
data_all_part_sorted = data_all_part(ind_sort,:);
ind = 0;
ind_noniid = 0;
for j=1:pms.J
    data_j = data_all_part_sorted(ind+1:ind+N_train_iid(j)+N_test_iid(j),:);
    ind_shuffle = randperm(N_train_iid(j)+N_test_iid(j));
    data_j_shuffle = data_j(ind_shuffle,:);
    X_train{j} = data_j_shuffle(1:N_train_iid(j),1:end-1);
    X_train{j} = [X_train{j};data_all_noniid(ind_noniid+1:ind_noniid+N_train_noniid(j),1:end-1)];
    Y_train{j} = data_j_shuffle(1:N_train_iid(j),end);
    Y_train{j} = [Y_train{j};data_all_noniid(ind_noniid+1:ind_noniid+N_train_noniid(j),end)];
    
    X_test{j} = data_j_shuffle(N_train_iid(j)+1:N_train_iid(j)+N_test_iid(j),1:end-1);
    X_test{j} = [X_test{j};data_all_noniid(ind_noniid+N_train_noniid(j)+1:ind_noniid+N_train_noniid(j)+N_test_noniid(j),1:end-1)];
    Y_test{j} = data_j_shuffle(N_train_iid(j)+1:N_train_iid(j)+N_test_iid(j),end);
    Y_test{j} = [Y_test{j};data_all_noniid(ind_noniid+N_train_noniid(j)+1:ind_noniid+N_train_noniid(j)+N_test_noniid(j),end)];
    ind = ind+N_train_iid(j)+N_test_iid(j);
    ind_noniid = ind_noniid+N_train_noniid(j)+N_test_noniid(j);
end

% Ready for output.
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

