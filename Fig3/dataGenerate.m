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
            N(j) = floor((2*j-1)/36*N_all);
            
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
ind = 0;
for j=1:pms.J
    X_train{j} = X_all(ind+1:ind+N_train(j),:);
    Y_train{j} = Y_all(ind+1:ind+N_train(j),:);
    X_test{j} = X_all(ind+N_train(j)+1:ind+N_train(j)+N_test(j),:);
    Y_test{j} = Y_all(ind+N_train(j)+1:ind+N_train(j)+N_test(j),:);
    ind = ind+N_train(j)+N_test(j);
end

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

