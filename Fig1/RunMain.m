clear;close all;
load('Results/Houses.mat');
%% Parameters setting.
pms.J = 10; % Number of nodes.
pms.num_nei = 4; % Number of neighbors on each node.
% pms.d = 12;
pms.rho_DKLA = 1e-4; 
pms.iter_max_DKLA = 1000;
pms.iter_max_Algo = 1000;
pms.result_type = 'rse'; % Relative squared error.
pms.rate = 20; % rate = D_0/D for picking features.

repeat_time = 10;
%% Preparation of data sets.
% filepath = 'realdata/Houses.mat'; % d_House=8; d_AirQuality=13; d_Energy=27; d_Toms=96.
% data = dataGenerate(pms,'real',filepath,'minmax');
% 
% N_train_all = sum(data.N_train);
%% Graph structure generation.
graph = gengraph_fixNei(pms.J,pms.num_nei); % Only include one-hop neighbors.
%% Cross validation for lambda and sigma.
% lambda_list = 10.^[-8 -7 -6 -5];
% sigma_list = [0.125 0.25 0.5 1 2 4];
% [lambda_opt,sigma_opt] = pick_lambda_sigma(data,pms,lambda_list,sigma_list);
% pms.lambda = lambda_opt;
% pms.sigma = sigma_opt; 
%% Repeated experiment.
DKLA_loop = [];
Algo_loop = [];
i = 1;
for iter_D = 20:20:120
    pms.D_cen = iter_D;
    pms.D_j = pms.D_cen;
    
    % Result: TrainRSE TestRSE TrainStd TestStd 
    % DKLA.
    DKLA_result = DKLA_repeat(data,pms,graph,repeat_time);
    DKLA_loop = [DKLA_loop; DKLA_result];
    
    % DeKRR-DDRF.
    Algo_result = Algo_repeat(data,pms,graph,repeat_time);
    Algo_loop = [Algo_loop; Algo_result];
    i = i+1;
end
fprintf('Finished.\n');