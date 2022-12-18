clear;close all;
%% Preparation of data sets.
load('Results/Air.mat'); % The data set in Fig3.
N_train_all = sum(data.N_train);
%% Parameters setting.
pms.J = 6; % Number of nodes.
pms.num_nei = 2; % Number of neighbors on each node.
pms.lambda = 1e-6;
pms.sigma = 0.25;
pms.rho_DKLA = 1e-4; % Small rho to make sure faster convergence.
pms.iter_max_DKLA = 1500;
pms.iter_max_Algo = 1500;
pms.result_type = 'rse'; % Relative squared error.
pms.rate = 20; % rate = D_0/D for picking features.

repeat_time = 10;
%% Graph structure generation.
graph = gengraph_fixNei(pms.J,pms.num_nei); % Only include one-hop neighbors.
%% Loop
DKLA_loop = [];
Algo_loop = [];
Algo_loop2 = [];
for iter_D = 80
    pms.D_cen = iter_D;
    pms.D_j = pms.D_cen;
    % Original DKLA.
    [~,DKLA_result] = DKLA_repeat(data,pms,graph,repeat_time);
    DKLA_loop = [DKLA_loop; DKLA_result];
    % Algo with different RFs (Equal D_j).
    [~,Algo_result] = Algo_repeat(data,pms,graph,repeat_time);
    Algo_loop = [Algo_loop; Algo_result];
    % Algo with different RFs (Different D_j).
    [~,Algo_result2] = Algo_repeat2(data,pms,graph,repeat_time);
    Algo_loop2 = [Algo_loop2; Algo_result2];
end
fprintf('Finished.\n');