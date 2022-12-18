function [graph,Nei,nei,card] = gengraph(n_node,n_edge)
%GENGRAPH
% Randomly generated a connected graph with n_node nodes and n_edge edges.
len_edge_vec = (n_node^2-n_node)/2; 
if n_edge > len_edge_vec
    error('gengraph_BadEdge');
end

edge_vec = zeros(1,len_edge_vec);
index_edge = randperm(len_edge_vec,n_edge);
edge_vec(index_edge) = 1;

n = (-1+sqrt(1+8*len_edge_vec))/2 ;
graph = zeros(n);
index_graph = find(tril(ones(n)));
graph(index_graph) = edge_vec; %#ok<FNDSB>
graph = graph';
graph = [zeros(n_node-1,1) graph;0 zeros(1,n_node-1)];
graph = graph + graph';

for j=1:n_node
    Nei{j} = graph(j,:); %#ok<*AGROW>
    nei{j} = find(graph(j,:)==1);
    card{j} = length(nei{j});
end
end

