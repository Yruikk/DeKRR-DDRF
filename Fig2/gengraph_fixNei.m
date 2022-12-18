function graph = gengraph_fixNei(n_node,n_nei)
%GENGRAPH_FIXNEI

adj_mat = zeros(n_node,n_node);
nei = cell(n_node,1);
card = cell(n_node,1);

temp = zeros(1,n_node);
for i=1:(n_nei/2)
    temp(1,i+1) = 1;
    temp(1,end-(i-1)) = 1;
end
for i=1:n_node
    if i == 1
        adj_mat(i,:) = temp;
    else
        temp = circshift(temp,1);
        adj_mat(i,:) = temp;
    end
end

for j=1:n_node
%     Nei{j} = adj_mat(j,:);
    nei{j} = find(adj_mat(j,:)==1);
    card{j} = length(nei{j});
end

graph.adj = adj_mat;
graph.nei = nei;
graph.card = card;
end

