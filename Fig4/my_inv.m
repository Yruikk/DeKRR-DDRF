function inv_mat = my_inv(mat)
%MY_INV 
%   Directly inversing kernel matrix may lead to undesirable numerical
%   problem. (Note that the eigenvalues ​​of kernel matrix are non-negetive, 
%   cause kernel matrix is ​​positive semi-definite)
thres = 1e-6;
[V,D] = eig(mat);
D_flat = diag(D);
% index = find(abs(D_flat)>thres);
index = find(D_flat>thres);
D_flat(index) = 1./D_flat(index);
inv_D = diag(D_flat);
inv_mat = V*inv_D*V';
end

