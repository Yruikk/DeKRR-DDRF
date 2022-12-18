function [z_new,omega_new,D_out,b_new] = EERFGivenRF(data,y,z,omega,D_in,b)
%EERFGIVENRF 

N = size(y,1);

S_emp = abs(1/N*z*y);
[~,ind] = sort(S_emp,'descend');

D_out = D_in;
omega_new = omega(:,ind(1:D_out));
b_new = b(ind(1:D_out));
z_new = sqrt(2/D_out)*cos(omega_new'*data'+b_new);
end

