function [Z_train,Z_test] = SERLSRFF(D, W, b, X,XT,y)

n = size(X,2);
Z = sqrt(1/D)*[cos(bsxfun(@plus,W'*X, b'));sin(bsxfun(@plus,W'*X, b'))];

temp = Z*y;%ones(n,1);
M = temp.^2;

l = sum(M);
n_feat_draw = 2*D;
pi_s = M;
qi_s = pi_s/l;
is_wgt = sqrt(1./qi_s);
[~,wgt_order]=sort(is_wgt,'ascend');
w_order = wgt_order((2*D-n_feat_draw+1):end);
WW = [W,W];

w_opm = WW(:,w_order); wgh_opm = is_wgt(w_order);
Z = sqrt(1/D)*[cos(bsxfun(@plus,w_opm(:,1:D)'*X, b'));sin(bsxfun(@plus,w_opm(:,D+1:end)'*X, b'))];
Z_train = repmat(wgh_opm,1,n).*Z;

clear Z

Z1 = sqrt(1/D)*[cos(bsxfun(@plus,w_opm(:,1:D)'*XT, b'));sin(bsxfun(@plus,w_opm(:,D+1:end)'*XT, b'))];
Z_test = repmat(wgh_opm,1,size(XT,2)).*Z1;


