function omega = laprnd(d,D,mu,b)
if nargin < 2
    error('laprnd_DataShapeError');
elseif nargin < 3
    mu = 0;
    b = 1;
elseif nargin < 4
    b = 1;
end
p = rand(d,D) ;
omega = mu-b*sign(p-0.5).*log(1-2*abs(p-0.5));
end

