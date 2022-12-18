function omega = cauchyrnd(d,D,x_0,gamma)
if nargin < 2
    error('cauchyrnd_DataShapeError');
elseif nargin < 3
    x_0 = 0;
    gamma = 1;
elseif nargin < 4
    gamma = 1;
end
p = rand(d,D) ;
omega = x_0+gamma*tan(pi*(p-0.5));
end