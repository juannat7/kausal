function dxdt = discreteVelocity(x)
%GETVELOCITY x is a column vector
T = size(x,1);
dxdt = zeros(size(x));
dxdt(2:T) = diff(x(:));
dxdt(1) = dxdt(2);
end

