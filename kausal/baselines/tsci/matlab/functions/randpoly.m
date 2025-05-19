function [f,coeff] = randpoly(order)
%RANDPOLY Generates a random polynomial function that acts entrywise) on a 
% column vector
coeff = randn(order+1,1)./factorial(0:order)';
f = @(x) (x.^(0:order))*coeff;
end

