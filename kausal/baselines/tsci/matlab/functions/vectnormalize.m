function [Vnorm] = vectnormalize(V)
%VECTNORMALIZE Summary of this function goes here
%   Detailed explanation goes here
Vnorm = V./sqrt(sum(V.^2,2));
end

