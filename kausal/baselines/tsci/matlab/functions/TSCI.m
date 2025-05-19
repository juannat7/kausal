function [Coefficients] = TSCI(xState,yState,xVect,yVect)
%% Pushing forward tangent vectors
% In this code, we assume that we have observation vectors of points on
% manifolds Mx and My. 
%   We proceed by estimating the velocity vector fields Vx and Vy on Mx 
%   and My, and then by pushing forward the vector field on Mx to My. 
%   If the degree of alignement between F_*Vx and Vy is high, we conclude
%   that Mx cross-maps onto My, which indicates the reverse causality.
%
% Coefficients(1)  =   Corr(V_x,F_*Vy) = Causality X --> Y
% Coefficients(2)  =   Corr(F_*V_x,Vy) = Causality X <-- Y



% % Normalize the tangent vectors
% xVect = vectnormalize(xVect);
% yVect = vectnormalize(yVect);

% Get dimensionality
Qx = size(xVect,2);
Qy = size(yVect,2);

%% Pushforward the tangent vectors
% This operation amounts to two steps:
%   for each data point x_n
%       1. Compute the velocity vector v(x_n)
%       2. Multiply v(x_n) by the Jacobian matrix DF(x_n)
xPushed = zeros(size(yVect));
K = 3*Qx;

for n = 1:size(xState,1)
	[dists,ids] = mink( sum( (xState - xState(n,:)).^2,2 ), K);
    xTangents = xState(ids,:) - xState(n,:);
    yTangents = yState(ids,:) - yState(n,:);
    J = xTangents\yTangents;
% 	w = exp(-dists)/sum(exp(-dists));
    xPushed(n,:) = xVect(n,:)*J;
end

%% REVERSED Pushforward the tangent vectors
% This operation amounts to two steps:
%   for each data point x_n
%       1. Compute the velocity vector v(x_n)
%       2. Multiply v(x_n) by the Jacobian matrix DF(x_n)
yPushed = zeros(size(xVect));
K = 3*Qy;

for n = 1:size(yState,1)
	[dists,ids] = mink( sum( (yState - yState(n,:)).^2,2 ), K);
    xTangents = xState(ids,:) - xState(n,:);
    yTangents = yState(ids,:) - yState(n,:);
    Jr = yTangents\xTangents;
% 	w = exp(-dists)/sum(exp(-dists));
    yPushed(n,:) = yVect(n,:)*Jr;
end


%% Compute correlation coefficient between vector-valued time series
dotprods = sum(yVect.*xPushed,2);
mags1 = sum(yVect.*yVect,2);
mags2 = sum(xPushed.*xPushed,2);
normalized = dotprods./sqrt(mags1.*mags2);
Coefficients(2) = mean(normalized);
% Coefficients(2)  =   Corr(F_*V_x,Vy) = Causality X <-- Y


%% REVERSE Compute correlation coefficient between vector-valued time series
dotprods = sum(xVect.*yPushed,2);
mags1 = sum(xVect.*xVect,2);
mags2 = sum(yPushed.*yPushed,2);
normalized = dotprods./sqrt(mags1.*mags2);
Coefficients(1) = mean(normalized);
% Coefficients(1)  =   Corr(V_x,F_*Vy) = Causality X --> Y
end

