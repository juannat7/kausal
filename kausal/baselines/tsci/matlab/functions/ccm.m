function [CC,My,gx] = ccm(x,y,Q,tau)
% CCM
% Computes the convergent cross-mapping (CCM) for the hypothesis that
% x is causally influencing y. 
%
% Inputs:
%   x,y are Nx1 column vectors representing signals in time
%   Q       is the embedding dimension used in state-space reconstruction (SSR)
%   tau     is the embedding delay used in SSR
%
% Outputs:
%   CC  convergence coefficients estimates in time
%   My  SSR embedding of the shadow manifold of the signal y(t) 
%   gx  sequential cross-map estimates of the signal x(t)

% Normalization of input signals
xmu = mean(x);
xsig= std(x);
x = normalize(x);
y = normalize(y);

% Step 1 - State space reconstruction (delay embedding)
My = takens(y,Q,tau);

% Step 2 - Cross-mapping
n0 = max((Q-1)*tau,1); % Time at which we have our first SSR vector
N = size(My,1); % Look at the first few guys 
K = Q+1;        % No. of points in a putative nbhd
gx = zeros(N,size(x,2)); % cross-map estimate of the time-series x(t)
for n = 1:N
    gx(n,:) = simplex(My([1:n-1,n+1:N],:),x([n0:n0+n-1,n0+n+1:n0+N],:), My(n,:));
end

% Save the cross-map result in the original units and timescale,
% since I provide gx(t) as an output
gx = gx*xsig + xmu;

% Step 3 - Convergence coefficients
CC = NaN(size(gx,1),1);
if size(x,2) > 1
    for t = (Q-1)*tau+1:size(gx,1)
        V1 = real(x((Q-1)*tau+(+1:t),:));
        V2 = real(gx(+1:t,:));
        CC(t) = abs(corr(  V1(:), V2(:) ,'Type','Pearson'));
    end 
else
    for t = (Q-1)*tau+1:size(gx,1)
        CC(t) = abs(corr(  real(x((Q-1)*tau+(+1:t))), real(gx(+1:t))  ,'Type','Pearson'));
    end
end

