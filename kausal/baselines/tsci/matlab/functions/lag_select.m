function tau = lag_select(x,theta)
% Select tau for SSR using the autocorrelation function
%
% Inputs:
%   x = your signal
%   theta = cutoff parameter for the autocorrelation function. When ACF(tau) < theta, we select tau as the embedding lag.
%
% Outputs:
%   tau, the estimate of the embedding lag

ACF = autocorr(x-mean(x));
if all(ACF>=theta)
    ACF = autocorr(x,min(numel(x)-1, 100));
end
tau = find(ACF<theta,1) - 1;
end