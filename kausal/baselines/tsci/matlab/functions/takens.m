function [Mnfd,target] = takens(y,Q,tau,k)
% Wrapper script to generate time-delay embedding vectors for state space
% reconstruction
%
% Inputs:
%   y   = your signal as a column vector
%   Q   = embedding dimension
%   tau = embedding lag
%   k   = forecast length, if you're doing that
%
% Outputs:
%   M = the shadow manifold, as an L*Q matrix where L=1+(Q-1)*tau
%   t = forecast targets, if you're doing that

if nargin < 4
    k = 0;
end
if nargin < 3
    tau = 1;
end
N = size(y,1);
idm = (1:tau:1+(Q-1)*tau) + (0:N-(Q-1)*tau-1-k)';
idt = idm(:,end) + k;
Mnfd = y(idm);
target = y(idt);
end