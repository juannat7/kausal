function distance = jsdist(p,q)
% Computes the Jensen-Shannon distance between two probability vectors p
% and q
if numel(p) ~= numel(q)
    error('Size error. Prob distributions should be same length.')
end
if any(p<0) || any(q<0)
    error('Negative values foundi n probability vector!')
end
P = p(:)/sum(p); % should be a column vector now
Q = q(:)/sum(q);
M = 0.5*(P+Q);
distance = 0.5*( Dkl(P,Q) + Dkl(Q,M));
end


function divergence = Dkl(P,Q)
    id = P~=0;
    divergence = -P(id)'*log(Q(id)./P(id));
end