function Yp = simplex(M,Y,Mp)
% Simplex projection script
%   Yp = simplex(M,Y,Mp)
%
%  Inputs:
%   M  input points for training
%   Y  output points for training
%   Mp input points for prediction
%
%  Outputs:
%   Yp output predictions

if size(M,2) ~= size(Mp,2)
	error('Training and prediction inputs must have equal dimension.')
end

K = size(M,2) + 1;
Yp = zeros(size(Mp,1),size(Y,2));

for n = 1:size(Mp)
	[dists,ids] = mink( sum( (M - Mp(n,:)).^2,2 ), K);
	w = exp(-dists)/sum(exp(-dists));
	Yp(n,:) = w'*Y(ids,:);
end

end