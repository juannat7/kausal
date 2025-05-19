%% Figure: Histograms
% Plots Mx and My from the Rossler-Lorenz system, and shows the histograms
% of cosine similarities

% Config
rng(0)
autocorrThresholdForSSR = 0.5;
C = 1;

%% Data generative model
odefun = @(t,x) [-6*(x(2)+x(3)),  6*(x(1)+0.2*x(2)),  6*(0.2 + x(3)*(x(1)-5.7)),  10*(-x(4)+x(5)),  28*x(4)-x(5)-x(4)*x(6)+C*x(2)^2,  x(4)*x(5)-8*x(6)/3]';
tspan = linspace(0,110,8000);
Z0 = [ -0.82   -0.80   -0.24    10.01    -12.19    10.70];
Z0 = Z0 + randn(size(Z0))*1e-3;
[t,Z] = ode45(odefun,tspan,Z0);

%% Takens embedding
% also called delay embedding, and state-space reconstruction
xSignal = Z(:,2);
ySignal = Z(:,4);

% Parameter selection for Takens' embedding
taux = lag_select(xSignal,autocorrThresholdForSSR);
tauy = lag_select(ySignal,autocorrThresholdForSSR);
Qx = 3;
Qy = 8;

% Compute the delay embedding vectors
xState = takens(xSignal,Qx,taux);
yState = takens(ySignal,Qy,tauy);

% Re-index these guys to align in time
truncator = min(size(xState,1),size(yState,1))-100; % Burn in by 100 samples
xState = xState(end-truncator:end,:);
yState = yState(end-truncator:end,:);

%% Tangent vectors in X-manifold
xTimeVelocity = discreteVelocity(xSignal);
yTimeVelocity = discreteVelocity(ySignal);

xVect = takens(xTimeVelocity,Qx,taux);
yVect = takens(yTimeVelocity,Qy,tauy);

% Re-index these guys to align in time
xVect = xVect(end-truncator:end,:);
yVect = yVect(end-truncator:end,:);

% Normalize the tangent vectors
xVect = vectnormalize(xVect);
yVect = vectnormalize(yVect);

%% Tangent Space Causal Inference

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

for n = 1:size(xState)
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

for n = 1:size(yState)
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
revnormalized = dotprods./sqrt(mags1.*mags2);
Coefficients(1) = mean(revnormalized);
% Coefficients(1)  =   Corr(V_x,F_*Vy) = Causality X --> Y

%% CCM
CC  = ccm(xSignal,ySignal,Qy,tauy); % X ---> Y
CCr = ccm(ySignal,xSignal,Qx,taux);

%% Let's get some PDFs
% Generate the sample data
DiscPDFSampler = randn(10000,20);
R = zeros(size(DiscPDFSampler));
for D = 1:size(DiscPDFSampler,2)
    % R = <X,e_1>/norm(X)norm(e_1)
    R(:,D) = DiscPDFSampler(:,1)./sqrt(sum(DiscPDFSampler(:,1:D).^2,2)); 
end

% Reference probability measures
edges = linspace(-1,1,50);
refs = zeros(numel(edges)-1,size(DiscPDFSampler,2));
for D = 1:size(DiscPDFSampler,2)
    refs(:,D) = histcounts(R(:,D),edges);
end

% Get PDFs from the simulation
fwdpdf = histcounts(normalized,edges);
revpdf = histcounts(revnormalized,edges);

refs = refs + 1e-18;
fwdpdf = fwdpdf + 1e-9;
revpdf = revpdf + 1e-9;

% Compute
probDistances = zeros(size(DiscPDFSampler,2),2);
for D = 1:size(DiscPDFSampler,2)
    probDistances(D,1) = jsdist(fwdpdf,refs(:,D));
    probDistances(D,2) = jsdist(revpdf,refs(:,D));
end

[~,fwddim] = min(probDistances(:,1));
[~,revdim] = min(probDistances(:,2));


%% Plot
figure(15)
tiledlayout(2,3,'TileSpacing','compact','Padding','compact')

% Generate some arbitrary indices to plot the tangent vectors for
ids = 3:20:1000;

nexttile([2,1])
plot3(xState(:,1),xState(:,2),xState(:,3),'k')
title('$\mathcal{M}_x$',sprintf('Embedding dim Q=%d',Qx),'FontSize',14,'Interpreter','latex')
view([1 -2 1])
xticks([])
yticks([])
zticks([])
axis equal
hold on
quiver3(xState(ids,1),xState(ids,2),xState(ids,3),...
    xVect(ids,1),xVect(ids,2),xVect(ids,3),...
    'red','LineWidth',2)
quiver3(xState(ids,1),xState(ids,2),xState(ids,3),...
    yPushed(ids,1),yPushed(ids,2),yPushed(ids,3),...
    'green','LineWidth',2)
hold off;


% Generate some arbitrary indices to plot the tangent vectors for
ids = 3:52:5000;


nexttile([2,1])
plot3(yState(:,1),yState(:,2),yState(:,3),'k')
title('$\mathcal{M}_y$',sprintf('Embedding dim Q=%d',Qy),'FontSize',14,'Interpreter','latex')
view([1 -2 1])
xticks([])
yticks([])
zticks([])
axis equal
hold on
quiver3(yState(ids,1),yState(ids,2),yState(ids,3),...
    yVect(ids,1),yVect(ids,2),yVect(ids,3),...
    'Color',[0.5,0.75,1],'LineWidth',3)
quiver3(yState(ids,1),yState(ids,2),yState(ids,3),...
    xPushed(ids,1),xPushed(ids,2),xPushed(ids,3),...
    'green','LineWidth',3)
hold off;



edges = linspace(-1,1,30);

nexttile
histogram(revnormalized,edges,'FaceColor','red','EdgeColor','white')
title('Alignment of $\mathbf{u}$ with $\mathbf{J}_F \mathbf{v}$ ($r_{X\rightarrow Y}$)',...
    sprintf('Mean=%0.2f',Coefficients(1)),'FontSize',14,'Interpreter','latex')

nexttile
% plot(normalized)
histogram(normalized,edges,'EdgeColor','white')
title('Alignment of $\mathbf{v}$ with $\mathbf{J}_F \mathbf{u}$ ($r_{Y\rightarrow X}$)',...
    sprintf('Mean=%0.2f',Coefficients(2)),'FontSize',14,'Interpreter','latex')
