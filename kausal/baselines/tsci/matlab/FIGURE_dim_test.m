%% Dimension test
% Produces the figure with Qx and Qy varied for both CCM and TSCI
rng(0)

% Config
autocorrThresholdForSSR = 0.5;

% Initialize
QxMax = 10;
QyMax = 10;
TSCImat = zeros(QxMax,QyMax);
CCMmat = zeros(QxMax,QyMax);
revTSCImat = zeros(QxMax,QyMax);
revCCMmat = zeros(QxMax,QyMax);

%% Data generative model
C = 1;
odefun = @(t,x) [-6*(x(2)+x(3))  6*(x(1)+0.2*x(2))  6*(0.2 + x(3)*(x(1)-5.7))  10*(-x(4)+x(5))  28*x(4)-x(5)-x(4)*x(6)+C*x(2)^2  x(4)*x(5)-8*x(6)/3, -x(8),x(7),-sqrt(2)*x(10),sqrt(2)*x(9)]';
tspan = linspace(0,100,8000);


Z0 = [ -0.82   -0.80   -0.24    10.01    -12.19    10.70 1 0 1 0];
Z0 = Z0 + randn(size(Z0))*1e-3;
[t,Z] = ode45(odefun,tspan,Z0);

xSignal = Z(:,2);
ySignal = Z(:,4);

% Parameter selection for Takens' embedding
taux = lag_select(xSignal,autocorrThresholdForSSR);
tauy = lag_select(ySignal,autocorrThresholdForSSR);

QxDefault = falsenearestneighbors(xSignal,taux,0.01,8);
QyDefault = falsenearestneighbors(ySignal,tauy,0.01,8);

for qx = 2:QxMax
    for qy = 2:QyMax
        [qx,qy]

        %% Takens embedding
        % also called delay embedding, and state-space reconstruction
        xSignal = Z(:,2);
        ySignal = Z(:,4);
        
        % Use the pre-selected dimensions
        Qx = qx;
        Qy = qy;

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
        [Coefficients] = TSCI(xState,yState,xVect,yVect);

        %% CCM
        CC  = ccm(xSignal,ySignal,Qy,tauy); % X ---> Y
        CCr = ccm(ySignal,xSignal,Qx,taux);

        %% Save results
        % Forward direction
        TSCImat(qx,qy) = Coefficients(1);
        CCMmat(qx,qy)  = CC(end);

        % Reverse direction
        revTSCImat(qx,qy) = Coefficients(2);
        revCCMmat(qx,qy)  = CCr(end);
    end
end


%% Plot
figure(16)
tiledlayout(2,2,'TileSpacing','compact','Padding','compact')

nexttile
imagesc(TSCImat)
title('TSCI  ($X \to Y$)','FontSize',14,'Interpreter','latex')
colorbar
colormap(cyno)
clim([-1, 1])
xlim([1.5,QyMax+0.5])
ylim([1.5,QxMax+0.5])

ylabel('Q_x')
hold on;
plot(0*xlim + QyDefault,ylim,'r', xlim, QxDefault+0*ylim,'r');
hold off;

nexttile
imagesc(revTSCImat)
title('TSCI  ($Y \rightarrow X$)','FontSize',14,'Interpreter','latex')
colorbar
colormap(cyno)
clim([-1, 1])
xlim([1.5,QyMax+0.5])
ylim([1.5,QxMax+0.5])

hold on;
plot(0*xlim + QyDefault,ylim,'r', xlim, QxDefault+0*ylim,'r');
hold off;

nexttile
imagesc(CCMmat)
title('CCM  ($X \rightarrow Y$)','FontSize',14,'Interpreter','latex')
colorbar
clim([-1, 1])
xlim([1.5,QyMax+0.5])
ylim([1.5,QxMax+0.5])

xlabel('Q_y')
ylabel('Q_x')
hold on;
plot(0*xlim + QyDefault,ylim,'r', xlim, QxDefault+0*ylim,'r');
hold off;


nexttile
imagesc(revCCMmat)
title('CCM  ($Y \rightarrow X$)','FontSize',14,'Interpreter','latex')
colorbar
colormap(cyno)
clim([-1, 1])
xlim([1.5,QyMax+0.5])
ylim([1.5,QxMax+0.5])

xlabel('Q_y')
hold on;
plot(0*xlim + QyDefault,ylim,'r', xlim, QxDefault+0*ylim,'r');
hold off;
xlim([1.5,QyMax+0.5])
ylim([1.5,QxMax+0.5])


%% Functions
function colorscheme = cyno(L)
if nargin < 1
    L = 100;
end
ctop = [2, 140, 222]/256; % a nice blue
cbot = [222, 119, 2]/256; % a nice orange
cmid = [255,255,255]/256;  % a white-ish color
t   = linspace(0,1,L)';
colorscheme = [cmid.*t + (1-t)*cbot;ctop.*t + (1-t)*cmid];
end