% MainRosslerForecast.m
%
% Forecasting the chaotic Rossler system with reservoir computing
% using an Echo State Network (ESN).
%
% The script:
%   1. Integrates the Rossler system;
%   2. Removes an initial transient;
%   3. Normalizes using training statistics only;
%   4. Trains the ESN by teacher forcing;
%   5. Performs autonomous closed-loop forecasting;
%   6. Computes forecasting diagnostics;
%   7. Generates publication-quality dark-background figures.
%
% No external toolboxes are required beyond standard MATLAB functions.

clear;
close all;
clc;

% ================================================================
% 1. PARAMETERS
% =================================================================

rng(7,'twister');       % Reproducibility

% Rossler parameters
aRossler = 0.2;
bRossler = 0.2;
cRossler = 5.7;

% Numerical integration
dt            = 0.01;
transientTime = 200;
trainingTime  = 400;
forecastTime  = 200;

initialCondition = [1;1;1];

% ESN architecture
inputDimension  = 3;
outputDimension = 3;

reservoirSize    = 500;
reservoirDensity = 0.02;

spectralRadius = 0.95;
inputScaling   = 0.30;
biasScaling    = 0.20;
leakRate       = 0.35;

washout   = 300;
ridgeBeta = 1e-4;

% Valid-prediction-time threshold
predictionThreshold = 0.40;

% Approximate largest Lyapunov exponent for the standard Rossler regime.
% This value is used only to express the prediction horizon in
% approximate Lyapunov times.
largestLyapunovApprox = 0.0714;

% Output files
timeSeriesFigureFile = 'Rossler_ESN_TimeSeries.png';
attractorFigureFile  = 'Rossler_ESN_Attractor.png';

% ================================================================
% 2. GENERATE THE ROSSLER DATA
% =================================================================

totalTime = transientTime + trainingTime + forecastTime;

integrationTime = 0:dt:totalTime;

rosslerModel = @(t,x) rosslerRHS( ...
    t,x,aRossler,bRossler,cRossler);

odeOptions = odeset( ...
    'RelTol',1e-10, ...
    'AbsTol',1e-12, ...
    'MaxStep',dt);

fprintf('Integrating the Rossler system...\n');

[tRaw,stateRaw] = ode45( ...
    rosslerModel, ...
    integrationTime, ...
    initialCondition, ...
    odeOptions);

% Arrange states by columns:
%
%     data(:,k) = [x(t_k); y(t_k); z(t_k)]
%
dataRaw = stateRaw.';

% ================================================================
% 3. REMOVE THE INITIAL TRANSIENT
% =================================================================

firstDataIndex = find( ...
    tRaw >= transientTime, ...
    1, ...
    'first');

data = dataRaw(:,firstDataIndex:end);

time = tRaw(firstDataIndex:end).' ...
     - tRaw(firstDataIndex);

nTraining = round(trainingTime/dt) + 1;
nForecast = round(forecastTime/dt);

requiredSamples = nTraining + nForecast;

if size(data,2) < requiredSamples
    error('Insufficient samples for the requested training/forecast split.');
end

data = data(:,1:requiredSamples);
time = time(1:requiredSamples);

trainingData = data(:,1:nTraining);
forecastTruth = data(:,nTraining+1:end);

fprintf('Training snapshots: %d\n',nTraining);
fprintf('Forecast snapshots: %d\n',nForecast);

% ================================================================
% 4. NORMALIZATION
% =================================================================
%
% The mean and standard deviation are computed using only the training
% data. Future information is therefore not used during preprocessing.

trainingMean = mean(trainingData,2);
trainingStd  = std(trainingData,0,2);

% Prevent division by zero.
trainingStd(trainingStd < sqrt(eps)) = 1;

normalizedData = normalizeStates( ...
    data, ...
    trainingMean, ...
    trainingStd);

normalizedTraining = ...
    normalizedData(:,1:nTraining);

normalizedForecastTruth = ...
    normalizedData(:,nTraining+1:end);

% ================================================================
% 5. CONSTRUCT THE RESERVOIR
% =================================================================
%
% Spectral radius:
%   Controls the effective memory and stability of the reservoir.
%
% Input scaling:
%   Controls how strongly the observations excite the reservoir.
%
% Leak rate:
%   Controls the time scale of the reservoir response.

% Input matrix including the bias channel.
Win = zeros( ...
    reservoirSize, ...
    1 + inputDimension);

Win(:,1) = ...
    biasScaling*(2*rand(reservoirSize,1)-1);

Win(:,2:end) = ...
    inputScaling*( ...
    2*rand(reservoirSize,inputDimension)-1);

% Sparse recurrent reservoir.
Wres = sprand( ...
    reservoirSize, ...
    reservoirSize, ...
    reservoirDensity);

% Replace nonzero entries by values in [-1,1].
Wres = 2*Wres - spones(Wres);

% Normalize to the requested spectral radius.
estimatedRadius = estimateSpectralRadius(Wres);

if estimatedRadius <= eps
    error('The generated reservoir has negligible spectral radius.');
end

Wres = ...
    (spectralRadius/estimatedRadius)*Wres;

verifiedRadius = estimateSpectralRadius(Wres);

fprintf('Requested spectral radius: %.4f\n',spectralRadius);
fprintf('Verified spectral radius:  %.4f\n',verifiedRadius);

extendedDimension = ...
    1 + inputDimension + reservoirSize;

% ================================================================
% 6. TEACHER-FORCED TRAINING
% =================================================================
%
% Teacher forcing means:
%
%   - the true state at time k is supplied to the reservoir;
%   - the target is the true state at time k+1.
%
% The reservoir update is
%
% r_{k+1} =
%     (1-alpha) r_k
%     + alpha tanh(Win[1;u_k] + Wres r_k).
%
% The washout interval removes dependence on the arbitrary initial
% reservoir state.

reservoirState = zeros(reservoirSize,1);

nTrainingPairs = nTraining - 1;
nRegressionPairs = nTrainingPairs - washout;

if nRegressionPairs <= 0
    error('Washout is too large for the selected training interval.');
end

Xstate = zeros( ...
    extendedDimension, ...
    nRegressionPairs);

Ytarget = zeros( ...
    outputDimension, ...
    nRegressionPairs);

column = 0;

for k = 1:nTrainingPairs

    inputState = normalizedTraining(:,k);

    reservoirState = ...
        (1-leakRate)*reservoirState ...
        + leakRate*tanh( ...
            Win*[1;inputState] ...
            + Wres*reservoirState);

    extendedState = [
        1
        inputState
        reservoirState
    ];

    if k > washout

        column = column + 1;

        Xstate(:,column) = extendedState;

        %Ytarget(:,column) = normalizedTraining(:,k+1);
        Ytarget(:,column) = normalizedTraining(:,k+1) - normalizedTraining(:,k);
    end
end

if column ~= nRegressionPairs
    error('Unexpected number of training regression pairs.');
end

% ================================================================
% 7. RIDGE-REGRESSION OUTPUT TRAINING
% =================================================================
%
% Only Wout is trained:
%
% Wout =
%   Ytarget Xstate' /
%   (Xstate Xstate' + beta I).
%
% Ridge regularization:
%   - stabilizes the regression;
%   - reduces sensitivity to nearly dependent reservoir features;
%   - improves autonomous forecasting robustness.
%
% No explicit inverse is formed.

gramMatrix = Xstate*Xstate.';

regularizedGram = ...
    gramMatrix ...
    + ridgeBeta*eye(extendedDimension);

Wout = ...
    (regularizedGram \ ...
    (Xstate*Ytarget.')).';

if ~isequal( ...
        size(Wout), ...
        [outputDimension,extendedDimension])

    error('Unexpected dimensions for Wout.');
end

% ================================================================
% 8. TEACHER-FORCED TRAINING RECONSTRUCTION
% =================================================================

trainingPredictionNormalized = ...
    nan(outputDimension,nTraining);

reservoirState = zeros(reservoirSize,1);

for k = 1:nTrainingPairs

    inputState = normalizedTraining(:,k);

    reservoirState = ...
        (1-leakRate)*reservoirState ...
        + leakRate*tanh( ...
            Win*[1;inputState] ...
            + Wres*reservoirState);

    extendedState = [
        1
        inputState
        reservoirState
    ];

    %trainingPredictionNormalized(:,k+1) = Wout*extendedState;
    deltaPrediction = Wout*extendedState;
    trainingPredictionNormalized(:,k+1) = normalizedTraining(:,k) + deltaPrediction;
end

% ================================================================
% 9. AUTONOMOUS CLOSED-LOOP FORECASTING
% =================================================================
%
% During forecasting:
%
%   - no true future state is supplied;
%   - the ESN prediction becomes its next input.
%
% This is the demanding autonomous forecasting mode.

autonomousInput = normalizedTraining(:,end);

forecastPredictionNormalized = ...
    zeros(outputDimension,nForecast);

for j = 1:nForecast

    reservoirState = ...
        (1-leakRate)*reservoirState ...
        + leakRate*tanh( ...
            Win*[1;autonomousInput] ...
            + Wres*reservoirState);

    extendedState = [
        1
        autonomousInput
        reservoirState
    ];

    %autonomousOutput = Wout*extendedState;
    %forecastPredictionNormalized(:,j) = autonomousOutput;
    % Closed-loop feedback
    %autonomousInput = autonomousOutput;

    predictedIncrement = Wout*extendedState;
    autonomousOutput = autonomousInput + predictedIncrement;
    forecastPredictionNormalized(:,j) = autonomousOutput;
    autonomousInput = autonomousOutput;

end

% ================================================================
% 10. RETURN TO PHYSICAL COORDINATES
% =================================================================

trainingPrediction = inverseNormalizeStates( ...
    trainingPredictionNormalized, ...
    trainingMean, ...
    trainingStd);

forecastPrediction = inverseNormalizeStates( ...
    forecastPredictionNormalized, ...
    trainingMean, ...
    trainingStd);

% ================================================================
% 11. ERROR ANALYSIS
% =================================================================

forecastNRMSE = zeros(3,1);

for i = 1:3

    forecastNRMSE(i) = computeNRMSE( ...
        forecastPrediction(i,:), ...
        forecastTruth(i,:));
end

% Overall normalized root-mean-square error.
overallForecastNRMSE = sqrt( ...
    mean( ...
        (forecastPredictionNormalized ...
        - normalizedForecastTruth).^2, ...
        'all'));

% Time-dependent Euclidean error in normalized coordinates.
normalizedEuclideanError = sqrt( ...
    mean( ...
        (forecastPredictionNormalized ...
        - normalizedForecastTruth).^2, ...
        1));

thresholdIndex = find( ...
    normalizedEuclideanError > predictionThreshold, ...
    1, ...
    'first');

if isempty(thresholdIndex)

    validPredictionTime = forecastTime;
    thresholdReached = false;

else

    validPredictionTime = ...
        (thresholdIndex-1)*dt;

    thresholdReached = true;
end

validPredictionLyapunovTimes = ...
    validPredictionTime*largestLyapunovApprox;

fprintf('\n');
fprintf('Forecast diagnostics\n');
fprintf('--------------------\n');
fprintf('NRMSE x: %.6f\n',forecastNRMSE(1));
fprintf('NRMSE y: %.6f\n',forecastNRMSE(2));
fprintf('NRMSE z: %.6f\n',forecastNRMSE(3));
fprintf('Overall normalized RMSE: %.6f\n', ...
    overallForecastNRMSE);

if thresholdReached

    fprintf( ...
        'Valid prediction time: %.3f time units\n', ...
        validPredictionTime);

else

    fprintf( ...
        ['The error threshold was not reached. ' ...
         'Valid prediction time is at least %.3f.\n'], ...
        validPredictionTime);
end

fprintf( ...
    'Approximate prediction horizon: %.3f Lyapunov times\n', ...
    validPredictionLyapunovTimes);

% ================================================================
% 12. TIME-SERIES FIGURE
% =================================================================

figureTime = figure( ...
    'Color',[0.03 0.03 0.04], ...
    'Position',[80 60 1450 900], ...
    'Renderer','painters');

layout = tiledlayout( ...
    3,1, ...
    'Padding','compact', ...
    'TileSpacing','compact');

stateNames = {
    'x(t)'
    'y(t)'
    'z(t)'
};

trainingRegionColor = [0.15 0.55 0.75];
forecastRegionColor = [0.38 0.25 0.06];

truthColor = [0.90 0.92 0.96];
trainingPredictionColor = [0.20 0.78 0.92];
forecastPredictionColor = [1.00 0.72 0.15];

boundaryColor = [0.95 0.35 0.30];

trainingEndTime = time(nTraining);

for i = 1:3

    ax = nexttile(layout);
    hold(ax,'on');

    yMinimum = min(data(i,:));
    yMaximum = max(data(i,:));

    yPadding = ...
        0.08*max(yMaximum-yMinimum,1);

    yLimits = [
        yMinimum-yPadding
        yMaximum+yPadding
    ];

    % Training-region shading
    patch( ...
        ax, ...
        [ ...
        time(1), ...
        trainingEndTime, ...
        trainingEndTime, ...
        time(1)], ...
        [ ...
        yLimits(1), ...
        yLimits(1), ...
        yLimits(2), ...
        yLimits(2)], ...
        trainingRegionColor, ...
        'FaceAlpha',0.10, ...
        'EdgeColor','none');

    % Forecast-region shading
    patch( ...
        ax, ...
        [ ...
        trainingEndTime, ...
        time(end), ...
        time(end), ...
        trainingEndTime], ...
        [ ...
        yLimits(1), ...
        yLimits(1), ...
        yLimits(2), ...
        yLimits(2)], ...
        forecastRegionColor, ...
        'FaceAlpha',0.18, ...
        'EdgeColor','none');

    % True trajectory
    plot( ...
        ax, ...
        time, ...
        data(i,:), ...
        'Color',truthColor, ...
        'LineWidth',1.35);

    % Teacher-forced reconstruction
    plot( ...
        ax, ...
        time(1:nTraining), ...
        trainingPrediction(i,:), ...
        'Color',trainingPredictionColor, ...
        'LineWidth',1.15);

    % Autonomous forecast
    plot( ...
        ax, ...
        time(nTraining+1:end), ...
        forecastPrediction(i,:), ...
        'Color',forecastPredictionColor, ...
        'LineWidth',1.65);

    xline( ...
        ax, ...
        trainingEndTime, ...
        '--', ...
        'Color',boundaryColor, ...
        'LineWidth',1.5);

    ylabel( ...
        ax, ...
        stateNames{i}, ...
        'Color',[0.92 0.92 0.95], ...
        'FontWeight','bold');

    ylim(ax,yLimits);
    xlim(ax,[time(1),time(end)]);

    styleDarkAxes(ax);

    if i == 1

        text( ...
            ax, ...
            time(1)+0.02*(time(end)-time(1)), ...
            yLimits(2)-0.12*diff(yLimits), ...
            'TRAINING', ...
            'Color',trainingPredictionColor, ...
            'FontWeight','bold', ...
            'FontSize',12);

        text( ...
            ax, ...
            trainingEndTime ...
            + 0.02*(time(end)-time(1)), ...
            yLimits(2)-0.12*diff(yLimits), ...
            'FORECAST', ...
            'Color',forecastPredictionColor, ...
            'FontWeight','bold', ...
            'FontSize',12);

        legend( ...
            ax, ...
            { ...
            'Training region', ...
            'Forecast region', ...
            'True trajectory', ...
            'Teacher-forced ESN', ...
            'Autonomous ESN', ...
            'End of training'}, ...
            'TextColor',[0.92 0.92 0.95], ...
            'Color',[0.07 0.07 0.09], ...
            'EdgeColor',[0.30 0.30 0.35], ...
            'Location','northeast', ...
            'NumColumns',3);
    end

    if i < 3

        ax.XTickLabel = [];

    else

        xlabel( ...
            ax, ...
            'Time', ...
            'Color',[0.92 0.92 0.95], ...
            'FontWeight','bold');
    end
end

title( ...
    layout, ...
    sprintf( ...
    ['Rössler Forecast with Reservoir Computing  |  ' ...
     'NRMSE = %.3f  |  Valid time = %.2f'], ...
    overallForecastNRMSE, ...
    validPredictionTime), ...
    'Color',[0.96 0.96 0.98], ...
    'FontSize',18, ...
    'FontWeight','bold');

exportFigureSafe( ...
    figureTime, ...
    timeSeriesFigureFile, ...
    250);

% ================================================================
% 13. ATTRACTOR FIGURE
% =================================================================

figureAttractor = figure( ...
    'Color',[0.03 0.03 0.04], ...
    'Position',[120 80 1400 650], ...
    'Renderer','painters');

layout3D = tiledlayout( ...
    1,2, ...
    'Padding','compact', ...
    'TileSpacing','compact');

trainingTrajectory = trainingData;
trueForecastTrajectory = forecastTruth;
predictedForecastTrajectory = forecastPrediction;

% ------------------------------------------------
% True attractor
% ------------------------------------------------

ax1 = nexttile(layout3D);
hold(ax1,'on');

plot3( ...
    ax1, ...
    trainingTrajectory(1,:), ...
    trainingTrajectory(2,:), ...
    trainingTrajectory(3,:), ...
    'Color',[0.28 0.55 0.72], ...
    'LineWidth',0.9);

plot3( ...
    ax1, ...
    trueForecastTrajectory(1,:), ...
    trueForecastTrajectory(2,:), ...
    trueForecastTrajectory(3,:), ...
    'Color',truthColor, ...
    'LineWidth',1.35);

title( ...
    ax1, ...
    'True Rössler attractor', ...
    'Color',[0.96 0.96 0.98], ...
    'FontWeight','bold');

xlabel(ax1,'x');
ylabel(ax1,'y');
zlabel(ax1,'z');

styleDarkAxes(ax1);

grid(ax1,'on');
axis(ax1,'vis3d');
view(ax1,38,24);

legend( ...
    ax1, ...
    {'Training segment','True forecast segment'}, ...
    'TextColor',[0.92 0.92 0.95], ...
    'Color',[0.07 0.07 0.09], ...
    'EdgeColor',[0.30 0.30 0.35], ...
    'Location','best');

% ------------------------------------------------
% ESN-predicted attractor
% ------------------------------------------------

ax2 = nexttile(layout3D);
hold(ax2,'on');

plot3( ...
    ax2, ...
    trainingTrajectory(1,:), ...
    trainingTrajectory(2,:), ...
    trainingTrajectory(3,:), ...
    'Color',[0.28 0.55 0.72], ...
    'LineWidth',0.9);

plot3( ...
    ax2, ...
    predictedForecastTrajectory(1,:), ...
    predictedForecastTrajectory(2,:), ...
    predictedForecastTrajectory(3,:), ...
    'Color',forecastPredictionColor, ...
    'LineWidth',1.45);

title( ...
    ax2, ...
    'ESN autonomous forecast', ...
    'Color',[0.96 0.96 0.98], ...
    'FontWeight','bold');

xlabel(ax2,'x');
ylabel(ax2,'y');
zlabel(ax2,'z');

styleDarkAxes(ax2);

grid(ax2,'on');
axis(ax2,'vis3d');
view(ax2,38,24);

legend( ...
    ax2, ...
    {'Training segment','ESN forecast segment'}, ...
    'TextColor',[0.92 0.92 0.95], ...
    'Color',[0.07 0.07 0.09], ...
    'EdgeColor',[0.30 0.30 0.35], ...
    'Location','best');

% Common axes for a fair visual comparison.
allX = [
    trainingTrajectory(1,:), ...
    trueForecastTrajectory(1,:), ...
    predictedForecastTrajectory(1,:)
];

allY = [
    trainingTrajectory(2,:), ...
    trueForecastTrajectory(2,:), ...
    predictedForecastTrajectory(2,:)
];

allZ = [
    trainingTrajectory(3,:), ...
    trueForecastTrajectory(3,:), ...
    predictedForecastTrajectory(3,:)
];

commonXLimits = [min(allX),max(allX)];
commonYLimits = [min(allY),max(allY)];
commonZLimits = [min(allZ),max(allZ)];

xlim(ax1,commonXLimits);
xlim(ax2,commonXLimits);

ylim(ax1,commonYLimits);
ylim(ax2,commonYLimits);

zlim(ax1,commonZLimits);
zlim(ax2,commonZLimits);

title( ...
    layout3D, ...
    'Rössler Dynamics: True Evolution and Reservoir Forecast', ...
    'Color',[0.96 0.96 0.98], ...
    'FontSize',18, ...
    'FontWeight','bold');

exportFigureSafe( ...
    figureAttractor, ...
    attractorFigureFile, ...
    250);

fprintf('\nFigures exported:\n');
fprintf('  %s\n',timeSeriesFigureFile);
fprintf('  %s\n',attractorFigureFile);

% ================================================================
% 14. PARAMETER-TUNING GUIDE
% =================================================================
%
% Reservoir size
%   Increase toward 700--800 if the model underfits the attractor.
%   Larger reservoirs increase memory and regression cost.
%
% Spectral radius
%   Values between approximately 0.8 and 1.0 are common.
%   Reduce it when autonomous trajectories become unstable.
%   Increase it slightly when reservoir memory is insufficient.
%
% Leak rate
%   Reduce it for slower reservoir dynamics and stronger smoothing.
%   Increase it when the reservoir reacts too slowly.
%
% Input scaling
%   Reduce it if tanh neurons remain near +/-1.
%   Increase it if reservoir activations are too small.
%
% Ridge parameter
%   Increase ridgeBeta if autonomous forecasts are noisy or unstable.
%   Reduce it when the output model visibly underfits the training data.
%
% Training length
%   Increase trainingTime so that the reservoir samples the attractor
%   repeatedly and observes the range of relevant dynamical states.
%
% Washout
%   Increase washout when the initial reservoir condition remains visible.
%   Keep enough post-washout samples for the regression.
%
% Chaotic forecasting
%   Pointwise predictions inevitably separate after a finite horizon.
%   Beyond that horizon, compare attractor geometry and statistics rather
%   than requiring exact phase agreement.

% ================================================================
% LOCAL FUNCTIONS
% =================================================================

function dx = rosslerRHS(~,x,a,b,c)
%ROSSLERRHS Rössler-system right-hand side.

    dx = [
        -x(2)-x(3)
         x(1)+a*x(2)
         b+x(3)*(x(1)-c)
    ];
end

function normalized = normalizeStates(data,mu,sigma)
%NORMALIZESTATES Normalize each state using training statistics.

    normalized = (data-mu)./sigma;
end

function data = inverseNormalizeStates(normalized,mu,sigma)
%INVERSENORMALIZESTATES Return normalized states to physical units.

    data = normalized.*sigma + mu;
end

function value = computeNRMSE(prediction,truth)
%COMPUTENRMSE RMSE normalized by the truth standard deviation.

    denominator = std(truth,0,2);

    if denominator < sqrt(eps)
        denominator = 1;
    end

    value = ...
        sqrt(mean((prediction-truth).^2,2)) ...
        / denominator;
end

function radius = estimateSpectralRadius(W)
%ESTIMATESPECTRALRADIUS Estimate the maximum eigenvalue magnitude.

    try

        dominantEigenvalue = eigs( ...
            W, ...
            1, ...
            'largestabs', ...
            'Tolerance',1e-8, ...
            'MaxIterations',2000);

        radius = abs(dominantEigenvalue);

    catch

        warning( ...
            'eigs failed; using a full eigenvalue decomposition.');

        radius = max(abs(eig(full(W))));
    end
end

function styleDarkAxes(ax)
%STYLEDARKAXES Apply consistent dark-background styling.

    ax.Color = [0.03 0.03 0.04];

    ax.XColor = [0.82 0.84 0.88];
    ax.YColor = [0.82 0.84 0.88];
    ax.ZColor = [0.82 0.84 0.88];

    ax.GridColor = [0.45 0.47 0.52];
    ax.GridAlpha = 0.22;
    ax.MinorGridAlpha = 0.10;

    ax.FontSize = 12;
    ax.LineWidth = 1.0;

    ax.Box = 'on';
    ax.Layer = 'top';
end

function exportFigureSafe(fig,filename,resolution)
%EXPORTFIGURESAFE Export a figure with a compatibility fallback.

    try

        exportgraphics( ...
            fig, ...
            filename, ...
            'Resolution',resolution);

    catch

        print( ...
            fig, ...
            filename, ...
            '-dpng', ...
            sprintf('-r%d',resolution));
    end
end