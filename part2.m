clc; clear; close all;
% Load data
load('COVIDdata.mat')
load('mockdata.mat')

%% Initalize variables
timelength = length(COVID_STLmetro.cases);

%% Preprocess the data

STLmetroPop = STLmetroPop * 100000;

% Cnvert it to percentage
COVID_STLmetro.cases = COVID_STLmetro.cases/STLmetroPop;
COVID_STLmetro.deaths = COVID_STLmetro.deaths/STLmetroPop;


%% Implementation of the cost function
%% Train model on the STL data

newcases = [0, diff(COVID_STLmetro.cases).'].';
COVIDData =  [newcases COVID_STLmetro.deaths];

costfun= @(x) cost(x,COVIDData);
%% set up rate and initial condition constraints
% Set A and b to impose a parameter inequality constraint of the form A*x < b
% Note that this is imposed element-wise
% If you don't want such a constraint, keep these matrices empty.
A=[];
b=[];
%% set up some fixed constraints

Af = [];
bf = [];
%% set up upper and lower bound constraints
ub = [1 1 1 1]';
lb = [0 0 0 0 ]';
% Specify some initial parameters for the optimizer to start from
x0 = [0.05 0.01 0.1 0.04]; 
% This is the key line that tries to opimize your model parameters in order to
% fit the data
% note tath you 
x = fmincon(costfun,x0,A,b,Af,bf,lb,ub);

y = predicting(x,timelength,[1 0 0 0]);

%% IMPLEMENTATIONS OF FUNCTIONS
%% Predicting function
% Note about the parameters of predicting function
% .     x(1) : The infection rate from susceptible
% .     x(2) : The fatality rate from COVID
% .     x(3) : Recover rate from COVID
% .     x(4) : Recover rate but susceptible to contract covid again
% .     iinitial_conditions(1,2,3,4): inital susceptible, infected,
% recovery, deaths respectively
function f = predicting(x,t,initial_conditions)

    % Set up transmission constants
    k_infections = x(1);
    k_fatality = x(2);
    k_recover = x(3);
    k_return = x(4);
    k_stay_in_infected = 1 - k_fatality - k_recover - k_return;
    % Initial conditions
    ic_susc = initial_conditions(1);
    ic_inf = initial_conditions(2);
    ic_rec = initial_conditions(3);
    ic_fatality = initial_conditions(4);
    
    % Set up SIRD within-population transmission matrix
    A= [
        1 - k_infections    k_return                0   0;
        k_infections        k_stay_in_infected      0   0;   
        0                   k_recover               1   0;
        0                   k_fatality              0   1;
    ];
    B = zeros(4,1);
    
    % Set up the vector of inital conditions
    x0 = [ic_susc ic_inf ic_rec ic_fatality];
    
    sys_sir_base = ss(A,B,eye(4),zeros(4,1),1);
    y = lsim(sys_sir_base,zeros(t,1),linspace(0,t-1,t),x0);
    legend('S','I','R','D');
    f = y;
end

%% Cost function
% Note about the parameters of cost function
% .     x(1) : The infection rate from susceptible
% .     x(2) : The fatality rate from COVID
% .     x(3) : Recover rate from COVID
% .     x(4) : Recover rate but susceptible to contract covid again
% . Data should only be in the form of [%newcases %deaths] 
% Note that this functions assume that the population is totally
% susceptible
function f = cost(x,data)
    % Set the initial condtions 
    ics = [1 0 0 0];
    % Setting the time frame
    timeframes = length(data);
    % Fitting the data
    y = predicting(x, timeframes, ics);
    % Calculating the cost functions
    comparableY = [ x(1)*y(:, 1)  y(:, 4)];
    diff = normalize(data-comparableY,2,'scale');
    % Matrix of squared errors
    SE = arrayfun(@(n) norm(diff(n,:)), 1:size(diff,1));
    f = mean(SE);

end