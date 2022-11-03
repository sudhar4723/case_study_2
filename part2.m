clc; clear; close all;
% Load data
load('COVIDdata.mat')

%% Initalize variables


%% Preprocess the data

STLmetroPop = STLmetroPop * 100000;
timelength = length(COVID_STLmetro.cases);
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

[x_opt,min] = findmin(costfun);

% Predicting the data
y = predicting(x_opt,timelength,[1 0 0 0]);

costfun([0.05 0.01 0.2 0.3])
% PDeaths
figure;
plot(y(:,4)); hold on; plot(COVID_STLmetro.deaths); hold off;
title('Deaths')
legend('Predicted deaths', 'Actual deaths')
%Newcases
figure;
plot(y(:,2)); hold on; plot(newcases); hold off;
title('New cases')
legend('Predicted newcases', 'Actual newcases')




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
    comparableY = [ y(:, 2)  y(:, 4)];
    diff = data-comparableY;
    % Matrix of squared errors
    SE = arrayfun(@(n) norm(diff(n,:)), 1:size(diff,1));
    f = mean(SE);

end

%% Find minimization with a given x0
function [x_min, minval] = findmin(f)
    minval = 1
    for x1=0:0.:1
        for x2=0:0.5:1
            for x3=0:0.5:1
                for x4=0:0.5:1
                    A=[];
                    b=[];
                    %% set up some fixed constraints
                    
                    Af = [];
                    bf = [];
                    %% set up upper and lower bound constraints
                    ub = [1 1 1 1]';
                    lb = [0 0 0 0]';
                    
                    x0 = [x1 x2 x3 x4];
                    
                    [x,cost] = fmincon(f,x0,A,b,Af,bf,lb,ub);
                    minval
                    if cost < minval
                        minval = cost;
                        x_min = x
                    end
                end
            end
        end
    end

end
