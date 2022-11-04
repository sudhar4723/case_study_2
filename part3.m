clc; clear; close all;
% Load data
load('mockdata.mat')

traindata = [newInfections.' cumulativeDeaths.'];

preVaxData = traindata(1:100,:);
postVaxData = traindata(100:length(traindata),:);
%% Train the model

%% Fit the model for the prevax data
i0 = [1 0 0 0];
costPreVaxfun= @(x) cost(x,preVaxData,i0);

[x_opt, minval] = findmin(costPreVaxfun);
y_prevax = predicting(x_opt, length(preVaxData), i0);

cases_proj = x_opt(1)*y_prevax(:,1);
deaths_proj = y_prevax(:,4);
figure;
% Plot the deaths
plot(y_prevax(:,4)); hold on; plot(preVaxData(:,2)); hold on;
% Plot the cases
plot(x_opt(1)*y_prevax(:,1)); hold on; plot(preVaxData(:,1)); hold off;
title('Prediction for pre vax')
xlabel('Days', 'FontSize',10)
ylabel('Fraction of total population','FontSize',10)
legend('Predicted deaths', 'Actual deaths', 'Predicted  cases', 'Actual cases', Location='northwest',fontsize=9)

%% Fit the model for the postvax data

ics = [y_prevax(length(y_prevax), : ) 0];
x0 = [x_opt zeros(1,4)];
costPostVax= @(x) costSIDRV(x,postVaxData,ics);

[opt_rate , minCost] = findminSIDRV(costPostVax, x0);

y_postvax = predictingSIDRV(opt_rate,length(postVaxData), ics);

figure;
% Plot the deaths

plot(y_postvax(:,4)); hold on; plot(postVaxData(:,2)); hold on;
% Plot the cases
plot(opt_rate(1)*y_postvax(:,1)); hold on; plot(postVaxData(:,1)); hold off;
xlabel('Days', 'FontSize',10)
title('Prediction for post vac')
ylabel('Fraction of total population','FontSize',10)
legend('Predicted deaths', 'Actual deaths', 'Predicted  cases', 'Actual cases', Location='best',fontsize=9)

%% IMPLEMENTATIONS OF FUNCTIONS AND MODEL
%% SIDR model
% Note about the parameters of predicting function
% .     x(1) : The infection rate from susceptible
% .     x(2) : The fatality rate from COVID
% .     x(3) : Recover rate from COVID
% .     x(4) : Recover rate but susceptible to contract covid again
% .     initial_conditions(1,2,3,4): inital susceptible, infected,
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
% Note that this function assumes that the population is totally
% susceptible
function f = cost(x,data,ics)
    % Set the initial condtions 
    % Setting the time frame
    timeframes = length(data);
    % Fitting the data
    y = predicting(x, timeframes, ics);
    % Calculating the cost functions
    comparableY = [ x(1)*y(:, 1)  y(:, 4)];
    diff = data-comparableY;
    % Matrix of squared errors
    SE = arrayfun(@(n) norm(diff(n,:)), 1:size(diff,1));
    f = mean(SE);

end

%% Find global minimum
function [x_min, minval] = findmin(f)
minval = 1;
  
    for x1=0:0.5:1
        for x2=0:0.5:1
            for x3=0:0.5:1
                for x4=0:0.5:1
                    % Settings up constrains
                    A=[];
                    b=[];
                    %% set up some fixed constraints
                    
                    Af = [];
                    bf = [];
                    %% set up upper and lower bound constraints
                    ub = [1 1 1 1]';
                    lb = [0 0 0 0]';

                    x0 = [x1 x2 x3 x4];
                    % This option disables the output of the fmincon
                    options = optimoptions('fmincon','Display', 'off');

                    [x,mincost] = fmincon(f,x0,A,b,Af,bf,lb,ub,[],options);
                    % Update if we found a better solution 
                    if mincost < minval
                        minval = mincost;
                        x_min = x;
                    end
                end
            end
        end
    end
end
%% SIDRVax model
%% Predicting function
% Note about the parameters of predicting function
% .     x(1) : The infection rate from susceptible
% .     x(2) : The fatality rate from COVID
% .     x(3) : Recover rate from COVID
% .     x(4) : Recover rate but susceptible to contract covid again
% .     x(5) : Vaccination rate
% .     x(6) : Breakthrough rate( vaxxed then got infected again)
% .     x(7) : rate of vaccination in the infected group
% .     x(8) : permanent immune from vaxxed people
% .     initial_conditions(1,2,3,4,5): inital susceptible, infected,
% recovery, deaths, vaxxed respectively
function f = predictingSIDRV(x,t,initial_conditions)

    % Set up transmission constants
    x(5)=0.2;
    k_infections = x(1);
    k_fatality = x(2);
    k_recover = x(3);
    k_return = x(4);
    k_vacination_rate = x(5);
    k_breakthrough = x(6);
    k_infected_vaxxed = x(7);
    k_vax_recover = x(8);
    k_stay_in_infected = 1 - k_fatality - k_recover - k_return - k_breakthrough - k_infected_vaxxed;
    k_stay_in_susc = 1 - k_infections - k_vacination_rate;
    k_stay_in_vax = 1 - k_breakthrough - k_vax_recover;
    % Initial conditions
    ic_susc = initial_conditions(1);
    ic_inf = initial_conditions(2);
    ic_rec = initial_conditions(3);
    ic_fatality = initial_conditions(4);
    ic_vaxed =  initial_conditions(5);
    % Set up SIRDV within-population transmission matrix
    A= [
        k_stay_in_susc          k_return                0   0   0   ;
        k_infections            k_stay_in_infected      0   0   k_breakthrough;   
        0                       k_recover               1   0   k_vax_recover;
        0                       k_fatality              0   1   0   ;
        k_vacination_rate       k_infected_vaxxed       0   0   k_stay_in_vax                      
    ];
    B = zeros(5,1);
    
    % Set up the vector of inital conditions
    x0 = [ic_susc ic_inf ic_rec ic_fatality ic_vaxed];
    
    sys_sir_base = ss(A,B,eye(5),zeros(5,1),1);
    y = lsim(sys_sir_base,zeros(t,1),linspace(0,t-1,t),x0);
    f = y;
end
%% Cost function
function f = costSIDRV(x,data,ics)
    % Set the initial condtions 
    % Setting the time frame
    timeframes = length(data);
    % Fitting the data
    y = predictingSIDRV(x, timeframes, ics);
    % Calculating the cost functions
    comparableY = [ x(1)*y(:, 1)  y(:, 4)];
    diff = data-comparableY;
    % Matrix of squared errors
    SE = arrayfun(@(n) norm(diff(n,:)), 1:size(diff,1));
    f = mean(SE);

end
%% Find global minimum for SIDRV
function [x_min, minval] = findminSIDRV(f,x0)
  
    % Settings up constrains
    A=[];
    b=[];
    %% set up some fixed constraints
    
    Af = [];
    bf = [];
    %% set up upper and lower bound constraints
    ub = ones(1,8).';
    lb = zeros(1,8).';

    
    % This option disables the output of the fmincon
    options = optimoptions('fmincon','Display', 'off');

    [x_min,minval] = fmincon(f,x0,A,b,Af,bf,lb,ub,[],options);
                 
end