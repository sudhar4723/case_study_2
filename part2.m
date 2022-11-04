clc; clear; close all;
% Load data
load('COVIDdata.mat')



%% Preprocess the data
% COVID_STLmetro = COVID_STLmetro(1:100,:);
STLmetroPop = STLmetroPop * 100000;
timelength = length(COVID_STLmetro.cases);
% Cnvert it to percentage
COVID_STLmetro.cases = COVID_STLmetro.cases/STLmetroPop;
COVID_STLmetro.deaths = COVID_STLmetro.deaths/STLmetroPop;

% Init necessary variables
newcases = [0, diff(COVID_STLmetro.cases).'].';

partition = [1 256 482 671 length(COVID_STLmetro.cases)];

COVIDData =  [newcases COVID_STLmetro.deaths];

%% Implementation of the cost function

%% Train model on the partitions of STL data
% Initial population
i0 = [1 0 0 0];
rates = [];
sidr = [];
error = [];
for i=2:length(partition)
    
    % Get the partition
    partitionData = COVIDData(partition(i-1): partition(i), :);
    
    % Train this partition
    
    costfun= @(x) cost(x,partitionData,i0);
    [x_opt,min] = findmin(costfun);
    y = predicting(x_opt,length(partitionData),i0);

    % Save results from this partitions 
    sidr = [sidr; y];
    rates = [rates; x_opt];
    error = [error min];

    % Plot the covid
    figure;
    % Plot the deaths
    plot(y(:,4)); hold on; plot(partitionData(:,2)); hold on;
    % Plot the cases
    plot(x_opt(1)*cumsum(y(:,1))); hold on; plot(cumsum(partitionData(:,1))); hold off;
    % Getting the beggining and end of the time frame
    dateStart = COVID_STLmetro(partition(i-1),:).date;
    dateEnd = COVID_STLmetro(partition(i),:).date;
    
    % Other graph's stuff
    title(" Prediction from " +string(dateStart) + " to " + string(dateEnd) )
    legend('Predicted deaths', 'Actual deaths', 'Predicted  cases', 'Actual cases', Location='northwest',fontsize=9)
    xlabel('Time', 'FontSize',10)
    ylabel('Fraction of total population','FontSize',10)
    dateaxis('x',12,dateStart)
    % Set the next state to be the old state
    i0 = y(length(y),:);
    
end

disp('Mean cost function throughout the timeframe ' +  string(mean(error)))

% Plot the whole SIDR for the whole timeframe
figure
plot(sidr)
title('SIDR model for the whole timeframe')
xlabel('Time')
ylabel('Fraction of total population')
dateaxis('x',12,COVID_STLmetro(1,:).date)

%% Implement new policies  


% Find the index of 5/1/2021 and 11/1/2021
indexStart = find(COVID_STLmetro.date == datetime(2021,5,1));
indexEnd = find(COVID_STLmetro.date == datetime(2021,11,1));

% Using the model from partition 3 to predict the data without mask
rateWithoutMask = rates(3,:);
y_wo_mask = predicting(rates(3,:),indexEnd-indexStart, sidr(indexStart,:));

% Predict the data with mask
rateWithMask = rates(3,:);
rateWithMask(1) = rateWithMask(1) * 0.8;

y_w_mask = predicting(rateWithMask,indexEnd-indexStart, sidr(indexStart,:));

% Plot the infected between the two groups
figure
plot(y_wo_mask(:,2)); hold on; plot(y_w_mask(:,2)); hold off;
dateaxis('x',12,datetime(2021,5,1))
title('Comparision of the proportion of infected groups')
legend('Without mask', 'With mask')
xlabel('Time')
ylabel('Fraction of total population')

% Plot the number of deaths
figure
plot(y_wo_mask(:,4)); hold on; plot(y_w_mask(:,4)); hold off;
dateaxis('x',12,datetime(2021,5,1))
title('Comparision of the number of deaths')
xlabel('Time')
ylabel('Fraction of total population')
legend('Without mask', 'With mask');



%% IMPLEMENTATIONS OF FUNCTIONS
%% Predicting function
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
