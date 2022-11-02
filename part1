% Exploring SIRD Model based of the Textbook
A = [.95 .04 0 0;
     0.05  .85 0 0;
     0  .1  1 0;
     0 .01  0 1];
x1 = [1;
    0;
    0;
    0];

matrix = [];
x = x1;
for i=1: 200
    matrix(:,i) = A * x;
    x = matrix(:,i);
end

time = linspace(1,200,200);
figure;
plot(time, matrix(1,:))
hold on
plot(time, matrix(2,:))
plot(time, matrix(3,:))
plot(time, matrix(4,:))
legend("Susceptible", "Infected", "Recovered", "Deceased", location='eastoutside');
hold off 
xlabel("Time t");
ylabel("x_t");
title("SIRD Textbook Graph"); 
 
% Exploring SIRD Model based on possibility of re-infections
B = [.97 .04 0 0;
     .02 .86 .2 0;
     .01 .09  .8 0;
     0 .01 0 1];
y = [0;
    0.5;
    0.5;
    0];
reinfection = [];
for i=1: 500
    reinfection(:,i) = B * y;
    y = reinfection(:,i);
end

time = linspace(1,500,500);
figure;
plot(time, reinfection(1,:))
hold on
plot(time, reinfection(2,:)) 
plot(time, reinfection(3,:))
plot(time, reinfection(4,:))
legend("Susceptible", "Infected", "Recovered", "Deceased", location='eastoutside');
hold off 
xlabel("Time t");
ylabel("x_t");
title("SIRD Re-Infections Graph");
