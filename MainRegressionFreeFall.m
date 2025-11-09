clc; clear; close all;
% training data
x = [0.1 0.2 0.3 0.4 0.5];
s = [5.49 -9.45 -31.39 -75.59 -113.69]*1e-2;
% regression process
A = [0.5*x.^2; x; ones(size(x))]'; y = s';
beta = A\y;
% regression function
xfit = 0:0.01:0.5; 
yfit = 0.5*beta(1)*xfit.^2 + beta(2)*xfit + beta(3);
% plot data and curve
plot(x,s,'om','LineWidth',2);
hold on;
plot(xfit,yfit,'-b','LineWidth',2);
hold off; 
xlabel('time'); ylabel('Heigth'); 
set(gca,'FontSize',18,'color','none');