clc; clear; close all;
% COVID-19 data for Rio de Janeiro
load COVID19RJ.mat;
% regression model and prediction error
num   = @(x,beta) beta(1)*beta(2)*exp(-beta(1)*(x-beta(3)));
den   = @(x,beta) (1+exp(-beta(1)*(x-beta(3)))).^2;
f     = @(x,beta) num(x,beta)./den(x,beta);
E     = @(beta) norm(f(xdata,beta)-ydata);
% nonlinear regression process
b_ast = fminsearch(E,[0.1 8e4 55]);
% plot curves
stem(ydata,'ob','LineWidth',0.5)
hold on
plot(xdata,f(xdata,b_ast),'-r' ,'LineWidth',3)
hold off
xlabel('days'); ylabel('new notifications per day'); 
set(gca,'FontSize',18,'Color','none')