clc; clear; close all
% training data
xdata = [8.025 10.170 11.202 10.736  9.092];
ydata = [8.310  6.355  3.212  0.375 -2.267];
% regression process
A = [xdata.^2; xdata.*ydata; ydata.^2; xdata; ydata]';
y = - ones(size(xdata))';
beta = A\y;
% plot orbit curve
figure(1)
a = beta(1); b = beta(2); c = beta(3); d = beta(4); e = beta(5);
h = ezplot(@(x,y) a*x.^2 + b*x.*y + c*y.^2 + d*x + e*y + 1,[-5 12 -10 10]);
hold on
plot(xdata,ydata,'om')
hold off
xlabel('x'); ylabel('y'); set(h,'LineWidth',2);
set(gca,'FontSize',18,'color','none')