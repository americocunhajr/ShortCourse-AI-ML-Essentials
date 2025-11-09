clc; clear; close all; 
% data
xd = [1 2 3 4 5 6 7 8 9 10];
yd = [0.2 0.5 0.3 3.5 1.0 1.5 1.8 2.0 2.3 2.2];
% prediction error functions
E_2 = @(beta) norm(beta(1)*xd+beta(2)-yd,2);
E_1 = @(beta) norm(beta(1)*xd+beta(2)-yd,1);
% regression process
p_2 = fminsearch(E_2,[1 1]);
p_1 = fminsearch(E_1,[1 1]);
% plot curves
plot(xd,yd,'om',xd(4),yd(4),'or','LineWidth',2);
hold on; ylim([0 4]);
plot(xd,polyval(p_2,xd),'-b' ,'LineWidth',2);
plot(xd,polyval(p_1,xd),'--k','LineWidth',2);
hold off; xlabel('x'); ylabel('y'); 
set(gca,'FontSize',18,'Color','none');
legend('data','outlier','E_2','E_1')