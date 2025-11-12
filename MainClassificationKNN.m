clc; clear; close all;
% labelled dataset
n = 1e4;
X = [randn(n/2,2)*0.8 +  [ 2  2] ; ...
     randn(n/2,2)*0.8 + [-2 -1]];
y = [ones(n/2,1) ; 2*ones(n/2,1)];
% KNN model trainig
k     = 3;
model = fitcknn(X,y,'NumNeighbors',k);
% classify a new point
xNew = [3 3]; pred = predict(model,xNew);
disp(pred)
% visual check
gscatter(X(:,1),X(:,2),y,'rb','o',8); hold on
plot(xNew(1),xNew(2),'ks','MarkerSize',10,'LineWidth',2)
legend('class 1','class 2','query point','Location','best')
xlabel('feature 1'); ylabel('feature 2');
set(gca,'FontSize',12,'color','none')