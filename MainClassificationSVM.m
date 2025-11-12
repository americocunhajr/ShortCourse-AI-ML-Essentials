clc; clear; close all;
% data
X = [randn(100,2)+2 ; randn(100,2)-2];
y = [ones(100,1) ; -ones(100,1)];
% train a SVM
svm = fitcsvm(X,y,'KernelFunction','linear');
% classify a new observation
xNew = [0 0];
pred = predict(svm,xNew)
% visual check
gscatter(X(:,1),X(:,2),y,'rb','o',8); hold on
plot(xNew(1),xNew(2),'ks','MarkerSize',10)
w = svm.Beta; b = svm.Bias;
xline = linspace(min(X(:,1))-1,max(X(:,1))+1,100);
yline = -(w(1)/w(2))*xline - b/w(2);
plot(xline,yline,'k-','LineWidth',1.5);
legend('class -1','class +1','query point')
xlabel('feature 1'); ylabel('feature 2');
set(gca,'FontSize',14,'Color','none');