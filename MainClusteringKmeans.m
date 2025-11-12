clc; clear; close all;
% training data
X = [randn(50,2)*0.3 + [ 2  2];... % cluster 1
     randn(50,2)*0.4 + [-2 -1];... % cluster 2
     randn(50,2)*0.3 + [ 0  4]];   % cluster 3
% k-means process
k        = 3;
[idx, C] = kmeans(X,k,'Replicates',10);
% plot
figure;  hold on
clr = lines(k);
for j = 1:k
    scatter(X(idx==j,1),X(idx==j,2),50,clr(j,:),'filled')
end
plot(C(:,1), C(:,2),'kp','MarkerSize',14,'LineWidth',2)
hold off
xlabel('feature 1');  ylabel('feature 2');
set(gca,'FontSize',16,'color','none')