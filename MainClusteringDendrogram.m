clc; clear; close all;
% data
X = randn(100,2);
figure(1)
plot(X(:,1),X(:,2),'r*');
xlabel('x'); ylabel('y');
set(gca,'FontSize',14,'color','none')
% dendrogram
Z = linkage(pdist(X),'ward');
figure(2)
dendrogram(Z)
xlabel('Point index'); ylabel('Linkage distance');
set(gca,'FontSize',14,'color','none')