close all;
load('Houses.mat');
x_need = 6;
x = iter_D(1:x_need);
y1 = DKLA_loop(1:x_need,2);
e1 = DKLA_loop(1:x_need,4);
y2 = Algo_loop(1:x_need,2);
e2 = Algo_loop(1:x_need,4);

figure;
errorbar(x,y1,e1,'LineWidth',8,'CapSize',20,'Color','#1E90FF');
% errorbar(x,y1,e1,'LineWidth',8,'CapSize',20,'Color','#EC9E51');
hold on;
errorbar(x,y2,e2,'LineWidth',8,'CapSize',20,'Color','#FFB90F');
% errorbar(x,y2,e2,'LineWidth',8,'CapSize',20,'Color','#2AB8D5');

set(gca,'FontName','Times New Roman','FontSize',50);
set(gca,'XLim',[x(1),x(x_need)]);
set(gca,'XTick',x);
% xlabel('$\bar{D}$','FontSize',40,'Interpreter','latex');
xlabel('Average number of node features','FontSize',60,'Interpreter','latex');

ylabel('RSE','FontSize',60,'Interpreter','latex');
leg = legend('DKLA','Ours','FontSize',60);
leg.ItemTokenSize = [80,60];
grid on;
set(figure(1),'Position',[50,50,1600,900]);