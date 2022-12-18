close all;
name = ['DKLA','Algo'];
load('Detail.mat');
DKLA = DKLA_loop(:,2);
Algo_sameD = Algo_loop;
Algo_diffD = Algo_loop2;
y=[DKLA,Algo_sameD,Algo_diffD];
b=bar(y,0.95);
axis([0.5,6.5,0,0.035]);
grid on;
hold on;

set(gca,'FontName','Times New Roman','FontSize',31);
set(gca,'YTick',0:0.01:0.03);
set(b(1),'FaceColor','#44A3FF');
set(b(2),'FaceColor','#FDCE5A');
set(b(3),'FaceColor','#FD5D5D');
set(b,'edgecolor','none');
leg = legend('DKLA','Ours (Equal $D_j$)','Ours (Different $D_j$)','Interpreter','latex');
leg.ItemTokenSize = [60,30];
leg.FontSize = 40;
xlabel('The number of data on the $j$th node','FontSize',40,'Interpreter','latex');
ylabel('RSE','FontSize',40,'Interpreter','latex');
xticklabels({'$N_1=\frac{1}{36}N$','$N_2=\frac{3}{36}N$','$N_3=\frac{5}{36}N$',...
    '$N_4=\frac{7}{36}N$','$N_5=\frac{9}{36}N$','$N_6=\frac{11}{36}N$'});
% set(get(gca,'XTick'),'FontSize',14);
set(gca,'TickLabelInterpreter','latex');
set(figure(1),'Position',[50,50,1600,900]);