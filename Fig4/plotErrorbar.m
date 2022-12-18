x = 70:10:150;
figure;
errorbar(x,y_DKLA,e_DKLA);
hold on;
errorbar(x,y_Algo,e_Algo);
legend('DKLA','Algo');
title('Train RSE');