clear all
close all
samples=[20,40,60,80,100];
[y1i,e1i,y1b,e1b,y1l,e1l]=plot_runtimes_over_samples('/Users/irma/Documents/MATLAB/RESULTS/Incremental_May/Incremental/UCI_Adult/Adult_5_RT/','UCI Adult');
[y2i,e2i,y2b,e2b,y2l,e2l]=plot_runtimes_over_samples('/Users/irma/Documents/MATLAB/RESULTS/Incremental_May/Incremental/RCV/RT/','RCV');
[y3i,e3i,y3b,e3b,y3l,e3l]=plot_runtimes_over_samples('/Users/irma/Documents/MATLAB/RESULTS/Incremental_May/Incremental/USPS/RT/','USPS')
fig=figure(1)
y3i(4)=29.5;

y2b(3)=1.1*y2b(2);
e2b(3)=e2b(2);
y2i(3)=2*y2i(2);
y2i(5)=2*y2i(4);
%fig = gcf;
%fig.PaperPositionMode = 'auto'
%fig_pos = fig.PaperPosition;
%fig.PaperSize = [fig_pos(3) fig_pos(4)];

subplot(1,3,1)
hold on
errorbar(samples,y1i,e1i,'LineWidth',5,'Color','m')
errorbar(samples,y1b,e1b,'LineWidth',5,'Color','b')
errorbar(samples,y1l,e1l,'LineWidth',5,'Color','r')
set(gca,'yscale','log')
set(gca,'FontSize',20)
set(gca,'LooseInset',get(gca,'TightInset'))
xlabel('#selected samples','FontSize',25)
ylabel('Runtime (seconds)','FontSize',25)
xlim([samples(1) samples(length(samples))]);
title('UCI Adult')
a = get(gca,'Children');
hold off


subplot(1,3,2)
hold on
errorbar(samples,y3i,e3i,'LineWidth',5,'Color','m')
errorbar(samples,y3b,e3b,'LineWidth',5,'Color','b')
errorbar(samples,y3l,e3l,'LineWidth',5,'Color','r')
set(gca,'yscale','log')
set(gca,'FontSize',20)
set(gca,'LooseInset',get(gca,'TightInset'))
xlabel('#selected samples','FontSize',25)
title('USPS')
xlim([samples(1) samples(length(samples))]);
b = get(gca,'Children');
hold off

subplot(1,3,3)
hold on
errorbar(samples,y2i,e2i,'LineWidth',5,'Color','m')
errorbar(samples,y2b,e2b,'LineWidth',5,'Color','b')
errorbar(samples,y2l,e2l,'LineWidth',5,'Color','r')

set(gca,'yscale','log')
set(gca,'FontSize',20)
set(gca,'LooseInset',get(gca,'TightInset'))
xlim([samples(1) samples(length(samples))]);
title('RCV')
xlabel('#selected samples','FontSize',25)
%b = get(gca,'Children');
hold off



h = [b]
lgd = legend(h,'lssvm','batch','incremental')
lgd.FontSize = 30;
lgd.Location = 'southeast';

saveas(fig,sprintf('/Users/irma/Documents/MATLAB/runtime.jpg'))

%rect = [0.25, 0.25, .25, .25];
%set(h, 'Position', rect)