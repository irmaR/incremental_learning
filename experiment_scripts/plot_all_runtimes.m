samples=[20,40,60,80,100];
[y1i,e1i,y1b,e1b]=plot_runtimes_over_samples('/Users/irma/Documents/MATLAB/RESULTS/Incremental_May/Incremental/UCI_Adult/Adult_5_RT/','UCI Adult');
[y2i,e2i,y2b,e2b]=plot_runtimes_over_samples('/Users/irma/Documents/MATLAB/RESULTS/Incremental_May/Incremental/RCV/RT/','RCV');
[y3i,e3i,y3b,e3b]=plot_runtimes_over_samples('/Users/irma/Documents/MATLAB/RESULTS/Incremental_May/Incremental/USPS/RT/','USPS');
fig=figure
%fig = gcf;
fig.PaperPositionMode = 'auto'
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];

subplot(131)
hold on
errorbar(samples,y1i,e1i,'LineWidth',5,'Color','m')
errorbar(samples,y1b,e1b,'LineWidth',5,'Color','b')
set(gca,'yscale','log')
set(gca,'FontSize',20)
set(gca,'LooseInset',get(gca,'TightInset'))
xlabel('#selected samples','FontSize',20)
ylabel('Runtime (seconds)','FontSize',20)
xlim([samples(1) samples(length(samples))]);
title('UCI Adult')
a = get(gca,'Children');
hold off


subplot(132)
hold on
errorbar(samples,y3i,e3i,'LineWidth',5,'Color','m')
errorbar(samples,y3b,e3b,'LineWidth',5,'Color','b')
set(gca,'yscale','log')
set(gca,'FontSize',20)
set(gca,'LooseInset',get(gca,'TightInset'))
xlabel('#selected samples','FontSize',20)
title('USPS')
xlim([samples(1) samples(length(samples))]);
b = get(gca,'Children');
hold off

subplot(133)
hold on
errorbar(samples,y2i,e2i,'LineWidth',5,'Color','m')
errorbar(samples,y2b,e2b,'LineWidth',5,'Color','b')
set(gca,'yscale','log')
set(gca,'FontSize',20)
set(gca,'LooseInset',get(gca,'TightInset'))
xlim([samples(1) samples(length(samples))]);
title('RCV')
xlabel('#selected samples','FontSize',20)
b = get(gca,'Children');
hold off



h = [b]
lgd = legend(h,'batch','incr')
lgd.FontSize = 30;
lgd.Location = 'southeast';

saveas(fig,sprintf('/Users/irma/Documents/MATLAB/runtime.jpg'))

%rect = [0.25, 0.25, .25, .25];
%set(h, 'Position', rect)