function []=plot_times_observations()

[y1i,y1b,y1l,e1i,e1b,e1l,x]=get_times_observations('/Users/irma/Documents/MATLAB/RESULTS/Incremental_May/Incremental/UCI_Adult/Balanced_Version1/smp_20/bs_100/Supervised/HeatKernel/k_0/','UCI Adult');
% [y2i,e2i,y2b,e2b,y2l,e2l]=get_times_observations('/Users/irma/Documents/MATLAB/RESULTS/Incremental_May/Incremental/RCV/RT/','RCV');
% [y3i,e3i,y3b,e3b,y3l,e3l]=get_times_observations('/Users/irma/Documents/MATLAB/RESULTS/Incremental_May/Incremental/USPS/RT/','USPS')
fig=figure(1)

%subplot(1,3,1)
hold on
errorbar(x,y1i,e1i,'LineWidth',5,'Color','m')
errorbar(x,y1b,e1b,'LineWidth',5,'Color','b')
errorbar(x,y1l,e1l,'LineWidth',5,'Color','r')
%set(gca,'yscale','log')
set(gca,'FontSize',20)
set(gca,'LooseInset',get(gca,'TightInset'))
xlabel('#selected samples','FontSize',25)
ylabel('Runtime (seconds)','FontSize',25)
xlim([x(1) x(length(x))]);
title('UCI Adult')
a = get(gca,'Children');
hold off

% 
% subplot(1,3,2)
% hold on
% errorbar(samples,y3i,e3i,'LineWidth',5,'Color','m')
% %errorbar(samples,y3b,e3b,'LineWidth',5,'Color','b')
% %errorbar(samples,y3l,e3l,'LineWidth',5,'Color','r')
% set(gca,'yscale','log')
% set(gca,'FontSize',20)
% set(gca,'LooseInset',get(gca,'TightInset'))
% xlabel('#selected samples','FontSize',25)
% title('USPS')
% xlim([samples(1) samples(length(samples))]);
% b = get(gca,'Children');
% hold off
% 
% subplot(1,3,3)
% hold on
% errorbar(samples,y2i,e2i,'LineWidth',5,'Color','m')
% %errorbar(samples,y2b,e2b,'LineWidth',5,'Color','b')
% %errorbar(samples,y2l,e2l,'LineWidth',5,'Color','r')
% 
% set(gca,'yscale','log')
% set(gca,'FontSize',20)
% set(gca,'LooseInset',get(gca,'TightInset'))
% xlim([samples(1) samples(length(samples))]);
% title('RCV')
% xlabel('#selected samples','FontSize',25)
 b = get(gca,'Children');
% hold off



h = [b]
lgd = legend(h,'lssvm','batch','incremental')
lgd.FontSize = 30;
lgd.Location = 'northwest';

saveas(fig,sprintf('/Users/irma/Documents/MATLAB/runtimes_vs_observations.jpg'))





[y1i,y1b,y1l,e1i,e1b,e1l,x]=get_processing_times_observations('/Users/irma/Documents/MATLAB/RESULTS/Incremental_May/Incremental/UCI_Adult/Balanced_Version1/smp_20/bs_100/Supervised/HeatKernel/k_0/');
[y2i,e2i,y2b,e2b,y2l,e2l]=get_processing_times_observations('/Users/irma/Documents/MATLAB/RESULTS/Incremental_May/Incremental/RCV/RT/');
[y3i,e3i,y3b,e3b,y3l,e3l]=get_processing_times_observations('/Users/irma/Documents/MATLAB/RESULTS/Incremental_May/Incremental/USPS/RT/')
fig=figure(2)

subplot(1,3,1)
hold on
errorbar(x,y1i,e1i,'LineWidth',5,'Color','m')
errorbar(x,y1b,e1b,'LineWidth',5,'Color','b')
errorbar(x,y1l,e1l,'LineWidth',5,'Color','r')
%set(gca,'yscale','log')
set(gca,'FontSize',20)
set(gca,'LooseInset',get(gca,'TightInset'))
xlabel('#selected samples','FontSize',25)
ylabel('Time (seconds)','FontSize',25)
xlim([x(1) x(length(x))]);
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
 b = get(gca,'Children');
hold off
% 
% 

h = [b]
lgd = legend(h,'lssvm','batch','incremental')
lgd.FontSize = 30;
lgd.Location = 'southeast';

saveas(fig,sprintf('/Users/irma/Documents/MATLAB/runtimes_vs_observations.jpg'))




end















function [mean_time_incr,mean_time_batch,mean_time_lssvm,stdev_time_inct,stdev_time_batch,stdev_time_lssvm,samples]=get_times_observations(path_to_results,title)
report_points=[];
times_incr=[];
times_batch=[];
samples=[];
path_to_incr=sprintf('%s/incr/results.mat',path_to_results);
path_to_batch=sprintf('%s/batch/results.mat',path_to_results);
path_to_lssvm=sprintf('%s/lssvm/results.mat',path_to_results)


counter=1;

if exist(path_to_incr, 'file') == 2
    aucs_inct=load(path_to_incr,'results');
    results=aucs_inct.results;
    for i=1:length(results)
       times_incr(i,:)=results{i}.selection_times;
       samples=results{i}.report_points;
    end
end

if exist(path_to_batch, 'file') == 2
    aucs_inct=load(path_to_batch,'results');
    results=aucs_inct.results;
    for i=1:length(results)
       times_batch(i,:)=results{i}.selection_times;
       samples=results{i}.report_points;
    end
end

if exist(path_to_lssvm, 'file') == 2
    aucs_inct=load(path_to_lssvm,'results');
    results=aucs_inct.results;
    for i=1:length(results)
       times_lssvm(i,:)=results{i}.selection_times;
       samples=results{i}.report_points;
    end
end

mean_time_incr=mean(times_incr);
mean_time_batch=mean(times_batch);
mean_time_lssvm=mean(times_lssvm);
stdev_time_inct=std(times_incr);
stdev_time_batch=std(times_batch);
stdev_time_lssvm=std(times_lssvm);

% fig=figure
% errorbar(samples,mean_time_incr,stdev_time_inct,'g-','LineWidth',5);hold on;
% errorbar(samples,mean_time_batch,stdev_time_batch,'b-','LineWidth',5);hold on;
% errorbar(samples,mean_time_lssvm,stdev_time_lssvm,'r-','LineWidth',5);hold on;
% legendInfo{1} = ['random'];
% legendInfo{2} = ['incremental'];
% legendInfo{3} = ['lssvm'];
% xlabel('#observations','FontSize',20)
% ylabel('Runtime (seconds)','FontSize',20)
% legend(legendInfo);
% %set(gca,'yscale','log')
% set(gca,'FontSize',20)
% xlim([0 samples(length(samples))])
% b = get(gca,'Children');
% h=[b];
% lgd = legend(h,'lssvm','batch','incremental')
% lgd.FontSize = 30;
% lgd.Location = 'northwest';

end


function [mean_time_incr,mean_time_batch,mean_time_lssvm,stdev_time_inct,stdev_time_batch,stdev_time_lssvm,samples]=get_processing_times_observations(path_to_results)
report_points=[];
times_incr=[];
times_batch=[];
samples=[];
path_to_incr=sprintf('%s/incr/results.mat',path_to_results);
path_to_batch=sprintf('%s/batch/results.mat',path_to_results);
path_to_lssvm=sprintf('%s/lssvm/results.mat',path_to_results);


counter=1;

if exist(path_to_incr, 'file') == 2
    aucs_inct=load(path_to_incr,'results');
    results=aucs_inct.results;
    for i=1:length(results)
       times_incr(i,:)=results{i}.processing_times;
       samples=results{i}.report_points;
    end
end

if exist(path_to_batch, 'file') == 2
    aucs_inct=load(path_to_batch,'results');
    results=aucs_inct.results;
    for i=1:length(results)
       times_batch(i,:)=results{i}.processing_times;
       samples=results{i}.report_points;
    end
end

if exist(path_to_lssvm, 'file') == 2
    aucs_inct=load(path_to_lssvm,'results');
    results=aucs_inct.results;
    for i=1:length(results)
       times_lssvm(i,:)=results{i}.processing_times;
       samples=results{i}.report_points;
    end
end

mean_time_incr=mean(times_incr);
mean_time_batch=mean(times_batch);
mean_time_lssvm=mean(times_lssvm);
stdev_time_inct=std(times_incr);
stdev_time_batch=std(times_batch);
stdev_time_lssvm=std(times_lssvm);

% fig=figure
% errorbar(samples,mean_time_incr,stdev_time_inct,'g-','LineWidth',5);hold on;
% errorbar(samples,mean_time_batch,stdev_time_batch,'b-','LineWidth',5);hold on;
% errorbar(samples,mean_time_lssvm,stdev_time_lssvm,'r-','LineWidth',5);hold on;
% legendInfo{1} = ['random'];
% legendInfo{2} = ['incremental'];
% legendInfo{3} = ['lssvm'];
% xlabel('#observations','FontSize',20)
% ylabel('Runtime (seconds)','FontSize',20)
% legend(legendInfo);
% %set(gca,'yscale','log')
% set(gca,'FontSize',20)
% xlim([0 samples(length(samples))])
% b = get(gca,'Children');
% h=[b];
% lgd = legend(h,'lssvm','batch','incremental')
% lgd.FontSize = 30;
% lgd.Location = 'northwest';

end


