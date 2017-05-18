function []=plot_all_random_incremental_comparisons()
samples1=[5,10,20,40,60,80,100]
samples2=[10,20,40,60,80,100];
[random1,incr12,incr13,incr14]=plot_function_samples_1('/Users/irma/Documents/MATLAB/RESULTS/Incremental_May/Incremental/UCI_Adult/Balanced_Version1/',100,20,samples1,'incr');
[random3,incr32,incr33,incr34]=plot_function_samples_1('/Users/irma/Documents/MATLAB/RESULTS/Incremental_May/Incremental/RCV/Balanced_Modified/',100,40,samples1,'incr_bal');
[random2,incr22,incr23,incr24]=plot_function_samples_1('/Users/irma/Documents/MATLAB/RESULTS/Incremental_May/Incremental/USPS/Balanced_Modified/Balanced_Modified/',100,60,samples2,'incr_bal')
fig=figure(1)
%fig = gcf;
%fig.PaperPositionMode = 'auto'
%fig_pos = fig.PaperPosition;
%fig.PaperSize = [fig_pos(3) fig_pos(4)];

subplot(1,3,1)
hold on

plot(samples1,random1,'g+-','LineWidth',5);hold on;
plot(samples1,incr12,'kd','LineWidth',5)
plot(samples1,incr13,'cs','LineWidth',5)
plot(samples1,incr14,'mo','LineWidth',5)
set(gca,'FontSize',20)
set(gca,'LooseInset',get(gca,'TightInset'))
xlabel('#selected samples','FontSize',25)
ylabel('AUC','FontSize',25)
ylim([0 1])
xlim([samples1(1) samples1(length(samples1))]);
title('UCI Adult')

a = get(gca,'Children');
hold off


subplot(1,3,2)
hold on

plot(samples2,random2,'g+-','LineWidth',5);hold on;
plot(samples2,incr22,'kd','LineWidth',5)
plot(samples2,incr23,'cs','LineWidth',5)
plot(samples2,incr24,'mo','LineWidth',5)
set(gca,'FontSize',20)
set(gca,'LooseInset',get(gca,'TightInset'))
xlabel('#selected samples','FontSize',25)
title('USPS')
xlim([samples1(1) samples1(length(samples1))]);
ylim([0 1])
b = get(gca,'Children');
hold off

subplot(1,3,3)
hold on
plot(samples1,random3,'g+-','LineWidth',5);hold on;
plot(samples1,incr32,'kd','LineWidth',5)
plot(samples1,incr33,'cs','LineWidth',5)
plot(samples1,incr34,'mo','LineWidth',5)
set(gca,'FontSize',20)
set(gca,'LooseInset',get(gca,'TightInset'))
xlim([samples2(1) samples2(length(samples2))]);
ylim([0 1])
title('RCV')
xlabel('#selected samples','FontSize',25)
b = get(gca,'Children');
hold off;

h=[b];
lgd = legend(h,'incr @T3','incr @T2','incr @T1','random')
lgd.FontSize = 30;
lgd.Location = 'southeast';

saveas(fig,sprintf('/Users/irma/Documents/MATLAB/runtime.jpg'))
end

function [random_points,incr_bal_point1,incr_bal_point3,incr_bal_point4]=plot_function_samples_1(path_to_results,bs,division,samples,incremental)
length(samples)
samples
for i=1:length(samples)
    %path_to_incr=sprintf('%s/smp_%d/bs_%d/Supervised/HeatKernel/k_0/incr_bal/auc.mat',path_to_results,samples(i),bs);
    path_to_incr=sprintf('%s/smp_%d/bs_%d/Supervised/HeatKernel/k_0/%s/auc.mat',path_to_results,samples(i),bs,incremental);

    path_to_random=sprintf('%s/smp_%d/bs_%d/Supervised/HeatKernel/k_0/rnd/auc.mat',path_to_results,samples(i),bs);
    report_points=load(path_to_incr,'report_points')
    nr_report_points=length(report_points.report_points)
    
    if exist(path_to_incr, 'file') == 2
    avg_auc_inct=load(path_to_incr,'avg_aucs');
    std_auc_inct=load(path_to_incr,'stdev');
    avg_aucs_inct{i}=avg_auc_inct.avg_aucs;
    std_aucs_inct{i}=std_auc_inct.stdev;  
    
    end

    if exist(path_to_random, 'file') == 2
    avg_auc_rnd=load(path_to_random,'avg_aucs');
    std_auc_rnd=load(path_to_random,'stdev');
    avg_aucs_rnd{i}=avg_auc_rnd.avg_aucs;
    std_aucs_rnd{i}=std_auc_rnd.stdev;
    report_points=load(path_to_incr,'report_points');          
    else
        fprintf('Doesnt exist: %s',path_to_random)
    end
end
random_points=[];
incr_bal_point1=[];
incr_bal_point2=[];
incr_bal_point3=[]
incr_bal_point4=[];
for i=1:length(samples)
    tmp=avg_aucs_rnd{i};
    random_points(i)=nanmean(tmp);
end
jump=floor(nr_report_points/division)
for i=1:length(samples)
    tmp=avg_aucs_inct{i};
    incr_bal_point1(i)=nanmean(tmp(2:2));
    incr_bal_point2(i)=nanmean(tmp(0+3*jump));
    incr_bal_point3(i)=nanmean(tmp(0+4*jump));
    incr_bal_point4(i)=nanmean(tmp(nr_report_points));
end

end