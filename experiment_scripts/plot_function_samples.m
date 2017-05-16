function []=plot_function_samples(path_to_results,title_text)
samples=[20,40,60,80,100];
bs=100;
%avg_aucs_inct=[];
%std_aucs_inct=[];

%avg_aucs_rnd=[];
%std_aucs_rnd=[];

%avg_runtime_inct=[];
%std_runtime_inct=[];

%avg_runtime_rnd=[];
%std_runtime_rnd=[];

%avg_aucs_inct=zeros(length(samples),length(report_points.report_points));
%avg_aucs_rnd=zeros(length(samples),length(report_points.report_points));
%std_aucs_inct=zeros(length(samples),length(report_points.report_points));
%std_aucs_rnd=zeros(length(samples),length(report_points.report_points));



for i=1:length(samples)
    path_to_incr=sprintf('%s/smp_%d/bs_%d/Supervised/HeatKernel/k_0/incr_bal/auc.mat',path_to_results,samples(i),bs);
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
jump=floor(nr_report_points/80)
for i=1:length(samples)
    tmp=avg_aucs_inct{i}
    incr_bal_point1(i)=nanmean(tmp(1:1));
    incr_bal_point2(i)=nanmean(tmp(0+3*jump));
    incr_bal_point3(i)=nanmean(tmp(0+4*jump));
    incr_bal_point4(i)=nanmean(tmp(nr_report_points));
end

fig=figure
plot(samples,random_points,'m+-','LineWidth',5);hold on;
plot(samples,incr_bal_point2,'ro','LineWidth',5);hold on;
plot(samples,incr_bal_point3,'go','LineWidth',5);hold on;
plot(samples,incr_bal_point4,'bo','LineWidth',5);hold on;
legendInfo{1} = ['random'];
legendInfo{2} = ['incremental'];
legend(legendInfo);
%ylim([0 1]);
% avg_runtimes_inct
% avg_runtimes_batch
% fig = figure('visible', 'off');
% %fig = figure
% colorVec = hsv(3);
% hold on;
% xlabel('#selected samples','FontSize',20)
% ylabel('Runtime (seconds)','FontSize',20)
% errorbar(samples,avg_runtimes_inct,std_runtimes_inct,'LineWidth',5,'Color',colorVec(1,:))
% errorbar(samples,avg_runtimes_batch,std_runtimes_batch,'LineWidth',5,'Color',colorVec(2,:))
% set(gca,'yscale','log')
% set(gca,'FontSize',20)
% title(title_text)
% xlim([samples(1) samples(length(samples))]);
% legendInfo{1} = ['incr'];
% legendInfo{2} = ['batch'];
% legend(legendInfo);
% hold off;

end