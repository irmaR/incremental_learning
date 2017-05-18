function []=plot_function_samples(path_to_results,title_text)
samples=[10,20,40,60,80,100];
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
    %path_to_incr=sprintf('%s/smp_%d/bs_%d/Supervised/HeatKernel/k_0/incr_bal/auc.mat',path_to_results,samples(i),bs);
    path_to_incr=sprintf('%s/smp_%d/bs_%d/Supervised/HeatKernel/k_0/incr_bal/auc.mat',path_to_results,samples(i),bs);
    path_to_batch=sprintf('%s/smp_%d/bs_%d/Supervised/HeatKernel/k_0/batch/auc.mat',path_to_results,samples(i),bs);
    path_to_lssvm=sprintf('%s/smp_%d/bs_%d/Supervised/HeatKernel/k_0/lssvm/auc.mat',path_to_results,samples(i),bs);
    report_points=load(path_to_incr,'report_points')
    nr_report_points=length(report_points.report_points)
    
    if exist(path_to_incr, 'file') == 2
    avg_auc_inct=load(path_to_incr,'avg_aucs');
    std_auc_inct=load(path_to_incr,'stdev');
    avg_aucs_inct{i}=avg_auc_inct.avg_aucs;
    std_aucs_inct{i}=std_auc_inct.stdev;  
    
    end
    
    if exist(path_to_batch, 'file') == 2
    avg_auc_batch=load(path_to_batch,'avg_aucs');
    std_auc_batch=load(path_to_batch,'stdev');
    avg_aucs_batch{i}=avg_auc_batch.avg_aucs;
    std_aucs_batch{i}=std_auc_batch.stdev;  
    
    end

    if exist(path_to_lssvm, 'file') == 2
    avg_auc_lssvm=load(path_to_lssvm,'avg_aucs');
    std_auc_lssvm=load(path_to_lssvm,'stdev');
    avg_aucs_lssvm{i}=avg_auc_lssvm.avg_aucs;
    std_aucs_lssvm{i}=std_auc_lssvm.stdev;
    report_points=load(path_to_lssvm,'report_points');          
    end
end
batch_points=[];
incr_points=[];
lssvm_points=[];

std_batch_points=[];
std_incr_points=[];
std_lssvm_points=[];

for i=1:length(samples)
    tmp=avg_aucs_batch{i};
    batch_points(i)=nanmean(tmp);
    std_batch_points(i)=std(tmp);
end


for i=1:length(samples)
    tmp=avg_aucs_inct{i};
    incr_points(i)=nanmean(tmp);
    std_incr_points(i)=std(tmp);
end


% for i=1:length(samples)
%     tmp=avg_aucs_lssvm{i};
%     lssvm_points(i)=nanmean(tmp);
%     std_lssvm_points(i)=std(tmp);
% end


fig=figure
%errorbar(samples,batch_points,std_batch_points,'b+-','LineWidth',5);hold on;
batch_points
incr_points
%lssvm_points
std_batch_points
std_incr_points
%std_lssvm_points
plot(samples,batch_points,'b+-','LineWidth',5);hold on;
plot(samples,incr_points,'m+-','LineWidth',5);hold on;
%plot(samples,lssvm_points,'r+-','LineWidth',5);hold on;


legendInfo{1} = ['random'];
legendInfo{2} = ['incremental'];
xlabel('#selected samples','FontSize',20)
ylabel('AUC','FontSize',20)
legend(legendInfo);
ylim([0 1]);
xlim([samples(1) samples(length(samples))])
%set(gca,'yscale','log')
set(gca,'FontSize',20)

b = get(gca,'Children');
h=[b];
lgd = legend(h,'lssvm','incremental','batch')
lgd.FontSize = 30;
lgd.Location = 'southeast';
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