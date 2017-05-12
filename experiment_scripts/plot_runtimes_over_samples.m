function []=plot_runtimes_over_samples(path_to_results,title_text)
samples=[20,40,60,80,100];
bs=100;
avg_runtimes_inct=[];
std_runtimes_inct=[];

avg_runtimes_batch=[];
std_runtimes_batch=[];

for i=1:length(samples)
    path_to_incr=sprintf('%s/smp_%d/bs_%d/Supervised/HeatKernel/k_0/incr/auc.mat',path_to_results,samples(i),bs)
    path_to_batch=sprintf('%s/smp_%d/bs_%d/Supervised/HeatKernel/k_0/batch/auc.mat',path_to_results,samples(i),bs)
    path_to_incr
  
    if exist(path_to_incr, 'file') == 2
    avg_runtime_inct=load(path_to_incr,'avg_runtime')
    std_runtime_inct=load(path_to_incr,'std_runtime')
    avg_runtimes_inct(i)=avg_runtime_inct.avg_runtime;
    std_runtimes_inct(i)=std_runtime_inct.std_runtime;    
    end

    if exist(path_to_batch, 'file') == 2
    avg_runtime_batch=load(path_to_batch,'avg_runtime')
    std_runtime_batch=load(path_to_batch,'std_runtime')
    avg_runtimes_batch(i)=avg_runtime_batch.avg_runtime;
    std_runtimes_batch(i)=std_runtime_batch.std_runtime;
    else
       avg_runtimes_batch(i)=NaN;
       std_runtimes_batch(i)=NaN;
    end
        
end
avg_runtimes_inct
avg_runtimes_batch
%fig = figure('visible', 'off');
fig = figure
colorVec = hsv(2);
hold on;
xlabel('#selected samples','FontSize',18)
ylabel('Runtime (seconds)','FontSize',18)
errorbar(samples,avg_runtimes_inct,std_runtimes_inct,'LineWidth',3,'Color',colorVec(1,:))
errorbar(samples,avg_runtimes_batch,std_runtimes_batch,'LineWidth',3,'Color',colorVec(2,:))
set(gca,'yscale','log')
set(gca,'FontSize',18)
title(title_text)
xlim([samples(1) samples(length(samples))]);
legendInfo{1} = ['incr'];
legendInfo{2} = ['batch'];
legend(legendInfo);
hold off;

end