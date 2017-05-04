function []=plot_results(path_to_results)
report_points=[];
aucs_incr=[];
aucs_batch=[];
aucs_lssvm=[];

results=[];

path_to_incr=sprintf('%s/incr/auc.mat',path_to_results);
path_to_batch=sprintf('%s/batch/auc.mat',path_to_results);
path_to_rnd=sprintf('%s/rnd/auc.mat',path_to_results);

path_to_lssvm=sprintf('%s/lssvm/auc.mat',path_to_results);

counter=1;

if exist(path_to_incr, 'file') == 2
    aucs_inct=load(path_to_incr,'avg_aucs')
    avg_runtime_inct=load(path_to_incr,'avg_runtime')
    std_runtime_inct=load(path_to_incr,'std_runtime')
    report_points=load(path_to_incr,'report_points')
    results{counter}=aucs_inct;
    labels{counter}='incr';
    counter=counter+1;
    
end

if exist(path_to_batch, 'file') == 2
    aucs_batch=load(path_to_batch,'avg_aucs')
    avg_runtime_batch=load(path_to_batch,'avg_runtime')
    std_runtime_batch=load(path_to_batch,'std_runtime')
    report_points=load(path_to_batch,'report_points')
    labels{counter}='batch';
    results{counter}=aucs_batch;
    counter=counter+1;
end

if exist(path_to_lssvm, 'file') == 2
    aucs_lssvm=load(path_to_lssvm,'avg_aucs') 
    avg_runtime_lssvm=load(path_to_lssvm,'avg_runtime')
    std_runtime_lssvm=load(path_to_lssvm,'std_runtime')
    labels{counter}='lssvm';
    report_points=load(path_to_lssvm,'report_points')
    results{counter}=aucs_lssvm;
end

if exist(path_to_rnd, 'file') == 2
    aucs_rnd=load(path_to_rnd,'avg_aucs') 
    avg_runtime_rnd=load(path_to_rnd,'avg_runtime')
    std_runtime_rnd=load(path_to_rnd,'std_runtime')
    labels{counter}='rnd';
    report_points=load(path_to_rnd,'report_points')
    results{counter}=aucs_rnd;
    counter=counter+1;
end

counter=counter-1;
fig = figure('visible', 'off');
colorVec = hsv(counter);
hold on;
xlabel('#observations')
ylabel('AUC-ROC')
for i=1:counter
      plot(report_points.report_points,results{1,i}.avg_aucs,'LineWidth',2,'Color',colorVec(i,:))
      legendInfo{i} = [labels{i}];
      legend(legendInfo);
      ylim([0 1]);
end
hold off;
if exist(sprintf('%s/aucs.jpg',path_to_results), 'file')==2
  delete(sprintf('%s/aucs.jpg',path_to_results));
end
saveas(fig,sprintf('%s/aucs.jpg',path_to_results))
close(fig)
clear fig
end
