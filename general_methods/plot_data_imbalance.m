function []=plot_data_imbalance(path_to_results,classes)
%plotting data imbalance
report_points=[];
aucs_incr=[];
aucs_batch=[];
aucs_lssvm=[];

results=[];

path_to_incr=sprintf('%s/incr/results.mat',path_to_results)
path_to_batch=sprintf('%s/batch/results.mat',path_to_results)

exist(path_to_incr, 'file')
counter=1;


if exist(path_to_incr, 'file') == 2
    results=load(path_to_incr,'results');
    results=results.results;
    num_samples=length(results{1}.selected_labels{1});
    avg_imbalance_incr=zeros(1,length(results{1}.report_points));
    report_points=results{1}.report_points;
    for i=1:length(results)
       for j=1:length(results{i}.report_points)
           avg_imbalance_incr(j)=avg_imbalance_incr(j)+get_imbalance(results{i}.selected_labels{j},classes);
       end
    end
    for i=1:length(avg_imbalance_incr)
      avg_imbalance_incr(i)=avg_imbalance_incr(i)/length(results);
    end
    labels{counter}='incr';
    results_imb{counter}=avg_imbalance_incr;
    

end

if exist(path_to_batch, 'file') == 2
    results=load(path_to_batch,'results');
    results=results.results;
    report_points=results{1}.report_points;
    num_samples=length(results{1}.selected_labels{1});
    avg_imbalance_batch=zeros(1,length(results{1}.report_points));
    for i=1:length(results)
       for j=1:length(results{i}.report_points)
           avg_imbalance_batch(j)=avg_imbalance_batch(j)+get_imbalance(results{i}.selected_labels{j},classes);
       end
    end
    for i=1:length(avg_imbalance_batch)
      avg_imbalance_batch(i)=avg_imbalance_batch(i)/length(results);
    end
    labels{counter}='batch';
    results_imb{counter}=avg_imbalance_batch;
end
results_imb
fig=figure(1)
counter
colorVec = hsv(counter)
hold on;
xlabel('#observations')
ylabel('Ratio Majority/Minority class')
for i=1:counter
      plot(report_points,results_imb{1,i},'LineWidth',2,'Color',colorVec(i,:))
      legendInfo{i} = [labels{i}];
      legend(legendInfo)
      ylim([1 num_samples/10])
end
hold off;
saveas(fig,sprintf('%s/data_imb.jpg',path_to_results))
end

function [imb]=get_imbalance(labels,classes)
  cl1=labels(classes(1));
  cl2=labels(classes(2));
  
  if cl1>=cl2
    imb=cl1/cl2;
  else
    imb=cl2/cl1;
  end
end