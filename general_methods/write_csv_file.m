function []=write_csv_file(path,samples,batches,report_points,results,experiment_name)
%[s, mess, messid] = mkdir(complete_path)
file_name_acc=sprintf('%s/accuracy_%d_%d.csv',path,samples,batches) 
file_name_F1_micro_score=sprintf('%s/microF_%d_%d.csv',path,samples,batches)
file_name_F1_macro_score=sprintf('%s/macroF_%d_%d.csv',path,samples,batches)
file_name_averages=sprintf('%s/averages_%d_%d.csv',path,samples,batches)


row_name=report_points';
T=table;
T1=table;
T2=table;


row_names={};
column_names={};
for i=1:length(results)
column_names{i}=sprintf('run%d%',i);
end
for i=1:length(report_points)
row_names{i}=sprintf('%d',report_points(i));
end
T.observations=row_names';
for k=1:length(column_names)
  T.(column_names{k})=results{k}.accuracies';
  T1.(column_names{k})=results{k}.F1_micro_scores';
  T2.(column_names{k})=results{k}.F1_macro_scores';
end
writetable(T,file_name_acc)
writetable(T1,file_name_F1_micro_score)
writetable(T2,file_name_F1_macro_score)

%calculate averages
T4=table;
avg_accuracies=zeros(1,length(report_points));
avg_F1_micro_scores=zeros(1,length(report_points));
avg_F1_macro_scores=zeros(1,length(report_points));
avg_runtimes=zeros(1,length(report_points));
for i=1:length(results)
avg_accuracies=avg_accuracies+results{i}.accuracies;
avg_F1_micro_scores=avg_F1_micro_scores+results{i}.F1_micro_scores;
avg_F1_macro_scores=avg_F1_macro_scores+results{i}.F1_macro_scores;
avg_runtimes=avg_runtimes+results{i}.runtimes;
end
avg_accuracies=avg_accuracies./length(results);
avg_F1_micro_scores=avg_F1_micro_scores./length(results);
avg_F1_macro_scores=avg_F1_macro_scores./length(results);
avg_runtimes=avg_runtimes./length(results);

column_names={'avg_Acc','avg_F1_micro','avg_F1_macro','avg_runtime'}
for i=1:length(report_points)
row_names{i}=sprintf('%d',report_points(i));
end
T4.observations=row_names';
T4.(column_names{1})=avg_accuracies';
T4.(column_names{2})=avg_F1_micro_scores';
T4.(column_names{3})=avg_F1_macro_scores';
T4.(column_names{4})=avg_runtimes';
writetable(T4,file_name_averages)   

end
