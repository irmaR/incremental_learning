function []=plot_results(path_to_results,number_of_samples,batch_size)
experiment_names={'incremental','batch'};
path_to_incremental=sprintf('%s/incremental/%d/averages_incremental_%d_%d.csv',path_to_results,number_of_samples,number_of_samples,batch_size)
path_to_batch=sprintf('%s/batch/%d/averages_batch_%d_%d.csv',path_to_results,number_of_samples,number_of_samples,batch_size)

%just display averages for now



if exist(path_to_incremental, 'file') == 2 && exist(path_to_batch, 'file') == 2
    incremental_average_results = readtable(path_to_incremental)
    batch_average_results = readtable(path_to_batch)
    x_axis_observations=incremental_average_results.observations
elseif exist(path_to_incremental, 'file') == 2 && exist(path_to_batch, 'file') == 0
    fprintf('Only incremental results exists....')
    incremental_average_results = readtable(path_to_incremental);
    x_axis_observations=incremental_average_results.observations';
    average_accuracy=incremental_average_results.avg_Acc';
    average_F1_micro=incremental_average_results.avg_F1_micro';
    average_F1_macro=incremental_average_results.avg_F1_macro';
    average_runtime=incremental_average_results.avg_runtime';
    
    %figure
    subplot(2,2,1)       % add first plot in 2 x 1 grid
    plot(x_axis_observations,average_accuracy,'-r','LineWidth',2)
    ylim([0 1])
    xlabel('#observed points')
    ylabel('Avg. Acc.')
    
    subplot(2,2,2)       % add first plot in 2 x 1 grid
    plot(x_axis_observations,average_F1_micro,'-r','LineWidth',2)
    ylim([0 1])
    xlabel('#observed points')
    ylabel('Avg. F1-micro')
    
    subplot(2,2,3)       % add first plot in 2 x 1 grid
    plot(x_axis_observations,average_F1_macro,'-r','LineWidth',2)
    ylim([0 1])
    xlabel('#observed points')
    ylabel('Avg. F1-macro')
   
    
    subplot(2,2,4)       % add first plot in 2 x 1 grid
    plot(x_axis_observations,average_runtime,'-r','LineWidth',2)
    %title('Avg. Runtime')
    xlabel('#observed points')
    ylabel('Avg. runtime (seconds)')
    legend('incremental')
    
    set(gcf,'NextPlot','add');
    axes;
    h = title(sprintf('Incremental app.: #samples=%d,#batch size=%d',number_of_samples,batch_size));
    set(gca,'Visible','off');
    set(h,'Visible','on');
    
    saveas(gcf,sprintf('%s/incremental/%d/averages_incremental_%d_%d.png',path_to_results,number_of_samples,number_of_samples,batch_size))
    
elseif exist(path_to_incremental, 'file') == 0 && exist(path_to_batch, 'file') == 2
end


end
