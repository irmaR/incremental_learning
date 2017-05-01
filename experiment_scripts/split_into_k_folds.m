function [folds]=split_into_k_folds(training_data,training_class,K)

% List out the category values in use.
categories = unique(training_class);

% Get the number of vectors belonging to each category.
vecsPerCat = getVecsPerCat(training_data, training_class', categories);
% Compute the fold sizes for each category.
foldSizes = computeFoldSizes(vecsPerCat, K);
% Randomly sort the vectors in X, then organize them by category.
[X_sorted, y_sorted] = randSortAndGroup(training_data, training_class, categories);

% For each round of cross-validation...
for roundNumber = 1 : K
% Select the vectors to use for training and cross validation.
[X_train, y_train, X_val, y_val] = getFoldVectors(X_sorted, y_sorted, categories, vecsPerCat, foldSizes, roundNumber);
fold=[];
fold.train=X_train;
fold.train_class=y_train;
fold.test=X_val;
fold.test_class=y_val;
folds{roundNumber}=fold;
end


end