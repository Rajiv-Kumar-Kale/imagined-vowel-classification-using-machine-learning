function [performance] = KFoldCrossValidaion(k_fold, X ,y)

% performance structre
performance = struct('classifier',[],'accuracy',[],'std',[],'precision',[],'recall',[],'f1Score',[]);


CVO = cvpartition(y,'k',k_fold);    % CV partition

% accuracy , precision and recall for each fold
% three rows are of 3 models and column represent data over each fold
acc = zeros(3,CVO.NumTestSets);
pre = zeros(3,CVO.NumTestSets);
recall = zeros(3,CVO.NumTestSets,1);


% Iterating over each fold
for i= 1:CVO.NumTestSets
    % getting test and training indices
    trIdx = CVO.training(i);
    teIdx = CVO.test(i);
    
    % dividing data into test(validation) and train for each iteration
    X_train =  X(trIdx,:);    y_train = y(trIdx);
    X_val = X(teIdx,:);       y_val = y(teIdx,:);
    
    %LINEAR SVM MODEL
    model_linear_svm = fitcsvm(X_train,y_train,'KernelFunction','linear','Standardize','on');
    
    %MEDIUM GAUSSIAN SVM MODEL
    model_gaussian_svm = fitcsvm(X_train,y_train,'KernelFunction','gaussian','KernelScale',47,'BoxConstraint',1,'Standardize','on');
    
    %ENSEMBLE BAGGED TREE
    %template = templateTree('MaxNumSplits',95);
    %model_ensemble_bagged_tree = fitcensemble(X_train,y_train,'Method','Bag','NumLearningCycle',30,'Learners',template);
    model_ensemble_bagged_tree = fitensemble(X_train,y_train, 'Bag', 200, 'Tree','Type', 'Classification');
    
    
    % predictions of the three models
    predictions_1 = predict( model_linear_svm,X_val);
    predictions_2 = predict(model_gaussian_svm,X_val);
    predictions_3  = predict(model_ensemble_bagged_tree,X_val);
    
    % Confusion matrix
    [C1,order] = confusionmat(y_val,predictions_1);
    [C2,order] = confusionmat(y_val,predictions_2);
    [C3,order] = confusionmat(y_val,predictions_3);
    
    % accuracy calculation from confusion matrix
    acc(1,i) = (C1(1,1)+C1(2,2))/(sum(sum(C1,1)))*100;
    acc(2,i) = (C2(1,1)+C2(2,2))/(sum(sum(C2,1)))*100;
    acc(3,i) = (C3(1,1)+C3(2,2))/(sum(sum(C3,1)))*100;
    
    % precision(PPV) calculation from confusion matrix
    pre(1,i) = (C1(1,1)/(C1(1,1)+C1(2,1)))*100;
    pre(2,i) = (C2(1,1)/(C2(1,1)+C2(2,1)))*100;
    pre(3,i) = (C3(1,1)/(C3(1,1)+C3(2,1)))*100;
    
    % recall(TPR) calculation from confusion matrix
    recall(1,i) = (C1(1,1)/(C1(1,1)+C1(1,2)))*100;
    recall(2,i) = (C2(1,1)/(C2(1,1)+C2(1,2)))*100;
    recall(3,i) = (C3(1,1)/(C3(1,1)+C3(1,2)))*100;
    
end

% Assigning values to performance matrix


% 1st classifier performance
performance(1).classifier = 'Linear_SVM';
performance(1).accuracy = mean(acc(1,:));
performance(1).std = std(acc(1,:));
performance(1).precision = nanmean(pre(1,:));
performance(1).recall = mean(recall(1,:));
performance(1).f1Score = (2*performance(1).precision*performance(1).recall)/(performance(1).precision + performance(1).recall);


% 2nd classifier performance
performance(2).classifier = 'Gaussian_SVM';
performance(2).accuracy = mean(acc(2,:));
performance(2).std = std(acc(2,:));
performance(2).precision = nanmean(pre(2,:));
performance(2).recall = mean(recall(2,:));
performance(2).f1Score = (2*performance(2).precision*performance(2).recall)/(performance(2).precision + performance(2).recall);


% 3rd classifier performance
performance(3).classifier = 'Ensemble_Bagged_SVM';
performance(3).accuracy = mean(acc(3,:));
performance(3).std = std(acc(3,:));
performance(3).precision = nanmean(pre(3,:));
performance(3).recall = mean(recall(3,:));
performance(3).f1Score = (2*performance(3).precision*performance(3).recall)/(performance(3).precision + performance(3).recall);


% display('prcision')
% pre
% end of function
end

