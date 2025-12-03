clear;

load cifar-10-data.mat;

data = double(data);  
studentID = 38862905;  
rng(studentID, "twister");  

%random selection of 3 classes 
numClasses = 10;
randomClasses = randperm(numClasses, 3);  
classes = sort(randomClasses)';  

%debug
%disp('Selected Classes:');
%disp(classes);

%10 is mislabeled so make it 0
adjustedClasses = classes;
adjustedClasses(adjustedClasses == 10) = 0;

selectedIndices = ismember(labels, adjustedClasses);
dataSelected = data(selectedIndices, :, :, :);
labelsSelected = labels(selectedIndices);

% debugging: check size of data and labels
%disp(['Size of dataSelected: ', num2str(size(dataSelected))]);
%disp(['Size of labelsSelected: ', num2str(size(labelsSelected))]);

% debugging: check number of samples per class
%classCounts = histcounts(labels, numClasses);  
%disp('Counts of each class in the original dataset:');
%disp(classCounts);  


% split data 50-50
rng(studentID, "twister"); 
numImages = size(dataSelected, 1);
trainingIndex = randperm(numImages, numImages / 2); 

% seperate training data and labels
trainData = dataSelected(trainingIndex, :, :, :);
trainLabels = labelsSelected(trainingIndex);

% seperate testing data and labels
testingIndex = setdiff(1:numImages, trainingIndex);
testData = dataSelected(testingIndex, :, :, :);
testLabels = labelsSelected(testingIndex);

% reshape data
trainData = reshape(trainData, [size(trainData, 1), 32 * 32 * 3]);
testData = reshape(testData, [size(testData, 1), 32 * 32 * 3]);

% convert labels to categorical
trainLabels = categorical(trainLabels);
testLabels = categorical(testLabels);

% placeholders for predictions
tepredictEuclidean = categorical.empty(size(testData, 1), 0);
tepredictCosine = categorical.empty(size(testData, 1), 0);

k = 5;

%% Euclidean Distance 

tic;
for i = 1:size(testData, 1)
    comp1 = trainData;
    comp2 = repmat(testData(i, :), [size(trainData, 1), 1]);
    
    l2 = sum((comp1 - comp2) .^ 2, 2);
    
    [~, ind] = sort(l2);
    ind = ind(1:k);
    
    % labels for the nearest neighbors and predict the label
    labs = trainLabels(ind);
    tepredictEuclidean(i, 1) = mode(labs); 
end
euclideanTime = toc;

%% Cosine Distance 
% normalise data 
trainData = trainData ./ vecnorm(trainData, 2, 2);
testData = testData ./ vecnorm(testData, 2, 2);


tic;
for i = 1:size(testData, 1)
    % cosine similarity
    cosineSimilarities = (trainData * testData(i, :)') ./ ...
                         (sqrt(sum(trainData .^ 2, 2)) * norm(testData(i, :)));

    cosineDistance = 1 - cosineSimilarities;  % similarity to distance
    
    [~, ind] = sort(cosineDistance);
    ind = ind(1:k);
    
    % labels for nearest neighbors and predict the label
    labs = trainLabels(ind);
    tepredictCosine(i, 1) = mode(labs);
end
cosineTime = toc;

% calculate accuracies 
correctPredictionsEuclidean = sum(testLabels == tepredictEuclidean);
accuracyEuclidean = correctPredictionsEuclidean / size(testLabels, 1);

correctPredictionsCosine = sum(testLabels == tepredictCosine);
accuracyCosine = correctPredictionsCosine / size(testLabels, 1);


% Use your student ID as a seed for reproducibility
studentID = 38862905;
rng(studentID, 'twister'); 

%% SVM

tic;

% train SVM
svmModel = fitcecoc(trainData, trainLabels, 'Coding', 'onevsall', 'Learners', 'linear');

SVMTimetaken = toc;

tic;
svmPredictions = predict(svmModel, testData);
SVMPredictiontime = toc;

% SVM accuracy
SVMAccuracy = mean(svmPredictions == testLabels) * 100;  

SVMConfusionmatrix = confusionmat(testLabels, svmPredictions);

fprintf('SVM Accuracy: %.2f%%\n', SVMAccuracy);
disp('SVM Confusion Matrix:');
disp(SVMConfusionmatrix);

%% Decision Tree
tic;

% train Decision Tree 
treeModel = fitctree(trainData, trainLabels);

decisiontreeTimetaken = toc;

tic;
treePredictions = predict(treeModel, testData);
decisiontreePredictiontime = toc;

% Decision Tree accuracy
decisiontreeAccuracy = mean(treePredictions == testLabels) * 100;  % Accuracy in percentage

% confusion matrix for Decision Tree
decisiontreeConfusionmatrix = confusionmat(testLabels, treePredictions);

euclideanConfusionmatrix = confusionmat(testLabels, tepredictEuclidean);

cosineConfusionmatrix = confusionmat(testLabels, tepredictCosine);


% display results for Decision Tree
fprintf('Decision Tree Accuracy: %.2f%%\n', decisiontreeAccuracy);
disp('Decision Tree Confusion Matrix:');
disp(decisiontreeConfusionmatrix);

fprintf('Euclidean K-NN Accuracy: %.2f%%\n', accuracyEuclidean * 100);
fprintf('Cosine K-NN Accuracy: %.2f%%\n', accuracyCosine * 100);

%% Save Results
save('cw1.mat', 'SVMTimetaken', 'SVMPredictiontime', 'SVMAccuracy', ...
     'SVMConfusionmatrix', 'decisiontreeTimetaken', 'decisiontreePredictiontime', ...
     'decisiontreeAccuracy', 'decisiontreeConfusionmatrix','tepredictEuclidean', 'tepredictCosine', 'euclideanTime', 'cosineTime', 'classes', 'trainingIndex');



% Save results (including selected classes and training index)
%save('knn_results.mat', 'tepredictEuclidean', 'tepredictCosine', 'euclideanTime', 'cosineTime', 'classes', 'trainingIndex');

% Plot Confusion Matrices
figure(1);
confusionchart(testLabels, tepredictEuclidean);
title("Confusion Matrix - Euclidean K-NN");
saveas(gcf, 'Confusion_Matrix_Euclidean.png');

figure(2);
confusionchart(testLabels, tepredictCosine);
title("Confusion Matrix - Cosine K-NN");
saveas(gcf, 'Confusion_Matrix_Cosine.png');


figure(3);
confusionchart(testLabels, treePredictions);
title("confusion Matrix - Decision Tree");
saveas(gcf, 'Confusion_Matrix_DT.png');


figure(4);
confusionchart(testLabels, svmPredictions);
title("Confusion Matrix - SVM")
saveas(gcf, 'Confusion_Matrix_SVM.png');


% Display Sample Images with Labels
figure(5);
numSamples = 4; 
sampleI = randperm(size(dataSelected, 1), numSamples);

for i = 1:numSamples
    subplot(1, numSamples, i);  
    img = squeeze(dataSelected(sampleI(i), :, :, :)); 
    imagesc(img / 255);
    classIdx = double(labels(sampleI(i)));
    if classIdx == 0
        classIdx = 10;
    end
    title(label_names{classIdx});    
    axis off;  
end

% save figure as  PNG file
saveas(gcf, 'sample_images.png');