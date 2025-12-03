ID = readtable("iris_data_150samples.csv");
IDmissing = readtable("iris_missing_values.csv");
%unique(ID.Class2);

features = table2array(ID(:,1:4));
size(features);
labels = table2array(ID(:,6));
size(labels);
whos features labels;

%rem_rows_2 = rmmissing(IDmissing, "MinNumMissing", 2);
%rem_cols_2 = rmmissing(IDmissing, 2, "MinNumMissing", 2);

numCols = width(IDmissing);

dataExceptLast = IDmissing(:, 1:numCols-1);

% Step 3: Detect outliers using the 'mean' method
outliersMean = varfun(@(x) isoutlier(x, 'mean'), dataExceptLast);

% Step 4: Detect outliers using the 'median' method
outliersMedian = varfun(@(x) isoutlier(x, 'median'), dataExceptLast); 

columnMeans = varfun(@mean, dataExceptLast);  
columnMedians = varfun(@median, dataExceptLast);

disp('Outliers using mean:');
disp(outliersMean);

disp('Outliers using median:');
disp(outliersMedian);

disp('Difference between mean and median:');
diffMeanMedian = columnMeans{:, :} - columnMedians{:, :}; % Difference matrix
disp(diffMeanMedian);


newSamples = [6.3, 2.6, 4.1, 1.2;
              4.7, 3.5, 1.5, 0.3;
              7.1, 2.9, 5.5, 2.1];
newSamplesTable = array2table(newSamples, 'VariableNames', ID.Properties.VariableNames(1:4));


dataWithSamples = [ID(:,1:4); newSamplesTable];


%mFeatures = mean(features, "omitnan");
%featuresfilled = fillmissing(features,"constant", mFeatures); 

newLabels = repmat({'New'}, size(newSamples, 1), 1); % Assign 'New' as label for new samples
allLabels = [labels; newLabels];
    
figure(1);
gscatter(features(:,1), features(:,2));
title('Scatter plot: Iris Dataset');
xlabel('Measure 1');
ylabel('Measure 2'); 

figure(2);
gscatter([features(:,1); newSamples(:,1)], [features(:,2); newSamples(:,2)], allLabels, 'rgbm', 'o+x');
title('Scatter plot: Iris Dataset with new samples');
xlabel('Measure 1');
ylabel('Measure 2'); 
legend('Setosa', 'Versicolor', 'Virginica', 'New Samples');
