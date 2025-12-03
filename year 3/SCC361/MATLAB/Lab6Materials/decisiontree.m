
% ID3 Decision Tree Algorithm


function[] = decisiontree(inputFileName, trainingSetSize, numberOfTrials)
% DECISIONTREE Create a decision tree by following the ID3 algorithm
% args:
%   inputFileName       - the fully specified path to input file
%   trainingSetSize     - integer specifying number of examples from input
%                         used to train the dataset
%   numberOfTrials      - integer specifying how many times decision tree
%                         will be built from a randomly selected subset
%                         of the training examples

%% Read dataset file 
% Read in the specified text file contain the examples
fid = fopen(inputFileName, 'rt');
dataInput = textscan(fid, '%s');
% Close the file
fclose(fid);

% Reformat the data into attribute array and data matrix of 1s and 0s for
% true or false
i = 1;
% First store the attributes into a cell array
while (~strcmp(dataInput{1}{i}, 'CLASS'))
    i = i + 1;
end
attributes = cell(1,i);
for j=1:i
    attributes{j} = dataInput{1}{j};
end

% NOTE: The classification will be the final attribute in the data rows
% below
numAttributes = i;
numInstances = (length(dataInput{1}) - numAttributes) / numAttributes;
% Then store the data into matrix
data = zeros(numInstances, numAttributes);
i = i + 1;
for j=1:numInstances
    for k=1:numAttributes
        data(j, k) = strcmp(dataInput{1}{i}, 'true');
        i = i + 1;
    end
end

%% Here is where the trials start
for i=1:numberOfTrials
    
    % Print the trial number
    fprintf('TRIAL NUMBER: %d\n\n', i);
    
    % Split data into training and validation sets randomly
    % Use randsample to get a vector of row numbers for the training set
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    rng(10);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    rows = sort(randsample(numInstances, trainingSetSize));
    % Initialize two new matrices, training set and validation set
    trainingSet = zeros(trainingSetSize, numAttributes);
    validationSetSize = (numInstances - trainingSetSize);
    validationSet = zeros(validationSetSize, numAttributes);
    % Loop through data matrix, copying relevant rows to each matrix
    training_index = 1; %% << here is when the data splitting starts
    validation_index = 1;
    for data_index=1:numInstances
        if (rows(training_index) == data_index)
            trainingSet(training_index, :) = data(data_index, :);
            if (training_index < trainingSetSize)
                training_index = training_index + 1;
            end
        else
            validationSet(validation_index, :) = data(data_index, :);
            if (validation_index < validationSetSize)
                validation_index = validation_index + 1;
            end
        end
    end

    % Construct a decision tree on the training set using the ID3 algorithm
    activeAttributes = ones(1, length(attributes) - 1);
    new_attributes = attributes(1:length(attributes)-1);
    tree = ID3(trainingSet, attributes, activeAttributes); %% ID3 function is the one that we learned from Week 8 Lecture
    
    % Print out the tree
    fprintf('DECISION TREE STRUCTURE:\n');
    PrintTree(tree, 'root');
    
    % Run tree against validation set
    % The second column is for actual classification, first for calculated
    ID3_Classifications = zeros(validationSetSize,2);
    ID3_numCorrect = 0; 
    for k=1:validationSetSize %over the validation set
        % Call a recursive function to follow the tree nodes and classify
        ID3_Classifications(k,:) = ClassifyByTree(tree, new_attributes, validationSet(k,:));
        
        if (ID3_Classifications(k,1) == ID3_Classifications(k, 2)) %correct
            ID3_numCorrect = ID3_numCorrect + 1;
        end
    end
    
    % Calculate the proportions correct and print out
    if (validationSetSize)
        ID3_Percentage = round(100 * ID3_numCorrect / validationSetSize);
    else
        ID3_Percentage = 0;
    end
    ID3_Percentages(i) = ID3_Percentage;
    
    fprintf('\t Percent of validation cases correctly classified by an ID3 decision tree = %d\n', ID3_Percentage);
end
 %% Reporting average accuracy on validation dataset
 meanID3 = round(mean(ID3_Percentages));
 
 % Print out remaining details
 fprintf('example file used = %s\n', inputFileName);
 fprintf('number of trials = %d\n', numberOfTrials);
 fprintf('training set size for each trial = %d\n', trainingSetSize);
 fprintf('validation set size for each trial = %d\n', validationSetSize);
 fprintf('mean performance (percentage correct) of decision tree over all trials = %d\n', meanID3);
end