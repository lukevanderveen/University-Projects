map=im2bw(imread('random_map.bmp'));

start = [1, 1];
finish = [500, 500];

populationSize = 200; 
n = 10; % number of path points
generations = 400;
crossover_prob = 0.8;
mutation_rate = 0.2;
population = round(rand(populationSize, 2 *n) * 499 + 1); %random number generation 

% Prompt user for selection, crossover, and mutation choices
disp('Selection Methods: 0 = RWS, 1 = Tournament, 2 = Rank-based');
selectionType = input('Enter selection method (0, 1, or 2): ');
if ~ismember(selectionType, [0, 1, 2]) % invalid result check
    error('Invalid selection method. Please choose 0, 1, or 2.');
end

disp('Crossover Methods: 0 = Single-point, 1 = Two-point');
crossoverType = input('Enter crossover method (0 or 1): ');
if ~ismember(crossoverType, [0, 1])% invalid result check
    error('Invalid crossover method. Please choose 0 or 1.');
end

disp('Mutation Methods: 0 = Gaussian Mutation, 1 = Random Resetting');
mutationType = input('Enter mutation method (0 or 1): ');
if ~ismember(mutationType, [0, 1])% invalid result check
    error('Invalid mutation method. Please choose 0 or 1.');
end

tic; %start timer 

bestFitness = inf;
bestPath = [];


%figure;
%imshow(map);
%hold on;

%GA loop
for gen = 1:generations
    fitness = arrayfun(@(i) calculateFitness(population(i, :), map, start, finish), 1:populationSize);

    [minFitness, minIdx] = min(fitness);
    if minFitness < bestFitness
        bestFitness = minFitness;
        bestPath = population(minIdx, :);
    end

    newPopulation = []; 
    for i = 1:populationSize / 2
        parent1 = Selection(population, fitness, selectionType);
        parent2 = Selection(population, fitness, selectionType);

        %%crossover
        if crossoverType == 0
            offspring1 = round(kPointCrossover(parent1, parent2, crossover_prob, 2));
            offspring2 = round(kPointCrossover(parent2, parent1, crossover_prob, 2));
        elseif crossoverType == 1
            offspring1 = round(uniformCrossover(parent1, parent2, crossover_prob));
            offspring2 = round(uniformCrossover(parent2, parent1, crossover_prob));
        else
            error('incorrect crossover method')
        end

        %%mutation
        offspring1 = round(mutation(offspring1, mutation_rate, mutationType));
        offspring2 = round(mutation(offspring2, mutation_rate, mutationType));

        newPopulation = [newPopulation; offspring1; offspring2];
    end
    population = newPopulation;

    disp(['Generation ' num2str(gen) ', Best Fitness: ' num2str(bestFitness)]);


    %%display path to watch each change in path
    %pathPoints = [start; reshape(bestPath, [], 2); finish];
    %plot(pathPoints(:, 2), pathPoints(:, 1), 'r-', 'LineWidth', 1);
    %pause(0.005);

    %if gen < generations
     %   cla;
      %  imshow(map);
       % hold on;
    %end
end

% End timer and display execution time
totalTime = toc;
disp(['Execution Time: ' num2str(totalTime) ' seconds']);

% Display the final best path
figure;
path = [start; reshape(bestPath, [], 2); finish]; % Correctly reshaped without additional scaling
clf;
imshow(map);
hold on;
rectangle('position', [1, 1, size(map) - 1], 'edgecolor', 'k'); % Add border
line(path(:, 2), path(:, 1), 'LineWidth', 1, 'Color', 'b'); % Final path in blue
hold off;

%%% FUNCTIONS %%%

function fitness = calculateFitness(path, map, start, finish)
    penalty = 0;
    maxLineLength = 50; %create a maximum line length
    maxLinePenalty = 150;
    obstaclePenaltyFactor = 500;

    %list of path points from start to finish (the reshape changes a 1D
    %array to 2D pair of cords
    pathPoints = [start; reshape(path, [], 2); finish];     
    diffs = diff(pathPoints);
    distances = sqrt(sum(diffs.^2, 2));

    for i = 1:size(pathPoints, 1) -1
       %extract a segment of the path (segment is a pair of cords)
       segment = [pathPoints(i, :); pathPoints(i+1, :)]; 
       segmentLength = sqrt(sum(diff(segment).^2));
       %euclidean distance between segment
       if segmentLength > maxLineLength
            penalty = penalty + maxLinePenalty * (segmentLength - maxLineLength);
        end

       %%check for obstacals 
       %generate x (100) evenly spaced points along segment to check for
       %obstical, then round to int
       lineX = round(linspace(segment(1,1), segment(2,1), 150));
       lineY = round(linspace(segment(1,2), segment(2,2), 150));
       lineX = max(1, min(size(map, 1), lineX));
       lineY = max(1, min(size(map, 1), lineY));
       indices = sub2ind(size(map), lineY, lineX);
      
       obstacleCount = sum(map(indices) == 0); %total obsticals touched
        if obstacleCount > 0
            % Apply additional penalty for touching obstacles
            penalty = penalty + obstaclePenaltyFactor * obstacleCount;
        end
    end
    fitness = sum(distances) + 10 * penalty;
end

function selected = Selection(population, fitness, method)
    switch method
        case  0 %RouletteWheelSelection
            %calculate selection based on probabilities inversely
            %proportional to fitness
            probabilities = 1 ./ fitness;
            probabilities = probabilities / sum(probabilities);
            cumuProbability = cumsum(probabilities);

            %select individuals based on cumulative probability
            selected = population(find(rand <= cumuProbability, 1), :); 
        case 1 % Tournament Selection
            %randomly select k individuals from population
            k = 5;
            indices = randi(size(population, 1), [1, k]);
             [~, idx] = sort(fitness(indices));
            selectedIdx = indices(idx(1)); % select best individual
            % Probability to choose second-best for diversity
            if rand < 0.2 
                selectedIdx = indices(idx(2));
            end
            selected = population(selectedIdx, :);
        case 2 % Rank-Based election
            %sort individuals based on fitness value
            [~, sortedIdx] = sort(fitness);
            ranks = 1:length(sortedIdx); % assign ranks
            probabilities = ranks / sum(ranks);
            cumuProbability = cumsum(probabilities);
            selected = population(find(rand <= cumuProbability, 1), :);
        otherwise 
            error('invalid selection option');
    end
end

function offspring = kPointCrossover(parent1, parent2, probability, k)
    if rand < probability % crossover probability
        %randomly select k points to crossover
        points = sort(randi([1, length(parent1)], 1, k)); 
        offspring = parent1;
        toggle = true; % toggle between parents
        for i=1:length(points)
            if toggle
                offspring(points(i):end) = parent2(points(i):end);
            else
                offspring(points(i):end) = parent1(points(i):end);
            end
            toggle = ~toggle;
        end
    else
        %if no corssover just make offspring parent 1
        offspring = parent1;
    end
end

function offspring = uniformCrossover(parent1, parent2, probability)
    if rand < probability % crossover probability
        mask = rand(size(parent1)) > 0.5; % random mask
        offspring = parent1; % copy genes
        offspring(mask) = parent2(mask); %apply mask to genes
    else
        offspring = parent1; % no crossover
    end
end

function mutated = mutation(individual, rate, method)
    for i = 1:length(individual)
        if rand < rate
            switch method
                case 0 %gaussian mutation
                    individual(i) = individual(i) + randn;
                case 1 %random resseting 
                    individual(i) = rand * 499 + 1;
                otherwise 
                    error('incorrect mutation method');
            end
            individual(i) = round(max(1, min(500, individual(i))));
        end 
    end 
    mutated = individual;
end