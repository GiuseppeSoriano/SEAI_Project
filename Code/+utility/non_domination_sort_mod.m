function f = non_domination_sort_mod(x, M, V)
%% non_domination_sort_mod - Fast non-dominated sorting with crowding distance
% This function sorts a given population based on non-domination, assigning
% each individual a "rank" (front number). Individuals in the first front get rank 1,
% the second front rank 2, and so on. It also calculates crowding distance
% within each front to promote diversity during selection.
%
% Inputs:
% - x: population matrix of size N x (V + M), where V = number of variables and M = objectives
% - M: number of objective functions
% - V: number of decision variables
%
% Output:
% - f: sorted population based on non-domination and crowding distance
    
    [N, ~] = size(x);
    
    % Initialize the front number to 1.
    front = 1;
    % Initialize first front as empty (stores indices of individuals in front 1)
    F(front).f = [];

    % Preallocate an array of structs to represent each individual in the population.
    % Each struct has:
    % - 'n': a counter of how many individuals dominate this one (initialized to 0)
    % - 'p': a list of individuals dominated by this one (initialized as empty array)
    %
    % This avoids dynamic struct growth inside loops, which is inefficient in MATLAB.
    individual = repmat(struct('n', 0, 'p', []), N, 1);
    
    %% Perform non-dominated sorting
    for i = 1 : N
        for j = 1 : N
            % dom_less: number of objectives where individual i is better than j
            % dom_equal: number of objectives where i and j are equal
            % dom_more: number of objectives where i is worse than j
            dom_less = 0;
            dom_equal = 0;
            dom_more = 0;
            for k = 1 : M
                if (x(i,V + k) < x(j,V + k))
                    dom_less = dom_less + 1;
                elseif (x(i,V + k) == x(j,V + k))
                    dom_equal = dom_equal + 1;
                else
                    dom_more = dom_more + 1;
                end
            end

            % Determine domination relationship:
            % i dominates j if it is no worse in all objectives (dom_less == 0)
            % and strictly better in at least one (dom_equal != M)

            if dom_less == 0 && dom_equal ~= M      % j dominates i -> increment number of individuals dominating i
                individual(i).n = individual(i).n + 1;
            elseif dom_more == 0 && dom_equal ~= M  % j is dominated by i -> add j to the domination list of i
                individual(i).p = [individual(i).p j];
            end
        end   

        % If no one dominates i, assign rank 1
        if individual(i).n == 0
            x(i,M + V + 1) = 1;
            F(front).f = [F(front).f i];
        end
    end

    %% Identify subsequent fronts
    % For each individual p in the current front:
    %   - Loop over all individuals q that are dominated by p
    %   - Decrease q’s domination count
    %   - If q is no longer dominated by anyone, assign it to the next front
    
    while ~isempty(F(front).f)
        Q = [];  % Temporary list for next front
        for i = 1:length(F(front).f)
            p = F(front).f(i);  % Current individual in the front
            for j = individual(p).p  % Individuals dominated by p
                individual(j).n = individual(j).n - 1;  % Reduce domination count
                if individual(j).n == 0
                    x(j, M + V + 1) = front + 1;  % Assign next front number
                    Q = [Q, j];  % Add j to next front
                end
            end
        end
        front = front + 1;      % Move to next front
        F(front).f = Q;         % Save the next front
    end
    
    %% Sort population by rank
    [~,index_of_fronts] = sort(x(:,M + V + 1));
    sorted_based_on_front = zeros(N, size(x, 2));  % Preallocate
    for i = 1 : length(index_of_fronts)
        sorted_based_on_front(i,:) = x(index_of_fronts(i),:);
    end

    current_index = 0;

    %% Crowding distance assignment
    % For each front Fi:
    %   - Initialize a temporary matrix y containing the individuals of Fi
    %   - Set the crowding distance to 0 for all individuals
    %   - For each objective function:
    %       * Sort individuals by that objective
    %       * Assign infinite distance to boundary solutions
    %       * For internal individuals, compute normalized distance
    %         based on spacing with neighbors
    %   - The total crowding distance is the sum over all objectives
    %   - Store the individuals with updated distances into final output array z

    z = zeros(N, M + V + 2);  % Preallocate final output array (includes rank and crowding distance)
    
    for front = 1:(length(F) - 1)
        % Extract current front individuals from sorted population
        y = zeros(length(F(front).f), size(x, 2));  % Temporary front matrix
        previous_index = current_index + 1;         % Track insertion position in z

        for i = 1:length(F(front).f)
            y(i, :) = sorted_based_on_front(current_index + i, :);
        end
        current_index = current_index + i;

        % Initialize crowding distance column to 0
        y(:, M + V + 2) = 0;

        for i = 1:M
            obj_col = V + i;  % Column index for the i-th objective
            [~, sorted_indices] = sort(y(:, obj_col));  % Sort by current objective

            f_max = y(sorted_indices(end), obj_col);  % Max value of objective
            f_min = y(sorted_indices(1), obj_col);    % Min value of objective

            % Assign infinite distance to boundary solutions
            y(sorted_indices(1), M + V + 2) = Inf;
            y(sorted_indices(end), M + V + 2) = Inf;

            % Compute normalized crowding distance for internal individuals
            for j = 2:(length(sorted_indices) - 1)
                next_obj = y(sorted_indices(j + 1), obj_col);
                prev_obj = y(sorted_indices(j - 1), obj_col);

                if f_max - f_min == 0
                    dist = 0;  % Avoid division by zero if all values are equal
                else
                    dist = (next_obj - prev_obj) / (f_max - f_min);
                end

                y(sorted_indices(j), M + V + 2) = ...
                    y(sorted_indices(j), M + V + 2) + dist;
            end
        end

        % Keep only the relevant columns and store the result in output matrix z
        y = y(:, 1:M + V + 2);  
        z(previous_index:current_index, :) = y;
    end


    f = z;  % Final population sorted by rank and with crowding distance
end

%% References
% [1] *Kalyanmoy Deb, Amrit Pratap, Sameer Agarwal, and T. Meyarivan*, |A Fast
% Elitist Multiobjective Genetic Algorithm: NSGA-II|, IEEE Transactions on 
% Evolutionary Computation 6 (2002), no. 2, 182 ~ 197.
%
% [2] *N. Srinivas and Kalyanmoy Deb*, |Multiobjective Optimization Using 
% Nondominated Sorting in Genetic Algorithms|, Evolutionary Computation 2 
% (1994), no. 3, 221 ~ 248.