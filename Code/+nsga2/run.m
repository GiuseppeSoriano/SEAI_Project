function A = run(pop, gen, problem_number)

    %% NSGA-II Main Function
    % Performs multi-objective optimization using the NSGA-II algorithm.
    %
    % Inputs:
    %   - pop: Population size (minimum 20 individuals)
    %   - gen: Number of generations (minimum 5)
    %   - problem_number: Identifier of the benchmark problem to solve.
    %                     The corresponding objective function is predefined.
    %
    % Description:
    %   This function implements the NSGA-II algorithm to evolve a population
    %   of candidate solutions toward the Pareto front of a selected test problem.
    %   The benchmark problem is selected using the problem_number argument,
    %   which sets the number of objectives (M), the number of decision variables (V),
    %   and their respective bounds.
    %
    % Reference:
    %   K. Deb, A. Pratap, S. Agarwal, T. Meyarivan,
    %   "A fast and elitist multiobjective genetic algorithm: NSGA-II",
    %   IEEE Trans. Evol. Comput., vol. 6, no. 2, pp. 182–197, 2002.
    %
    % For more details, visit: http://www.iitk.ac.in/kangal/

    
    %% Input validation
    % Assign default values if inputs are missing

    if nargin < 3
        problem_number = 1;
    end
    
    if nargin < 2
        gen = 100;    
    end
    
    if nargin < 1
        pop = 30;
    end
    
    % Check that inputs are numeric
    if isnumeric(pop) == 0 || isnumeric(gen) == 0
        error('Both input arguments pop and gen should be integer datatype');
    end
    % Minimum population size has to be 20 individuals
    if pop < 20
        error('Minimum population for running this function is 20');
    end
    if gen < 5
        error('Minimum number of generations is 5');
    end
    
    % Make sure pop and gen are integers
    pop = round(pop);
    gen = round(gen);
    
    %% Objective Function
    % The objective function description contains information about the
    % objective function. M is the dimension of the objective space, V is the
    % dimension of decision variable space, min_range and max_range are the
    % range for the variables in the decision variable space. User has to
    % define the objective functions using the decision variables. 
    [M, V, min_range, max_range] = utility.get_problem_settings(problem_number);
    
    % Generate the true Pareto front for the selected problem.
    % This is used only for visualization and performance evaluation purposes.
    % The second output (ignored here) contains the reference point used to
    % compute HV.
    [true_pareto, ~] = utility.generate_true_pareto(problem_number, 200);
    
    %% Initialize the population
    % Population is initialized with random values which are within the
    % specified range. Each chromosome consists of the decision variables. Also
    % the value of the objective functions, rank and crowding distance
    % information is also added to the chromosome vector but only the elements
    % of the vector which has the decision variables are operated upon to
    % perform the genetic operations like crossover and mutation.
    chromosome = utility.initialize_variables(pop, M, V, min_range, max_range, problem_number);
    
    %% Sort the initialized population
    % Sort the population using non-domination-sort. This returns two columns
    % for each individual which are the rank and the crowding distance
    % corresponding to their position in the front they belong. At this stage
    % the rank and the crowding distance for each chromosome is added to the
    % chromosome vector for easy of computation.
    chromosome = utility.non_domination_sort_mod(chromosome, M, V);
    
    %% Start the evolution process
    % The following are performed in each generation
    % * Select the parents which are fit for reproduction
    % * Perfrom crossover and Mutation operator on the selected parents
    % * Perform Selection from the parents and the offsprings
    % * Replace the unfit individuals with the fit individuals to maintain a
    %   constant population size.
    
    upd = utility.textprogressbar(gen);

    for i = 1 : gen
        % Select the parents
        % Parents are selected for reproduction to generate offspring. The
        % original NSGA-II uses a binary tournament selection based on the
        % crowded-comparision operator. The arguments are 
        % pool - size of the mating pool. It is common to have this to be half the
        %        population size.
        % tour - Tournament size. Original NSGA-II uses a binary tournament
        %        selection, but to see the effect of tournament size this is kept
        %        arbitary, to be choosen by the user.
        pool = round(pop/2);
        tour = 2;
        % Selection process
        % A binary tournament selection is employed in NSGA-II. In a binary
        % tournament selection process two individuals are selected at random
        % and their fitness is compared. The individual with better fitness is
        % selcted as a parent. Tournament selection is carried out until the
        % pool size is filled. Basically a pool size is the number of parents
        % to be selected. The input arguments to the function
        % tournament_selection are chromosome, pool, tour. The function uses
        % only the information from last two elements in the chromosome vector.
        % The last element has the crowding distance information while the
        % penultimate element has the rank information. Selection is based on
        % rank and if individuals with same rank are encountered, crowding
        % distance is compared. A lower rank and higher crowding distance is
        % the selection criteria.

        parent_chromosome = nsga2.tournament_selection(chromosome, pool, tour);
    
        % Perform crossover and mutation to generate offspring population.
        % This implementation uses real-coded genetic operators:
        % - Simulated Binary Crossover (SBX), applied with 90% probability.
        % - Polynomial Mutation, applied with 10% probability.
        %
        % The parameters `mu` and `mum` are the distribution indices that control the
        % spread of offspring during crossover and mutation, respectively:
        %   - `mu`: Higher values lead to offspring closer to parents in SBX.
        %   - `mum`: Higher values lead to smaller perturbations in mutation.
        %
        % Typical values are mu = 20 and mum = 20, which encourage local search
        % while preserving diversity in early generations.
        mu = 20;
        mum = 20;
        offspring_chromosome = ...
            nsga2.genetic_operator(parent_chromosome, ...
            M, V, mu, mum, min_range, max_range, problem_number);

        % Combine current population and offspring into a single
        % intermediate population saved in intermediate_chromosome (size = 2*pop)

        [main_pop,~] = size(chromosome);
        [offspring_pop,~] = size(offspring_chromosome);
        intermediate_chromosome(1:main_pop,:) = chromosome;
        intermediate_chromosome(main_pop + 1 : main_pop + offspring_pop,1 : M+V) = ...
            offspring_chromosome;
    
        % The intermediate population is sorted again based on non-domination sort
        % before the replacement operator is performed on the intermediate
        % population.
        intermediate_chromosome = ...
            utility.non_domination_sort_mod(intermediate_chromosome, M, V);
        
        % Perform environmental selection to build the next generation.
        % Individuals are selected from the intermediate population in order of increasing rank.
        % If the last selected front exceeds the remaining slots, individuals with the highest
        % crowding distance are preferred to preserve diversity.
        chromosome = nsga2.replace_chromosome(intermediate_chromosome, M, V, pop);

        upd(i);
        
        % --- Optional: Visualize intermediate progress (disabled by default) ---
        % The following code allows visualization of the population's evolution
        % during the generational loop. It displays the Pareto front every 100 generations
        % (for M = 2 or M = 3), helping to monitor convergence and diversity.
        %
        % This section is left commented to ensure fair and efficient comparisons
        % across multiple algorithms within the experimental framework.
        % Enabling real-time plotting may significantly slow down performance
        % in large batch experiments or multi-run evaluations.
        %
        %     if ~mod(i,100)
        %         clc
        %         fprintf('%d generations completed\n',i);
        %     end
        %     if M == 2
        %         plot(chromosome(:,V + 1),chromosome(:,V + 2),'*');
        %     elseif M == 3
        %         plot3(chromosome(:,V + 1),chromosome(:,V + 2),chromosome(:,V + 3),'*');
        %     end
        %     fprintf('%d generations completed\n',i);
        %     figure(gcf);
        %     drawnow;
        %     pause(0.01);

    end
    
    %% Result
    % Save the result in ASCII text format.
    save solution.txt chromosome -ASCII
    
    %% Visualize
    % The following is used to visualize the result if objective space
    % dimension is visualizable.
    if M == 2
        plot(chromosome(:,V + 1),chromosome(:,V + 2),'b*'); hold on;
        if ~isempty(true_pareto)
            plot(true_pareto(:,1), true_pareto(:,2), 'r-', 'LineWidth', 1.5);
            legend('NSGA-II population', 'True Pareto front', 'Location', 'best');
        end
        hold off;
        saveas(gcf, 'output/nsga2.png');
    elseif M ==3
        plot3(chromosome(:,V + 1),chromosome(:,V + 2),chromosome(:,V + 3),'*'); hold on;
        if ~isempty(true_pareto)
            plot3(true_pareto(:,1), true_pareto(:,2), true_pareto(:,3), 'r-', 'LineWidth', 1.5);
            legend('NSGA-II population', 'True Pareto front', 'Location', 'best');
        end
        saveas(gcf, 'output/nsga2.png');
    end
    
    % Extract the first Pareto front (rank = 1) from the final population.
    % The structure of each chromosome is as follows:
    % - Columns 1 to V       : decision variables
    % - Columns V+1 to V+M   : objective function values
    % - Column  V+M+1        : non-domination rank (1 = first front)
    % - Columns V+M+2 onward : crowding distance (not used here)
    %
    % The final output A contains only the objective values of the non-dominated
    % individuals (first front), which are used for performance metrics such as
    % GD, IGD, Δ, and HV.
    f_sorted = utility.non_domination_sort_mod(chromosome, M, V);
    A = f_sorted(f_sorted(:, M + V + 1) == 1, V + 1 : V + M);
end
           
