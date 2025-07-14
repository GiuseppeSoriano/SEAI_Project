function f  = genetic_operator(parent_chromosome, M, V, mu, mum, l_limit, u_limit, problem_number)

    %% function f  = genetic_operator(parent_chromosome, M, V, mu, mum, l_limit, u_limit, problem_number)
    % 
    % This function is utilized to produce offsprings from parent chromosomes.
    % The genetic operators crossover and mutation which are carried out with
    % slight modifications from the original design. For more information read
    % the document enclosed. 
    %
    % parent_chromosome - the set of selected chromosomes.
    % M - number of objective functions
    % V - number of decision varaiables
    % mu - distribution index for crossover (read the enlcosed pdf file)
    % mum - distribution index for mutation (read the enclosed pdf file)
    % l_limit - a vector of lower limit for the corresponding decsion variables
    % u_limit - a vector of upper limit for the corresponding decsion variables
    %
    % The genetic operation is performed only on the decision variables, that
    % is the first V elements in the chromosome vector. 
    %
    % Output:
    %   f                 : [N x (V+M)] matrix of offspring
    
    % Number of parents
    [N,~] = size(parent_chromosome);
    
    % Offspring counter
    p = 1;

    %Preallocate offspring matrix
    child = zeros(N, V + M);
    
    while p <= N
        % With 90 % probability perform crossover
        if rand(1) < 0.9 && p <= N-1
            % Select the first parent
            parent_1 = randi([1, N]);
            % Select the second parent
            parent_2 = randi([1, N]);
            % Make sure both the parents are not the same. 
            % Ensure that the two selected parents are not identical
            max_attempts = 10000;
            attempt = 0;
            while isequal(parent_chromosome(parent_1,1:V), parent_chromosome(parent_2,1:V))
                parent_2 = randi([1, N]);  % Select a new parent index in [1, N]
                attempt = attempt + 1;
                if attempt == max_attempts
                    error('Max attempts reached while selecting different parents. Skipping crossover.');
                end
            end

            % Get the chromosome information for each randomnly selected
            % parents
            parent_1 = parent_chromosome(parent_1,:);
            parent_2 = parent_chromosome(parent_2,:);

            % Perform corssover for each decision variable in the chromosome.
            child_1 = zeros(1, V);
            child_2 = zeros(1, V);
            for j = 1 : V
                % SBX (Simulated Binary Crossover).
                % For more information about SBX refer the enclosed pdf file.
                % Generate a random number
                u = rand(1);
                if u <= 0.5
                    beta = (2*u)^(1/(mu+1));
                else
                    beta = (1/(2*(1 - u)))^(1/(mu+1));
                end

                % Generate the jth element of first child
                child_1(j) = 0.5*(((1 + beta)*parent_1(j)) + (1 - beta)*parent_2(j));
                % Generate the jth element of second child
                child_2(j) = 0.5*(((1 - beta)*parent_1(j)) + (1 + beta)*parent_2(j));

                % Make sure that the generated element is within the specified
                % decision space else set it to the appropriate extrema.
                if child_1(j) > u_limit(j)
                    child_1(j) = u_limit(j);
                elseif child_1(j) < l_limit(j)
                    child_1(j) = l_limit(j);
                end
                if child_2(j) > u_limit(j)
                    child_2(j) = u_limit(j);
                elseif child_2(j) < l_limit(j)
                    child_2(j) = l_limit(j);
                end
            end

            % Evaluate the objective function for the offsprings and as before
            % concatenate the offspring chromosome with objective value.
            child_1(:,V + 1: M + V) = utility.evaluate_objective(child_1, M, V, problem_number);
            child_2(:,V + 1: M + V) = utility.evaluate_objective(child_2, M, V, problem_number);
            
            % Set the crossover flag. When crossover is performed two children
            % are generate, while when mutation is performed only only child is
            % generated.
            was_crossover = 1;
            was_mutation = 0;

        % With 10 % probability perform mutation. Mutation is based on
        % polynomial mutation. 
        else
            % Select at random the parent.
            parent_3 = randi([1,N]);

            % Get the chromosome information for the randomnly selected parent.
            child_3 = parent_chromosome(parent_3,:);
            
            % Perform mutation on each element of the selected parent using the
            % full Polynomial Mutation operator, as originally proposed in:
            %
            % K. Deb and M. Goyal, "A Combined Genetic Adaptive Search (GeneAS) for
            % Engineering Design", *Computer Science and Informatics*, vol. 26(4), pp. 30–45, 1996.
            %
            % A detailed procedural version is also available in:
            % M. Hamdan, "The Distribution Index in Polynomial Mutation for Evolutionary
            % Multiobjective Optimisation Algorithms: An Experimental Study",
            % *Proc. Int. Conf. on Electronics Computer Technology*, 2012.
            %
            % This version accounts for the true bounds of each variable and
            % avoids assuming a normalized domain [0,1]. For each variable, the
            % mutation delta is scaled according to the local distance to bounds,
            % ensuring consistency across variables of different magnitudes.
            
            for j = 1 : V
                r = rand(1);
            
                % Normalize current value with respect to decision bounds
                delta1 = (child_3(j) - l_limit(j)) / (u_limit(j) - l_limit(j));
                delta2 = (u_limit(j) - child_3(j)) / (u_limit(j) - l_limit(j));
            
                % Compute perturbation delta_q according to polynomial mutation distribution
                if r <= 0.5
                    delta_q = (2*r + (1 - 2*r)*(1 - delta1)^(mum+1))^(1/(mum+1)) - 1;
                else
                    delta_q = 1 - (2*(1 - r) + 2*(r - 0.5)*(1 - delta2)^(mum+1))^(1/(mum+1));
                end
            
                % Scale the perturbation by the actual decision range
                child_3(j) = child_3(j) + delta_q * (u_limit(j) - l_limit(j));
            
                % Ensure resulting value respects the bounds
                if child_3(j) > u_limit(j)
                    child_3(j) = u_limit(j);
                elseif child_3(j) < l_limit(j)
                    child_3(j) = l_limit(j);
                end
            end
            
            % -------------------------------------------------------------------------
            % Previous simplified version of the mutation (commented below):
            %
            % for j = 1 : V
            %     r = rand(1);
            %     if r < 0.5
            %         delta = (2*r)^(1/(mum+1)) - 1;
            %     else
            %         delta = 1 - (2*(1 - r))^(1/(mum+1));
            %     end
            %     child_3(j) = child_3(j) + delta;
            % end
            %
            % This approximation omits the normalization step and applies the delta directly,
            % assuming the decision variable lies in [0,1]. While computationally efficient,
            % it breaks the scale-invariance of the mutation operator and can produce
            % inconsistent behavior depending on the magnitude of the decision variable range.

            % Evaluate the objective function for the offspring and as before
            % concatenate the offspring chromosome with objective value.    
            child_3(:,V + 1: M + V) = utility.evaluate_objective(child_3, M, V, problem_number);
            % Set the mutation flag
            was_mutation = 1;
            was_crossover = 0;
        end
        % Keep proper count and appropriately fill the child variable with all
        % the generated children for the particular generation.
        if was_crossover
            child(p,:) = child_1;
            child(p+1,:) = child_2;
            p = p + 2;
        elseif was_mutation
            child(p,:) = child_3(1,1 : M + V);
            p = p + 1;
        end
    end
    f = child;
end