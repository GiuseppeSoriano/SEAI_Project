function f = genetic_operator(parents, M, V, mu, mum, l_limit, u_limit, problem_number)
    %% genetic_operator - Genetic operator for MOEA/D (1-offspring version)
    % This function generates a **single offspring** from **two parents** using 
    % **Simulated Binary Crossover (SBX)** and **Polynomial Mutation**, in a style
    % consistent with NSGA-II. While the original MOEA/D uses only mutation, here
    % a combined operator is adopted for comparability across algorithms.
    %
    % Inputs:
    % - parents:        [2 x (V+M)] matrix containing two parent individuals.
    % - M:              Number of objective functions.
    % - V:              Number of decision variables.
    % - mu:             Distribution index for SBX crossover.
    % - mum:            Distribution index for polynomial mutation.
    % - l_limit:        Lower bounds for decision variables [1 x V].
    % - u_limit:        Upper bounds for decision variables [1 x V].
    % - problem_number: Problem identifier for evaluation.
    %
    % Output:
    % - f:              [1 x (V+M)] offspring chromosome (decision variables + objectives).

    % Extract the two parents
    parent_1 = parents(1, :);
    parent_2 = parents(2, :);

    % Initialize offspring (decision variables only)
    child = zeros(1, V);

    % SBX crossover with 90% probability
    if rand < 0.9
        for j = 1:V
            u = rand;
            if u <= 0.5
                beta = (2*u)^(1/(mu+1));
            else
                beta = (1/(2*(1 - u)))^(1/(mu+1));
            end

            % Compute child gene j using SBX formula
            child(j) = 0.5 * ((1 + beta) * parent_1(j) + (1 - beta) * parent_2(j));

            % Boundary correction
            child(j) = min(max(child(j), l_limit(j)), u_limit(j));
        end
    else
        % Mutation only (10% probability)
        child = parent_1(1:V);  % clone parent_1

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
            delta1 = (child(j) - l_limit(j)) / (u_limit(j) - l_limit(j));
            delta2 = (u_limit(j) - child(j)) / (u_limit(j) - l_limit(j));
            
            % Compute perturbation delta_q according to polynomial mutation distribution
            if r <= 0.5
                delta_q = (2*r + (1 - 2*r)*(1 - delta1)^(mum+1))^(1/(mum+1)) - 1;
            else
                delta_q = 1 - (2*(1 - r) + 2*(r - 0.5)*(1 - delta2)^(mum+1))^(1/(mum+1));
            end
            
            % Scale the perturbation by the actual decision range
            child(j) = child(j) + delta_q * (u_limit(j) - l_limit(j));
            
            % Ensure resulting value respects the bounds
            if child(j) > u_limit(j)
                child(j) = u_limit(j);
            elseif child(j) < l_limit(j)
                child(j) = l_limit(j);
            end
        end
    end

    % Assemble the full chromosome: decision vars + objectives
    f = zeros(1, M + V);
    f(1:V) = child;
    f(V+1 : V+M) = utility.evaluate_objective(child, M, V, problem_number);
end
