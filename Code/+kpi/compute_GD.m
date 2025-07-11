function gd = compute_GD(A, Z)
    % COMPUTE_GD - Generational Distance (GD) metric
    %
    % This function computes the Generational Distance (GD) between a set of 
    % approximated Pareto-optimal solutions and the true Pareto front.
    %
    % Inputs:
    %   A - N x M matrix of approximated solutions (N points in objective space)
    %   Z - K x M matrix of true Pareto front solutions
    %
    % Output:
    %   gd - scalar value representing the average minimum Euclidean distance 
    %        from each point in A to the closest point in Z
    % 
    % Reference:
    %   Li and Xao (2019), «Quality Evaluation of Solution Sets in Multiobjective
    %   Optimization: A survey», ACM Comput. Surv., vol. 52, no. 2, art. 26
    
    n = size(A, 1);       % number of approximated solutions
    d = zeros(n, 1);      % vector to store minimum distances to Z

    for i = 1:n
        % For each point in A, compute the Euclidean distance to all points in Z
        % and store the minimum
        d(i) = min(vecnorm(Z - A(i,:), 2, 2));
    end

    % GD is the mean of the minimum distances
    gd = mean(d);
end

