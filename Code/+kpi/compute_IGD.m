function igd = compute_IGD(A, Z)
    % COMPUTE_IGD - Inverted Generational Distance (IGD) metric
    %
    % This function computes the Inverted Generational Distance (IGD), which 
    % quantifies how well the approximated front A covers the true Pareto front Z.
    % Unlike GD, IGD measures the average distance from each point in the true 
    % Pareto front to the closest point in the approximated front.
    %
    % Inputs:
    %   A - N x M matrix of approximated solutions (in objective space)
    %   Z - K x M matrix of reference Pareto-optimal solutions
    %
    % Output:
    %   igd - scalar value representing the mean distance from Z to A
    %
    % Reference:
    %   Li and Xao (2019), «Quality Evaluation of Solution Sets in Multiobjective
    %   Optimization: A survey», ACM Comput. Surv., vol. 52, no. 2, art. 26

    m = size(Z, 1);      % number of reference points
    d = zeros(m, 1);     % vector to store distances from Z to A

    for i = 1:m
        % For each reference point in Z, compute the minimum distance
        % to any point in the approximation set A
        d(i) = min(vecnorm(A - Z(i,:), 2, 2));
    end

    % IGD is the average of these distances
    igd = mean(d);
end

