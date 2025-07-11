function val = scalarization(f, z, lambda, scalarization_type)
    %% Scalarization function
    % Computes scalarized fitness from an objective vector `f`, using:
    % - `lambda`: weight vector for the subproblem
    % - `z`: ideal point (component-wise minimum over the population)
    %
    % Supports:
    %   'cheby'  - Weighted Chebyshev: max_i λ_i * |f_i - z_i|
    %   'linear' - Weighted L1 distance: sum_i λ_i * |f_i - z_i|
    %
    % Used in MOEA/D to compare solutions within each neighborhood.
    %
    % Ref: Q. Zhang & H. Li, IEEE TEVC, 2007 ("MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition")

    switch scalarization_type
        case 'cheby'
            % Weighted Chebyshev approach: focuses on the worst deviation
            val = max(lambda .* abs(f - z));
        case 'linear'
            % Weighted L1 distance: sums deviations across objectives
            val = sum(lambda .* abs(f - z));
        otherwise
            error('Unknown scalarization type. Must be linear or cheby.');
    end
end
