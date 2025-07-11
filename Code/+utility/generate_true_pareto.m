function [Z, ref_point] = generate_true_pareto(problem_number, n_points)
    % Generates the true Pareto front and reference point for test problems.
    %
    % [Z, ref_point] = generate_true_pareto(problem_number, n_points)
    %
    % Inputs:
    %     problem_number - Integer in [1, 10], selects the benchmark function:
    %                      1: SCH, 2: FON, 3: POL, 4: KUR, 5–9: ZDT1–6, 10: VLMOP2
    %     n_points       - Number of points to generate on the Pareto front.
    %
    % Outputs:
    %     Z         - [n x 2] matrix of objective values [f1, f2] on the true front.
    %     ref_point - 1x2 vector, reference point for Hypervolume calculation.
    %
    % Notes:
    %     - Analytical expressions used when available (e.g., ZDTs, FON).
    %     - POL and KUR are sampled numerically and filtered via non-dominated sorting.
    %     - Assumes 'non_domination_sort_mod' is available in the path.
    %
    % Example:
    %     [Z, ref] = generate_true_pareto(6, 1000);  % ZDT2 front with 1000 points

    %% Reference point selection rationale:
    % For Hypervolume (HV) calculation, the choice of reference point (ref_point) 
    % must ensure that all relevant Pareto-optimal points dominate it. 
    % This is critical to avoid underestimating HV or excluding large portions of the front.
    %
    % SCH case:
    % The true Pareto front of the SCH problem spans a wide asymmetric range:
    % - f1 ∈ [0, 1] from x^2
    % - f2 ∈ [1, 4] from (x - 2)^2, when x ∈ [-1, 1]
    % Using a reference point like [1.1, 1.1] (commonly used in problems normalized in [0,1]) 
    % would result in only a very narrow portion near x ≈ 1 (where f2 ≈ 1) contributing to HV,
    % and all the central and left-hand side of the front (e.g., the point [0, 4]) being ignored.
    % Therefore, a larger ref_point such as [2, 2] is used to safely encompass the entire front
    % and ensure that every solution contributes meaningfully to the Hypervolume measure.
    %
    % FON case:
    % The Fonseca–Fleming problem produces a Pareto front in a curved region where both objectives
    % lie within [0, 1] due to the exponential mapping from symmetric input domains.
    % The maximum objective values are approximately f1 ≈ 0.998 and f2 ≈ 0.998.
    % Therefore, using a reference point at [1.0, 1.0] is sufficient to fully include
    % the entire front for Hypervolume computation, while avoiding overextension beyond
    % the meaningful part of the objective space.
    %
    % POL case:
    % The Poloni problem produces a discontinuous and non-convex Pareto front with objective values
    % that are not known in closed form and must be sampled densely in the decision space.
    % To ensure that only non-dominated solutions contribute to the Hypervolume,
    % we generate a dense grid and extract the first front. The ref_point is then dynamically computed 
    % as [max(f1)*1.1, max(f2)*1.1], ensuring full coverage of the actual non-dominated region.
    % This adaptive approach avoids the risk of excluding parts of the front or inflating HV 
    % when the actual front lies far from canonical bounds like [0,1]^2.
    %
    % KUR case:
    % The Kursawe problem generates a complex, non-convex Pareto front with strong discontinuities 
    % and wide value ranges, especially in f2, which can significantly exceed 1.
    % Since the front is sampled numerically and filtered via non-dominated sorting,
    % its range varies with the random sampling process.
    % Therefore, the reference point is computed as [max(f1)*1.1, max(f2)*1.1], 
    % ensuring that the entire extracted front contributes to HV,
    % regardless of how spread or asymmetric the objective values are.
    %
    % ZDT cases (ZDT1–ZDT6):
    % In contrast, all ZDT problems are constructed so that their Pareto fronts are fully contained
    % within the [0, 1] × [0, 1] objective space. Therefore, a small margin above the front,
    % like ref_point = [1.1, 1.1], is sufficient and standard practice. It guarantees that the
    % entire front is within the HV region while avoiding excessive inflation of the measured volume.
    %
    % VLMOP2 case:
    % The VLMOP2 problem generates a smooth, convex, symmetric Pareto front entirely within [0, 1]^2.
    % In particular, both f1 and f2 approach ~0.99 at the extremes.
    % A reference point of [1.1, 1.1] ensures complete inclusion of the front for HV calculation
    % while introducing a small buffer zone outside the front, consistent with common benchmarking practices.


    switch problem_number
        case 1 % SCH
            x = linspace(-1, 1, n_points)';
            f1 = x.^2;
            f2 = (x - 2).^2;
            Z = [f1, f2];
            ref_point = [2.0, 2.0];

        case 2 % FON
            x = linspace(-1, 1, n_points)';
            X = repmat(x, 1, 3);  % ogni punto è (x, x, x) in R^3
            f1 = 1 - exp(-sum((X - 1/sqrt(3)).^2, 2));
            f2 = 1 - exp(-sum((X + 1/sqrt(3)).^2, 2));
            Z = [f1, f2];
            ref_point = [1.0, 1.0];

        case 3 % POL
            % Generate a numerical approximation of the Pareto front for the POL problem
            % The POL problem (Poloni function) does not have a known closed-form expression 
            % for its Pareto front. Therefore, a grid-based sampling approach is used to 
            % approximate the front through evaluation of the objective functions and 
            % non-dominated sorting.
            
            % Step 1: Uniform sampling of the decision space [-π, π]^2
            % The total number of points is approximately n_points. To create a uniform 2D grid, 
            % we take the square root of n_points to define the number of points along each axis.
            x1 = linspace(-pi, pi, round(sqrt(n_points)));
            x2 = linspace(-pi, pi, round(sqrt(n_points)));
            
            % Create a meshgrid of all (x1,x2) combinations to cover the decision space
            [X1, X2] = meshgrid(x1, x2);
            
            % Flatten the 2D grid to obtain a list of input points in R²
            X1 = X1(:);
            X2 = X2(:);
            
            % Step 2: Compute the fixed constants A1 and A2 
            % These are part of the f1 objective formula and depend on sin(1), sin(2), etc.
            % They are scalar reference values used to compute squared differences with B1 and B2
            A1 = 0.5 * sin(1) - 2 * cos(1) + sin(2) - 1.5 * cos(2);
            A2 = 1.5 * sin(1) - cos(1) + 2 * sin(2) - 0.5 * cos(2);
            
            % Step 3: Evaluate the two objective functions f1 and f2 at all input points
            % B1 and B2 are the non-linear parts of f1, computed element-wise
            B1 = 0.5 * sin(X1) - 2 * cos(X1) + sin(X2) - 1.5 * cos(X2);
            B2 = 1.5 * sin(X1) - cos(X1) + 2 * sin(X2) - 0.5 * cos(X2);
            
            % f1 is a non-convex objective function measuring squared distance from (A1, A2)
            f1 = 1 + (A1 - B1).^2 + (A2 - B2).^2;
            
            % f2 is a convex quadratic function centered at (-3, -1), resulting in circular level sets
            f2 = (X1 + 3).^2 + (X2 + 1).^2;
            
            % Step 4: Combine decision variables and objectives into a full population matrix
            % Columns 1–2: decision variables, Columns 3–4: objective values
            population = [X1, X2, f1, f2];
            
            % Apply non-dominated sorting to extract Pareto-optimal solutions
            % The function assigns a rank to each point based on domination; rank 1 means Pareto front
            sorted = non_domination_sort_mod(population, 2, 2);
            
            % Step 5: Select only the points belonging to the first non-dominated front (rank = 1)
            % Extract columns corresponding to the objective values: f1 (col 3), f2 (col 4)
            Z = sorted(sorted(:, 5) == 1, 3:4);
            
            % Compute a reference point for Hypervolume calculation
            % This is set as 10% beyond the worst objective values in the Pareto front
            ref_point = [max(Z(:,1))*1.1, max(Z(:,2))*1.1];

        case 4 % KUR (Kursawe)
            % Kursawe problem (V = 3)
            % The Pareto front is not analytically known, so we approximate it numerically
            % by evaluating the objective functions over a large sample of decision vectors
            % and selecting the non-dominated ones.
        
            V = 3;  % Number of decision variables (fixed by definition of KUR)
            n_samples = n_points * 50;  % Oversampling to ensure a sufficient Pareto front subset
        
            % Step 1: Uniform random sampling in the decision space [-5, 5]^V
            X = -5 + 10 * rand(n_samples, V);
        
            % Step 2: Evaluate objective function f1
            % f1 = sum_{i=1}^{V-1} -10 * exp(-0.2 * sqrt(x_i^2 + x_{i+1}^2))
            f1 = zeros(n_samples, 1);
            for i = 1:V-1
                f1 = f1 - 10 * exp(-0.2 * sqrt(X(:,i).^2 + X(:,i+1).^2));
            end
        
            % Step 3: Evaluate objective function f2
            % f2 = sum_{i=1}^{V} (|x_i|^0.8 + 5 * sin(x_i^3))
            f2 = sum(abs(X).^0.8 + 5 * sin(X.^3), 2);
        
            % Step 4: Non-dominated sorting of the full population
            % Format: [decision variables | objectives]
            population = [X, f1, f2];  % size: n_samples x (V + 2)
            sorted = non_domination_sort_mod(population, 2, V);
        
            % Step 5: Extract first non-dominated front (rank = 1)
            Z = sorted(sorted(:, V+3) == 1, V+1:V+2);  % objective values are in columns V+1 and V+2
        
            % Step 6: Reference point for Hypervolume computation
            % Set slightly beyond the worst point of the Pareto front
            ref_point = [max(Z(:,1))*1.1, max(Z(:,2))*1.1];

        case 5 % ZDT1
            f1 = linspace(0, 1, n_points)';
            f2 = 1 - sqrt(f1);
            Z = [f1, f2];
            ref_point = [1.1, 1.1];

        case 6 % ZDT2
            f1 = linspace(0, 1, n_points)';
            f2 = 1 - (f1).^2;
            Z = [f1, f2];
            ref_point = [1.1, 1.1];

        case 7 % ZDT3
            f1 = linspace(0, 1, n_points)';
            f2 = 1 - sqrt(f1) - f1 .* sin(10 * pi * f1);
            Z = [f1, f2];
            ref_point = [1.1, 1.1];

        case 8 % ZDT4
            f1 = linspace(0, 1, n_points)';
            f2 = 1 - sqrt(f1);
            Z = [f1, f2];
            ref_point = [1.1, 1.1];

        case 9 % ZDT6
            f1 = linspace(0, 1, n_points)';
            f1 = 1 - exp(-4*f1) .* (sin(6*pi*f1)).^6;
            f2 = 1 - (f1).^2;
            Z = [f1, f2];
            ref_point = [1.1, 1.1];

        case 10 % VLMOP2
            x = linspace(-2, 2, n_points)';
            f1 = 1 - exp(-((x - 1/sqrt(2)).^2));
            f2 = 1 - exp(-((x + 1/sqrt(2)).^2));
            Z = [f1, f2];
            ref_point = [1.1, 1.1];

        otherwise
            error('Unknown problem number. Must be 1 to 10.');
    end
end
