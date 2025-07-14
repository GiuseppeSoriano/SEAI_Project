function hv = compute_HV_platemo(points, ref_point)
    % compute_HV - Computes the hypervolume (HV) of a Pareto front approximation.
    %
    % Inputs:
    %   points     - Matrix of non-dominated objective vectors (N x M)
    %   ref_point  - Reference point (worse than all solutions), 1 x M
    %
    % Output:
    %   hv         - Hypervolume scalar value
    %
    % Note:
    %   - Uses exact computation for M=2
    %   - Uses Monte Carlo estimation for M > 2

    hv = sum(CalHV(points, ref_point, size(points,1), 10000));  % k = N, nSample = 10,000
end



function F = CalHV(points,bounds,k,nSample)
    % CalHV - Computes hypervolume-based fitness for each point
    %
    % Inputs:
    %   points   - N x M matrix of solutions
    %   bounds   - Reference point (hypervolume upper bounds)
    %   k        - Index cutoff for inclusion (usually = N)
    %   nSample  - Number of Monte Carlo samples for estimation
    %
    % Output:
    %   F        - Hypervolume contribution of each point
    
    %------------------------------- Copyright --------------------------------
    % Copyright (c) 2025 BIMK Group. You are free to use the PlatEMO for
    % research purposes. All publications which use this platform or any code
    % in the platform should acknowledge the use of "PlatEMO" and reference "Ye
    % Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
    % for evolutionary multi-objective optimization [educational forum], IEEE
    % Computational Intelligence Magazine, 2017, 12(4): 73-87".
    %--------------------------------------------------------------------------
    
    % This function is modified from the code in
    % http://www.tik.ee.ethz.ch/sop/download/supplementary/hype/

    [N,M] = size(points);
    if M > 2
        %% Approximate HV for M > 2 using Monte Carlo sampling
        alpha = zeros(1,N); 
        for i = 1 : k 
            alpha(i) = prod((k-[1:i-1])./(N-[1:i-1]))./i; 
        end

        % Create random sample points between ideal point and reference point
        Fmin = min(points,[],1);    % ideal point
        S    = unifrnd(repmat(Fmin,nSample,1),repmat(bounds,nSample,1)); % uniformly sampled solutions

        PdS  = false(N,nSample);    % matrix: PdS(i,j) true if point i dominates sample j
        dS   = zeros(1,nSample);    % how many solutions dominate each sample point

        for i = 1 : N
            % For each point in the current solution set (point i)...
        
            % Compute which of the randomly sampled points S are **dominated** by point i.
            %
            % Explanation:
            % - `repmat(points(i,:), nSample, 1)` replicates point i into a matrix of size nSample x M.
            % - Subtracting S gives the vector difference (point_i - sample_j) for each j.
            % - The condition `<= 0` means point i is **no worse** than sample j in all objectives.
            % - `sum(..., 2) == M` checks that this condition holds for **all** M objectives → i dominates sample j.
            %
            % Result:
            %   - `x` is a logical column vector (nSample x 1)
            %   - x(j) = true if point i dominates sample j
            
            x = sum(repmat(points(i,:), nSample, 1) - S <= 0, 2) == M;
        
            % Mark that point i dominates those sample points
            PdS(i, x) = true;
        
            % For each sample point j that is dominated by point i,
            % increment dS(j), i.e., count how many solutions dominate each sample
            dS(x) = dS(x) + 1;
        end


        % Accumulate hypervolume contribution using the alpha coefficients
        F = zeros(1,N);  % F(i) will store the estimated hypervolume contribution of solution i
        
        for i = 1 : N
            % For each sample point dominated by solution i, retrieve its dS count
            % Then use that to index into the alpha vector:
            % - alpha(dS(j)) gives the contribution of that point according to its "shared" dominance
            % - Sample points dominated by fewer solutions (i.e., smaller dS) contribute more
        
            % Sum all such contributions to compute the estimated share of hypervolume of solution i
            F(i) = sum(alpha(dS(PdS(i,:))));
        end


        % Normalize by total sampled volume
        F = F.*prod(bounds-Fmin)/nSample;
    else
        % === Accurate Hypervolume Computation for Bi-objective Problems (M == 2) ===
        
        % Create an index vector to keep track of original positions of the points.
        % This is important because the algorithm will sort the points but we want
        % to preserve mapping to the original individuals.
        pvec  = 1:size(points,1);
        
        % Precompute alpha coefficients for exact hypervolume contribution.
        % Alpha weights are based on combinatorial factors derived from the
        % inclusion-exclusion principle. They represent how many subsets a point
        % appears in as the best point when computing hypervolume contributions.
        alpha = zeros(1,k);
        for i = 1 : k 
            j = 1:i-1;
            % The formula is: alpha(i) = [product_{j=1}^{i-1} (k-j)/(N-j)] / i
            % See Emmerich & Deutz (2007), and Zitzler & Thiele for its derivation.
            alpha(i) = prod((k-j)./(N-j))./i;
        end
        
        % Call the recursive helper function that computes the exact hypervolume
        % contribution using objective space slicing (HSO method).
        % This implementation assumes M = 2, so it will recurse once down to M = 1.
        F = hypesub(N, points, M, bounds, pvec, alpha, k);
    end
end

function h = hypesub(l,A,M,bounds,pvec,alpha,k)
    % hypesub - Recursive function to compute exact hypervolume contribution
    %           for bi-objective problems.
    %
    % Inputs:
    %   l       - Number of individuals (solutions)
    %   A       - Current set of points in M-dimensional space
    %   M       - Current number of objectives (starts from M, decreases recursively)
    %   bounds  - Reference point in the objective space
    %   pvec    - Mapping from sorted to original indices
    %   alpha   - Precomputed alpha weights
    %   k       - Maximum number of points considered (used for alpha cutoff)
    %
    % Output:
    %   h       - Vector of hypervolume contributions for each individual

    %------------------------------- Copyright --------------------------------
    % Copyright (c) 2025 BIMK Group. You are free to use the PlatEMO for
    % research purposes. All publications which use this platform or any code
    % in the platform should acknowledge the use of "PlatEMO" and reference "Ye
    % Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
    % for evolutionary multi-objective optimization [educational forum], IEEE
    % Computational Intelligence Magazine, 2017, 12(4): 73-87".
    %--------------------------------------------------------------------------

    % Initialize hypervolume contribution vector for each solution.
    h = zeros(1, l); 

    % Sort the points based on the last objective (dimension M).
    % This creates slabs (slices) of the objective space by slicing along dimension M.
    [S, i] = sortrows(A, M); 
    % Update index vector to keep mapping between original and sorted points.
    pvec = pvec(i); 

    % Loop over all sorted points to construct slabs and compute contributions.
    for i = 1 : size(S,1) 
        % Compute the width of the current slab (difference in the M-th dimension).
        if i < size(S,1) 
            extrusion = S(i+1, M) - S(i, M);  % Between current and next point
        else
            extrusion = bounds(M) - S(i, M);  % From last point to reference bound
        end

        % If only 1 dimension left (base case), multiply extrusion by alpha.
        if M == 1
            if i > k
                break;  % Only compute contributions for first k points
            end
            if alpha(i) >= 0
                % Add the contribution to all current and previous points
                % (those up to i) as per the inclusion-exclusion formulation.
                h(pvec(1:i)) = h(pvec(1:i)) + extrusion * alpha(i); 
            end

        % If more than 1 dimension remains, recurse into M-1 dimensions.
        elseif extrusion > 0
            % Recursively compute the volume of the slice formed by the first i points.
            % Multiply it by the width (extrusion) to accumulate full volume.
            h = h + extrusion * hypesub(l, S(1:i, :), M-1, bounds, pvec(1:i), alpha, k); 
        end
    end
end