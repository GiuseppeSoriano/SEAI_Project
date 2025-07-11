function hv = compute_HV_rectangles(points, ref_point)
    % compute_hypervolume - Compute the 2D hypervolume using rectangles.
    %
    % Inputs:
    %   points     - N x 2 matrix of non-dominated points (assumed to be Pareto front)
    %   ref_point  - 1x2 vector [r1, r2], the reference point (nadir)
    %
    % Outputs:
    %   hv         - Total hypervolume value
    %   area       - Struct with field area.b(i), the area of each rectangle

    % Sort the Pareto front by the first objective (ascending)
    pareto_front = sortrows(points, 1);
    N = size(pareto_front, 1);
    area = struct();
    
    hv = 0;
    for i = 1:(N-1)
        area.b(i) = (ref_point(2) - pareto_front(i,2)) * ...
                    (pareto_front(i+1,1) - pareto_front(i,1));
        if area.b(i) >= 0
            hv = hv + area.b(i);
        end
    end

    % Last area: from last point to reference point (bottom-right corner)
    base_last = ref_point(1) - pareto_front(end,1);
    height_last = ref_point(2) - pareto_front(end,2);
    area.b(N) = base_last * height_last;
    hv = hv + area.b(N);
end