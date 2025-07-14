function hv = compute_HV_HSO(points, ref_point)
    % compute_HV_HSO - Computes the exact Hypervolume (HV) using the 
    %                  Hypervolume by Slicing Objectives (HSO) method.
    %
    % Inputs:
    %   points     - Matrix of non-dominated solutions (N x M)
    %   ref_point  - Reference point (1 x M), worse than all points
    %
    % Output:
    %   hv         - Hypervolume scalar value
    %
    % Notes:
    %   - Only supports M < 4 (i.e., up to tri-objective problems)
    %   - For higher dimensions, use approximate methods
    
    [~, M] = size(points);

    % Normalize the points into the unit hypercube [0,1]^M
    fmin = min(min(points,[],1), zeros(1,M));
    fmax = ref_point;
    points = (points - fmin) ./ (fmax - fmin);

    % Remove any points outside the normalized bounds
    points(any(points > 1, 2), :) = [];

    % Redefine reference point to be ones (after normalization)
    RefPoint = ones(1, M);

    if isempty(points)
        hv = 0;
        return;
    elseif M < 4
        %% HSO Algorithm (Exact) for M < 4
        pl = sortrows(points);
        S  = {1, pl};

        for k = 1 : M-1
            S_ = {};
            for i = 1 : size(S,1)
                % Slice current region along dimension k
                Stemp = Slice(S{i,2}, k, RefPoint);
                for j = 1 : size(Stemp,1)
                    % Accumulate slices and their measure recursively
                    temp = {Stemp{j,1} * S{i,1}, Stemp{j,2}};
                    S_ = Add(temp, S_);
                end
            end
            S = S_;
        end

        % Final accumulation of volumes along last dimension
        hv = 0;
        for i = 1 : size(S,1)
            p = Head(S{i,2});
            hv = hv + S{i,1} * abs(p(M) - RefPoint(M));
        end

        % Scale back hypervolume according to reference point scaling
        hv = hv * prod(ref_point);
    else
        error('This function supports only M < 4 for exact HSO.');
    end
end


function S = Slice(pl, k, RefPoint)
    % Slice - Performs slicing along the k-th objective axis.
    %
    % Inputs:
    %   pl       - Set of sorted points in current region
    %   k        - Current slicing dimension
    %   RefPoint - Normalized reference point (all ones)
    %
    % Output:
    %   S        - Cell array of {measure, subregion points} after slicing
    
    p = Head(pl);
    pl = Tail(pl);
    ql = [];
    S = {};

    while ~isempty(pl)
        % Insert the point into ql while preserving dominance
        ql = Insert(p, k+1, ql);
        p_ = Head(pl);
        % Create new slice between p and p_ in dimension k
        cell_ = {abs(p(k)-p_(k)), ql};
        S = Add(cell_, S);
        p = p_;
        pl = Tail(pl);
    end

    % Final slice up to reference point
    ql = Insert(p, k+1, ql);
    cell_ = {abs(p(k)-RefPoint(k)), ql};
    S = Add(cell_, S);
end


function ql = Insert(p, k, pl)
    % Insert - Inserts point p into list pl maintaining order in dimension k
    %          and removing dominated points in dimensions >= k.
    %
    % Inputs:
    %   p   - Point to insert (1 x M)
    %   k   - Dimension index to sort on
    %   pl  - Current list of points
    %
    % Output:
    %   ql  - Updated list with point p inserted
    
    flag1 = 0;
    flag2 = 0;
    ql = [];

    % Insert p while maintaining increasing order along dimension k
    hp = Head(pl);
    while ~isempty(pl) && hp(k) < p(k)
        ql = [ql; hp];
        pl = Tail(pl);
        hp = Head(pl);
    end
    ql = [ql; p];
    m = length(p);

    % Remove dominated points (those that p dominates in dims >= k)
    while ~isempty(pl)
        q = Head(pl);
        for i = k:m
            if p(i) < q(i)
                flag1 = 1;
            elseif p(i) > q(i)
                flag2 = 1;
            end
        end
        if ~(flag1 && ~flag2)
            ql = [ql; q];
        end
        pl = Tail(pl);
    end
end


function p = Head(pl)
    % Head - Returns the first point of list pl
    if isempty(pl)
        p = [];
    else
        p = pl(1,:);
    end
end


function ql = Tail(pl)
    % Tail - Returns the tail of the list pl (all points except the first)
    if size(pl,1) < 2
        ql = [];
    else
        ql = pl(2:end,:);
    end
end


function S_ = Add(cell_, S)
    % Add - Adds or merges a slice into the current list of slices
    %
    % Inputs:
    %   cell_ - {measure, point list} to add
    %   S     - Current list of slices {measure, point list}
    %
    % Output:
    %   S_    - Updated list of slices
    
    n = size(S,1);
    m = 0;
    for k = 1 : n
        % Merge slices with identical point lists
        if isequal(cell_{1,2}, S{k,2})
            S{k,1} = S{k,1} + cell_{1,1};
            m = 1;
            break;
        end
    end
    if m == 0
        S(n+1,:) = cell_;
    end
    S_ = S;
end
