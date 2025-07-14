function [M, V, min_range, max_range] = get_problem_settings(problem_number)
    %GET_PROBLEM_SETTINGS - Returns problem-specific settings for benchmark problems.
    %
    % Inputs:
    %   problem_number - integer specifying the problem
    %
    % Outputs:
    %   M         - number of objective functions
    %   V         - number of decision variables
    %   min_range - lower bounds for decision variables (1xV)
    %   max_range - upper bounds for decision variables (1xV)
    %
    % Reference problems mostly from:
    %   K. Deb et al., "A fast and elitist Multiobjective Genetic Algorithm: NSGA-II",
    %   IEEE Transactions on Evolutionary Computation, vol. 6, no. 2, April 2002.

    switch problem_number

        case 1 % SCH
            M = 2;
            V = 1;
            min_range = -10^3;
            max_range = +10^3;
            
        case 2 % FON
            M = 2;
            V = 3;
            min_range = [-4, -4, -4];
            max_range = [+4, +4, +4];
            
        case 3 % POL
            M = 2;
            V = 2;
            min_range = [-3.14, -3.14];
            max_range = [+3.14, +3.14];
            
        case 4 % KUR
            M = 2;
            V = 3;
            min_range = [-5, -5, -5];
            max_range = [+5, +5, +5];
    
            
        case 5 % ZDT1
            M = 2;
            V = 2; % V can range from 2 to 30;
            min_range = zeros(1,V);
            max_range = ones( 1,V);

        case 6 % ZDT2
            M = 2;
            V = 2; % V can range from 2 to 30;
            min_range = zeros(1,V);
            max_range = ones( 1,V);
        
        case 7 % ZDT3
            M = 2;
            V = 2; % V can range from 2 to 30;
            min_range = zeros(1,V);
            max_range = ones( 1,V);
    
        case 8 % ZDT4
            M = 2;
            V = 3; % V can range from 2 to 30;
            min_range = [0, -5*ones(1,V-1)];
            max_range = [1, +5*ones(1,V-1)];
    
        case 9 % ZDT6
            M = 2;
            V = 3; % V can range from 2 to 10;
            min_range = zeros(1,V);
            max_range = ones(1,V);
            
        case 10 % VLMOP2 (from ParEGO paper)
		    M = 2;
            V = 2; 
            min_range = -2*ones(1, V); % x1,x2 in [-2,2]
            max_range = +2*ones(1, V);       

        case 11 % DTLZ1 (3 objectives)
            M = 3;                  % Number of objectives
            k = 0;                  % Number of distance-related variables (tipico: 5 o 10)
            V = M + k - 1;          % Number of decision variables (es. 7 con k=5)
            min_range = zeros(1, V);
            max_range = ones(1, V);
            
        otherwise 
            error('Wrong problem number');
    end
end
