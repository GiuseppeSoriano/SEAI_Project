function f = evaluate_objective(x, M, V, problem_number)

    %% function f = evaluate_objective(x, M, V, problem_number)
    % Function to evaluate the objective functions for the given input vector
    % x. x is an array of decision variables and f(1), f(2), etc are the
    % objective functions. The algorithm always minimizes the objective
    % function hence if you would like to maximize the function then multiply
    % the function by negative one. M is the numebr of objective functions and
    % V is the number of decision variables. 
    %
    % This functions is basically written by the user who defines his/her own
    % objective function. Make sure that the M and V matches your initial user
    % input. Make sure that the
    
    
    f = [];
            
    switch problem_number
        % Next problems have been used in the following paper:
        %  K. Deb., A. Pratap, S. Agarwal, and T. Meyarivan, "A fast and
        %  elitist Multiobjective Genetic Algorithm: NSGA-II", IEEE Trans.
        %  Evol. Comp., vol. 6, no. 2, April 2002.
    
        case 1 % Problem SCH
            f(1) = x(1).^2;
            f(2) = (x(1)-2).^2;
    
        case 2 % FON
            f(1) = 1 - exp(-( ((x(1)-inv(sqrt(3)))^2) + ((x(2)-inv(sqrt(3)))^2) + ((x(3)-inv(sqrt(3)))^2)));
            f(2) = 1 - exp(-( ((x(1)+inv(sqrt(3)))^2) + ((x(2)+inv(sqrt(3)))^2) + ((x(3)+inv(sqrt(3)))^2)));
            
        case 3 % POL
            A1 = 0.5*sin(1)-2*cos(1)+1*sin(2)-1.5*cos(2);
            A2 = 1.5*sin(1)-1*cos(1)+2*sin(2)-0.5*cos(2);
            B1 = 0.5*sin(x(1))-2*cos(x(1))+2*sin(x(2))-1.5*cos(x(2));
            B2 = 1.5*sin(x(1))-1*cos(x(1))+2*sin(x(2))-0.5*cos(x(2));        
            f(1) = 1+(A1-B1).^2+(A2-B2).^2;
            f(2) = ((x(1)+3).^2) + ((x(2)+1).^2);
            
        case 4 % KUR
            % Objective function one
            s = 0;
            for i = 1 : V - 1
                s = s - 10*exp(-0.2*sqrt((x(i))^2 + (x(i + 1))^2));
            end
            % Decision variables are used to form the objective function.
            f(1) = s;
    
            % Objective function two
            s = 0;
            for i = 1 : V
                s = s + (abs(x(i))^0.8 + 5*(sin(x(i)))^3);
            end
            % Decision variables are used to form the objective function.
            f(2) = s;
            
        case 5 % ZDT1
            f(1) = x(1);
            %% Intermediate function
            g_x = 1 + 9*(sum(x(2:V))/(V-1));
            f(2) = g_x*(1-sqrt(x(1)/g_x));
    
        case 6 % ZDT2
            f(1) = x(1);
            %% Intermediate function
            g_x = 1 + 9*(sum(x(2:V))/(V-1));
            f(2) = g_x*(1-((x(1)/g_x))^2);
        case 7 % ZDT3
            f(1) = x(1);
            %% Intermediate function
            g_x = 1 + 9*(sum(x(2:V))/(V-1));
            f(2) = g_x*(1-sqrt(x(1)/g_x)-(x(1)/g_x)*sin(10*pi*x(1)));
        case 8 % ZDT4
            f(1) = x(1);
            %% Intermediate function
            g_x = 1 + 10*(V-1) + sum(x(2:V).^2-10*cos(4*pi*x(2:V)));
            f(2) = g_x*(1-sqrt(x(1)/g_x));
        case 9 % ZDT6
            %% Objective function one
            % Decision variables are used to form the objective function.
            f(1) = 1 - exp(-4*x(1))*(sin(6*pi*x(1)))^6;
            s = 0;
            for i = 2 : V
                s = s + x(i)/4;
            end
            %% Intermediate function
            g_x = 1 + 9*(s)^(0.25);
    
            %% Objective function two
            f(2) = g_x*(1 - ((f(1))/(g_x))^2);
            
        case 10 % VLMOP2 (from ParEGO)                       
            sum1=0;
            sum2=0;        
            for i = 1 : V
                sum1 = sum1 + (x(i)-(1/sqrt(2)))^2;
                sum2 = sum2 + (x(i)+(1/sqrt(2)))^2;
            end        
            f(1) = 1 - exp(-sum1);
            f(2) = 1 - exp(-sum2);

        case 11  % DTLZ1 (3 objectives)
            k = V - M + 1;
            g = 100 * (k + sum((x(end-k+1:end) - 0.5).^2 - cos(20 * pi * (x(end-k+1:end) - 0.5))));
            f = zeros(1, M);
            for i = 1:M
                product = 1;
                for j = 1:(M - i)
                    product = product * x(j);
                end
                if i > 1
                    product = product * (1 - x(M - i + 1));
                end
                f(i) = 0.5 * (1 + g) * product;
            end
    
        otherwise 
            error('Wrong problem number');
    end
           
    
    %% Check for error
    if length(f) ~= M
        error('The number of decision variables does not match you previous input. Kindly check your objective function');
    end
end