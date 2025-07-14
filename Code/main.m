%% === CONFIGURATION ===
n_runs = 30;
pop = 200;
gen = 2000;
problem_number = 11;

% --- Problem settings and reference Pareto front ---
[M, ~, ~, ~] = utility.get_problem_settings(problem_number);
[Z, ref_point] = utility.generate_true_pareto(problem_number, 1000);

% --- Initialize output structures ---
detailed_results = {'Algorithm', 'Run', 'GD', 'IGD', 'Delta', 'HV_Platemo', 'HV_Rectangles', 'HV_HSO'};

algos = {'moead_linear', 'moead_cheby', 'moead_mod_linear', 'moead_mod_cheby', 'nsga2'};
metrics = struct();
for a = 1:length(algos)
    metrics.(algos{a}) = struct('gd', [], 'igd', [], 'delta', [], 'hv_platemo', [], 'hv_rectangles', [], 'hv_hso', []);
end


%% === MAIN EXPERIMENT LOOP ===
for r = 1:n_runs
    seed = r;  % Fixed seed for reproducibility
    rng(seed);

    for a = 1:length(algos)
        algo = algos{a};
        fprintf('[Seed %2d] Running %s...\n', seed, algo);

        % --- Run the selected algorithm ---
        switch algo
            case 'moead_cheby'
                A = moead.run(pop, gen, problem_number, 'cheby');
            case 'moead_linear'
                A = moead.run(pop, gen, problem_number, 'linear');
            case 'moead_mod_cheby'
                A = moead_modified.run(pop, gen, problem_number, 'cheby');
            case 'moead_mod_linear'
                A = moead_modified.run(pop, gen, problem_number, 'linear');
            case 'nsga2'
                A = nsga2.run(pop, gen, problem_number);
        end

        % --- Compute performance metrics ---
        gd_val = kpi.compute_GD(A, Z);
        igd_val = kpi.compute_IGD(A, Z);
        delta_val = max(gd_val, igd_val);
        hv_val_platemo = kpi.compute_HV_platemo(A, ref_point);

        if M == 2
            hv_val_rectangles = kpi.compute_HV_rectangles(A, ref_point);
        else
            hv_val_rectangles = NaN;
        end

        hv_val_hso = kpi.compute_HV_HSO(A, ref_point);

        % --- Store detailed results ---
        detailed_results(end+1,:) = {algo, r, gd_val, igd_val, delta_val, hv_val_platemo, hv_val_rectangles, hv_val_hso};

        % --- Store metrics for averages ---
        metrics.(algo).gd(end+1) = gd_val;
        metrics.(algo).igd(end+1) = igd_val;
        metrics.(algo).delta(end+1) = delta_val;
        metrics.(algo).hv_platemo(end+1) = hv_val_platemo;
        metrics.(algo).hv_rectangles(end+1) = hv_val_rectangles;
        metrics.(algo).hv_hso(end+1) = hv_val_hso;
    end
end


%% === AVERAGE RESULTS REPORT ===
fprintf('\nAverage metrics over %d runs:\n', n_runs);
fprintf('%-20s %-8s %-8s %-8s %-12s %-12s %-12s\n', 'Algorithm', 'GD', 'IGD', 'Delta', 'HV_Platemo', 'HV_Rectangles', 'HV_HSO');

% Prepare results table for CSV output
results_table = cell(length(algos)+1, 7);
results_table(1,:) = {'Algorithm', 'GD', 'IGD', 'Delta', 'HV_Platemo', 'HV_Rectangles', 'HV_HSO'};

for a = 1:length(algos)
    algo = algos{a};
    avg_gd = mean(metrics.(algo).gd);
    avg_igd = mean(metrics.(algo).igd);
    avg_delta = mean(metrics.(algo).delta);
    avg_hv_platemo = mean(metrics.(algo).hv_platemo);
    avg_hv_rectangles = mean(metrics.(algo).hv_rectangles);
    avg_hv_hso = mean(metrics.(algo).hv_hso);

    fprintf('%-20s %-8.4f %-8.4f %-8.4f %-12.4f %-12.4f %-12.4f\n', ...
        algo, avg_gd, avg_igd, avg_delta, avg_hv_platemo, avg_hv_rectangles, avg_hv_hso);

    results_table{a+1,1} = algo;
    results_table{a+1,2} = avg_gd;
    results_table{a+1,3} = avg_igd;
    results_table{a+1,4} = avg_delta;
    results_table{a+1,5} = avg_hv_platemo;
    results_table{a+1,6} = avg_hv_rectangles;
    results_table{a+1,7} = avg_hv_hso;
end


%% === SAVE OUTPUT FILES ===
utility.cell2csv('output/average_metrics.csv', results_table);
utility.cell2csv('output/detailed_metrics.csv', detailed_results);

%% === FINAL CONSOLE MESSAGE ===
fprintf('\nExperiment completed successfully.\n');
fprintf('Results saved in:\n - output/average_metrics.csv\n - output/detailed_metrics.csv\n');
