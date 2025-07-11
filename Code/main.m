% === CONFIG ===
n_runs = 30;
pop = 50;
gen = 1000;
problem_number = 7; 

[Z, ref_point] = utility.generate_true_pareto(problem_number, 1000);

detailed_results = {'Algorithm', 'Run', 'GD', 'IGD', 'Delta', 'HV_Platemo', 'HV_Rectangles'};

% === Output storage ===
algos = {'moead_cheby', 'moead_linear', 'moead_mod_cheby', 'moead_mod_linear', 'nsga2'};
metrics = struct();
for a = 1:length(algos)
    metrics.(algos{a}) = struct('gd', [], 'igd', [], 'delta', [], 'hv_platemo', [], 'hv_rectangles', []);
end

% === LOOP su run ===
for r = 1:n_runs
    seed = r; % fisso per riproducibilità
    rng(seed);

    for a = 1:length(algos)
        algo = algos{a};
        fprintf('[Seed %2d] Running %s...\n', seed, algo);

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

        % === Calcola metriche ===
        gd_val = kpi.compute_GD(A, Z);
        igd_val = kpi.compute_IGD(A, Z);
        delta_val = max(gd_val, igd_val);
        hv_val_platemo = kpi.compute_HV_platemo(A, ref_point);
        hv_val_rectangles = kpi.compute_HV_rectangles(A, ref_point);

        % Salva risultato corrente nello storico
        detailed_results(end+1,:) = {algo, r, gd_val, igd_val, delta_val, hv_val_platemo, hv_val_rectangles};

        % === Salva metriche ===
        metrics.(algo).gd(end+1) = gd_val;
        metrics.(algo).igd(end+1) = igd_val;
        metrics.(algo).delta(end+1) = delta_val;
        metrics.(algo).hv_platemo(end+1) = hv_val_platemo;
        metrics.(algo).hv_rectangles(end+1) = hv_val_rectangles;
    end
end

% === Stampa e salva risultato medio ===
fprintf('\nAverage metrics over %d runs:\n', n_runs);
fprintf('%-20s %-8s %-8s %-8s %-8s %-8s\n', 'Algorithm', 'GD', 'IGD', 'Delta', 'HV_Platemo', 'HV_Rectangles');

% Prepara la tabella per la scrittura su file
results_table = cell(length(algos)+1, 6);
results_table(1,:) = {'Algorithm', 'GD', 'IGD', 'Delta', 'HV_Platemo', 'HV_Rectangles'};

for a = 1:length(algos)
    algo = algos{a};
    avg_gd = mean(metrics.(algo).gd);
    avg_igd = mean(metrics.(algo).igd);
    avg_delta = mean(metrics.(algo).delta);
    avg_hv_platemo = mean(metrics.(algo).hv_platemo);
    avg_hv_rectangles = mean(metrics.(algo).hv_rectangles);

    fprintf('%-20s %-8.4f %-8.4f %-8.4f %-8.4f %-8.4f\n', algo, avg_gd, avg_igd, avg_delta, avg_hv_platemo, avg_hv_rectangles);

    results_table{a+1,1} = algo;
    results_table{a+1,2} = avg_gd;
    results_table{a+1,3} = avg_igd;
    results_table{a+1,4} = avg_delta;
    results_table{a+1,5} = avg_hv_platemo;
    results_table{a+1,6} = avg_hv_rectangles;
end

% Salvataggio su CSV
utility.cell2csv('output/average_metrics.csv', results_table);
utility.cell2csv('output/detailed_metrics.csv', detailed_results);
