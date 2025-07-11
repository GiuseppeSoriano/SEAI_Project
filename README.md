## On the Impact of Scalarization in MOEA/D

This project investigates how different scalarization strategies affect the performance of MOEA/D in multi-objective optimization tasks, compared to NSGA-II. The study focuses on three types of Pareto fronts—convex, non-convex, and disconnected—using benchmark problems from the ZDT family. The analysis includes:

* **MOEA/D** with Chebyshev and Linear (Weighted Sum) scalarization
* **MOEA/D + SBX** (a variant that incorporates crossover)
* **NSGA-II**, a dominance-based baseline algorithm

Performance is evaluated using metrics such as **Generational Distance (GD)**, **Inverted GD (IGD)**, **Diversity ($\Delta$)**, and **Hypervolume (HV)**.

---

### 📁 Project Structure

```
.
├── Report/                # Contains the full PDF report
│   ├── Resources/         # Images and resources used in the report   
│   ├── Documentation.md   # Markdown source for the report
│   ├── Documentation.pdf  # Compiled PDF report using Pandoc
│   ├── ieee.csl           # Citation style for the report
|   └── references.bib     # Bibliography file
├── Code/
│   ├── +kpi/              # Metric computation scripts
│   ├── +moead/            # MOEA/D (standard) implementation
│   ├── +moead_modified/   # MOEA/D + SBX variant
│   ├── +nsga2/            # NSGA-II implementation
│   ├── +utility/          # Helper functions and utilities
│   ├── output/            # Output CSVs and plots
│   ├── table-gen/         # Notebook and markdown table generator
│   └── main.m             # Entry point for running experiments
```

---

### 🚀 How to Run

1. Open MATLAB.
2. Add the `Code/` folder and its subfolders to the path:

```matlab
addpath(genpath('Code'));
```

3. Run the main script:

```matlab
main
```

This will execute the experiments for all algorithms on the selected problem, which can be chosen by modifying the `main.m` script. There are 10 predefined problems available, in the `get_problem_settings.m` function, including:
* SCH
* POL
* FON
* KUR
* ZDT1
* ZDT2
* ZDT3
* ZDT4
* ZDT6
* VLMOP2

---

### 📊 Output

* `.csv` files with averaged and detailed metric results per problem/algorithm
* `.png` plots showing the final Pareto fronts per algorithm

---

### ⚙️ Algorithms

* **NSGA-II**: Based on the original implementation by Deb et al. (2002)
* **MOEA/D**: Based on decomposition via scalarization (Chebyshev or Linear)
* **MOEA/D + SBX**: Uses Simulated Binary Crossover + mutation

All versions support modular configuration and reproducibility via fixed random seeds.

---

### 🧪 Benchmark Problems

Implemented problems:

* **ZDT1** – Convex front
* **ZDT2** – Non-convex front
* **ZDT3** – Disconnected front

Each problem is solved using:

* 50 individuals
* 1000 generations
* 30 independent runs

---

### 📐 Evaluation Metrics

Implemented in `+kpi/`:

* Generational Distance (GD)
* Inverted GD (IGD)
* Diversity ($\Delta$)
* Hypervolume (HV), computed using:

  * PlatEMO method
  * Exact rectangle-based method (for 2D)

---

### 📚 Report

The full report (`Report/Documentation.pdf`) includes:

* Theoretical background
* Experimental setup
* Detailed metric definitions
* Tables and analysis of results
* Graphical comparisons

---

### 🧑‍💻 Author

* **Name**: Giuseppe Soriano
* **Institution**: University of Pisa
* **Course**: Symbolic and Evolutionary Artificial Intelligence (MSc AI & Data Engineering)
* **Supervisor**: Prof. Marco Cococcioni

---

### 📅 Academic Year

2024 / 2025