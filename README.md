Implementations of halving and aliquoting distance algorithms in Python 3. The minimum dependencies required are `numpy`, `pandas`, `matplotlib`, and `seaborn`

* `generate_data.ipynb`: Jupyter notebook that generates simulated profiles and evaluates the halving and aliquoting algorithms
* `analysis.ipynb`: Jupyter notebook that generates figures
* `outputs.pkl`: pickled pandas dataframe containing our results of experiments (generated with `generate_data.ipynb`)
* `aliquoting.py`: implementation of halving and aliquoting algorithms
* `distance.py`: implementation of copy number distance
* `CopyNumberDistanceFunctions.py`: other copy number distance-related functions, not used in our implementation or analysis; these functions require Gurobi and other dependencies
