Implementations of halving and aliquoting distance algorithms in Python 3. The minimum dependencies required are `numpy`, `pandas`, `matplotlib`, and `seaborn`.

* `generate_data.ipynb`: Jupyter notebook that generates simulated profiles and evaluates the halving and aliquoting algorithms
* `analysis.ipynb`: Jupyter notebook that generates figures used in our paper
* `outputs.pkl`: pickled pandas dataframe containing our results of experiments as shown in our paper (generated with `generate_data.ipynb`)
* `aliquoting.py`: implementation of halving and aliquoting algorithms
* `distance.py`: implementation of copy number distance
* `CopyNumberDistanceFunctions.py`: other copy number distance-related functions, not used in our implementation or analysis; these functions require Gurobi and other dependencies

# Example
To find the halving and aliquoting distances of a profile `T`:
``` python
from aliquoting import cnd_halving, cnd_aliquoting_I

T = [1,2,3,4,5]
p = 3
halving_distance, halving_predup_profile = cnd_halving(T)[:2]
aliquoting_distance, aliquoting_predup_profile = cnd_aliquoting_I(T, p)[:2]
```
