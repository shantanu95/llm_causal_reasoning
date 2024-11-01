The `narratives` directory contain the semi-synthetic and real-world narratives (in the forward and reverse direction) and the corresponding causal chain graphs we use in our experiments.

The following two notebooks contain the code for generating the plots for experiments
with semi-synthetic and real-world data:
 - `cause_effect_pair_estimation.ipynb`: Contains code for the direct and CoT based prompts.
 - `graph_estimation.ipynb`: Contains code for the estimating the chain graph based on the given narrative.

 In both notebooks, the `NARRATIVE_DIR` variable needs to be set to point to a directory inside the `narratives` folder.