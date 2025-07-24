# BIDS Statistical Models for Group-Level fMRI Analysis

This repository demonstrates how to use BIDS Statistical Models (BIDS-SM) to perform hierarchical fMRI analyses, from run-level GLMs to group-level mass univariate analyses. The workflow covers various (six) statistical models with examples, including:

- one-sample t-tests
- two-sample t-tests
- covariate analyses
- interaction models 
- ANOVA designs 

These examples are comparable to those listed for [FSL](https://web.mit.edu/fsl_v5.0.10/fsl/doc/wiki/GLM.html). However, BIDS-SM faciliates running these models across multiple software AFNI (3dREMLfit) and Nilearn (SPM GLM) [Coming soon: FSL]

## Overview

Functional MRI analysis involves a three-level hierarchical approach:
1. **Run-level (First-level)**: voxelwise GLM analysis applied to individual fMRI runs for *each* subject
2. **Subject-level (Second-level)**: Averaging or contrasting across runs *within* subjects
3. **Group-level (Third-level)**: Statistical inference *across* or *between* subjects

This repository provides a complete pipeline using [BIDS Statistical Models (BIDS-SM)](https://bids-standard.github.io/stats-models/) to perform these analyses. Data were downloaded using [OpenNeuro Fitlins GLM](https://github.com/demidenm/openneuro_glmfitlins)

## Dataset

The analyses use OpenNeuro dataset **[ds000171](https://openneuro.org/datasets/ds000171)**. 
The dataset includes:
- **Task**: Music listening paradigm (positive music, negative music, non-musical tones)
- **Participants**: 39 subjects (20 never-depressed control participants (ND), 19 with Major Depressive Disorder (MDD))
- **Design**: 3 runs per subject with a wide age range (18-59 years)
- **Contrasts**: Multiple task contrasts including music vs. sounds comparisons, which is the focus on most examples here.

> This dataset was chosen for no other reason besides it had the run-, subject- and group-level information to faciliated numerous model estimating procedures. These are to serve as an example and not a recommendation for these data.

## Repository Structure

```
├── notebooks/
│   └── model_illustrations.ipynb    # analysis notebook with information and illustrations re: BIDS-SM
├── scripts/
│   ├── sbatch_fitlins/
│   │   └── submit_fitlins_mods.sh    # SLURM submission script to run fitlins models
│   ├── model_specs/
│   │   ├── base_spec.json           # Base model specification
│   │   └── group_models.json        # Group-level model definitions, iteratively grab specs in w/ submission
│   └── utils.py                     # Utility functions, used in notebook
├── path_config.json                 # Configuration file, update paths to rerun
└── README.md                       
```

## Statistical Models Implemented

### 1. One-Sample T-Test
Tests whether brain activation is significantly different from zero across all subjects.
- **Model**: `Y = β₀ + ε`
- **Use case**: Basic activation maps

### 2. One-Sample T-Test with Covariate
Controls for age effects while testing for significant activation.
- **Model**: `Y = β₀ + β₁(age_centered) + ε`
- **Use case**: Activation maps controlling for demographic variables (e.g. age)

### 3. Two-Sample T-Test
Compares brain activation between two groups (Controls vs. MDD).
- **Model**: `Y = β₀ + β₁(group_control) + β₂(group_mdd) + ε`
- **Use case**: Group differences in activation

### 4. Two-Sample T-Test with Covariate
Group comparison while controlling for age.
- **Model**: `Y = β₀ + β₁(group_control) + β₂(group_mdd) + β₃(age_centered) + ε`
- **Use case**: Group differences controlling for confounds (e.g. age)

### 5. Group × Continuous Interaction
Tests whether the relationship between age and brain activation differs between groups (e.g., comparing slopes of age).
- **Model**: `Y = β₀ + β₁(group_control) + β₂(group_mdd) + β₃(age_control) + β₄(age_mdd) + ε`
- **Use case**: Differential age effects between groups

### 6. Three-Group ANOVA (omnibus test of 2+ groups)
Compares activation across three age groups (younger, middle, older).
- **Model**: `Y = β₀ + β₁(age_young) + β₂(age_middle) + β₃(age_older) + ε`
- **Use case**: Multiple group comparisons with F-test to control Type-I error rate

### 7. Single-group paired difference t-test (e.g. between-run changes)

*Coming soon.*

## Setup & Usage

### Installation

1. **Clone the repository**:

```bash
git clone https://github.com/demidenm/bidsstatsmod_groupmods.git
cd bidsstatsmod_groupmods
```

2. **Set up environment**:

```bash
bash setup_uv.sh
```

3. **Configure paths**:

Edit `path_config.json` to match your data directories:
```json
{
  "datasets_folder": "/path/to/your/data",
  "openneuro_glmrepo": "/path/to/this/repo",
  "tmp_folder": "/path/to/scratch/space"
}
```

## Usage

### Interactive Analysis
Open and run the Jupyter notebook for step-by-step analysis:
```bash
jupyter lab notebooks/model_illustrations.ipynb
```

The notebook demonstrates:
- Data inspection and validation
- Model specification setup
- Transformation verification
- Design matrix visualization
- Statistical map generation

### Batch Processing
For large-scale analyses, use the SLURM submission script:

```bash
# Submit one-sample t-test
sbatch scripts/sbatch_fitlins/submit_fitlins_mods.sh ds000171 one_sample_ttest

# Submit two-sample t-test with covariate
sbatch scripts/sbatch_fitlins/submit_fitlins_mods.sh ds000171 two_sample_ttest_covage

# Submit three-group ANOVA  
sbatch scripts/sbatch_fitlins/submit_fitlins_mods.sh ds000171 anova_3grp
```

Available model types:
- `one_sample_ttest`
- `one_sample_ttest_covage`
- `two_sample_ttest`
- `two_sample_ttest_covage`
- `two_sample_ttest_covinteract`
- `anova_3grp`

## Model Output

The pipeline generates:
- Statistical maps (z-stat, t-stat, f-stat)
- Parameter estimate maps (effect sizes)
- Design matrices and model summaries
- Quality control plots and metrics

Results are saved in BIDS-compatible format:
```
analyses/bidssm_ds000171/task-music/
├── node-runLevel/
├── node-subjectLevel/
[group-level reports:]
├── node-onesampleT/
│   ├── contrast-musicvsounds_stat-z_statmap.nii.gz
│   └── contrast-musicvsounds_stat-effect_statmap.nii.gz
├── node-twosampleT/
└── node-anova3grp/
```

## Citation

Coming soon

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenNeuro for providing open datasets
- BIDS community for standardization efforts  
- FitLins developers for the analysis engine
- Stanford Research Computing (Sherlock) for computational resources