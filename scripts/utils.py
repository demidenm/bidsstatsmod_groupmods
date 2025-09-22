import os
import nbformat
import re
import json
import pandas as pd
import numpy as np
from IPython.display import display, Markdown
from bids.modeling import BIDSStatsModelsGraph
from bids.layout import BIDSLayout, BIDSLayoutIndexer
from nilearn.interfaces.bids import parse_bids_filename
from nilearn.image import index_img, load_img, new_img_like, mean_img
from nilearn.glm import expression_to_contrast_vector
from pyrelimri import similarity


def get_numvolumes(nifti_path_4d):
    """
    Alternative method to get number of volumes using Nilearn.
    
    Parameters:
    nifti_path(str) : Path to the fMRI NIfTI (.nii.gz) file
    
    Returns:
    Number of volumes in the fMRI data using nilearn image + shape
    """
    try:
        # Load 4D image
        img = load_img(nifti_path_4d)
        
        # Get number of volumes
        return img.shape[3] if len(img.shape) == 4 else None
    
    except Exception as e:
        print(f"Nilearn error reading file {nifti_path_4d}: {e}")
        return None


def generate_tablecontents(notebook_name):
    """Generate a Table of Contents from markdown headers in the current Jupyter Notebook."""
    toc = ["# Table of Contents\n"]

    # Get the current notebook name dynamically
    notebook_path = os.getcwd()
    notebook_file = os.path.join(notebook_path, notebook_name)
    
    if not notebook_file:
        print("No notebook file found in the directory.")
        return
    
    # Load the notebook content
    with open(notebook_file, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    for cell in notebook.cells:
        if cell.cell_type == "markdown":  # Only process markdown cells
            lines = cell.source.split("\n")
            for line in lines:
                match = re.match(r"^(#+)\s+([\d.]+)?\s*(.*)", line)  # Match headers with optional numbering
                if match:
                    level = len(match.group(1))  # Number of `#` determines header level
                    header_number = match.group(2) or ""  # Capture section number if present
                    header_text = match.group(3).strip()  # Extract actual text
                    
                    # Format the anchor link correctly for Jupyter:
                    # 1. Keep original casing
                    # 2. Preserve periods
                    # 3. Replace spaces with hyphens
                    # 4. Remove special characters except `.` and `-`
                    anchor = f"{header_number} {header_text}"
                    anchor = anchor.replace(" ", "-")  # Convert spaces to hyphens
                    anchor = re.sub(r"[^\w.\-]", "", anchor)  # Remove special characters except `.` and `-`

                    toc.append(f"{'  ' * (level - 1)}- [{header_number} {header_text}](#{anchor})")

    # diplay table of contents in markdown
    display(Markdown("\n".join(toc)))


def get_bidstats_events(bids_path, spec_cont, scan_length=125, ignored=None, return_events_num=0):
    """
    Initializes a BIDS layout, processes a BIDSStatsModelsGraph, 
    and returns a DataFrame of the first collection's entities.

    Parameters:
    - bids_inp (str): Path to the BIDS dataset.
    - spec_cont: Specification content for BIDSStatsModelsGraph.
    - scan_length (int, optional): Scan length parameter for load_collections. Default is 125.
    - ignored (list, optional): List of regex patterns to ignore during indexing.
    - return_events_num (int, optional): Number of events to return. Default is 0.

    Returns:
    - DataFrame: Data representation of the first collection with entities, or None if errors occur.
    """
    try:
        indexer = BIDSLayoutIndexer(ignore=ignored) if ignored else BIDSLayoutIndexer()
    except Exception as e:
        print(f"Error initializing BIDSLayoutIndexer: {e}")
        return None

    try:
        bids_layout = BIDSLayout(root=bids_path, reset_database=True, indexer=indexer)
    except Exception as e:
        print(f"Error initializing BIDSLayout: {e}")
        return None

    try:
        graph = BIDSStatsModelsGraph(bids_layout, spec_cont)
        graph.load_collections(scan_length=scan_length)
    except Exception as e:
        print(f"Error creating or loading BIDSStatsModelsGraph: {e}")
        return None

    try:
        root_node = graph.root_node
        colls = root_node.get_collections()
        if not colls:
            raise ValueError("No collections found in the root node.")
        return colls[return_events_num].to_df(entities=True), root_node, graph
    except Exception as e:
        print(f"Error processing root node collections: {e}")
        return None


def extract_model_info(model_spec):
    """
    Extracts subject numbers, node levels, convolve model type, regressors, and contrasts from a BIDS model specification,
    and multiplies each condition by its corresponding weight.

    Parameters:
    model_spec (dict): The BIDS model specification dictionary.

    Returns:
    dict: A dictionary containing the extracted information, including weighted conditions.
    """

    extracted_info = {
        "subjects": model_spec.get("Input", {}).get("subject", []),
        "nodes": [],
    }

    for node in model_spec.get("Nodes", []):
        node_info = {
            "level": node.get("Level"),
            "regressors": node.get("Model", {}).get("X", []),
            "contrasts": [
                {
                    "name": contrast.get("Name", "Unnamed Contrast"),
                    "conditions": contrast.get("ConditionList", []),
                    "weights": contrast.get("Weights", []),
                    "test": contrast.get("Test", "t")  # Default test type to "t"
                }
                for contrast in node.get("Contrasts", [])
            ],
            "convolve_model": "spm",  # Default value spm
            "convolve_inputs": [],            # Obtain regressors that were convolved
            "if_derivative_hrf": False,    # Track if HRF derivative is used
            "if_dispersion_hrf": False,    # Track if HRF dispersion is used
            "target_var": [] # Track values receiving an assignment duration (e.g. not parametric)
        }

        # Extract HRF convolution model type and derivative status
        transformations = node.get("Transformations", {}).get("Instructions", [])
        for instruction in transformations:
            if instruction.get("Name") == "Convolve":
                node_info["convolve_model"] = instruction.get("Model", "Unknown")
                node_info["if_derivative_hrf"] = instruction.get("Derivative", False) == True
                node_info["if_dispersion_hrf"] = instruction.get("Dispersion", False) == True
                node_info["convolve_inputs"] = instruction.get("Input", [])
                break  # Stop searching after finding first Convolve transformation
            
            if instruction.get("Name") == "Assign" and instruction.get("TargetAttr") == "duration":
                targets = instruction.get("Target", [])
                if isinstance(targets, str):
                    targets = [targets]
                node_info["target_var"].extend(targets)

        extracted_info["nodes"].append(node_info)
    
    return extracted_info



# below est_contrast_vifs code is courtsey of Jeanette Mumford's repo: https://github.com/jmumford/vif_contrasts
def est_contrast_vifs(desmat, contrasts):
    """
    IMPORTANT: This is only valid to use on design matrices where each regressor represents a condition vs baseline
     or if a parametrically modulated regressor is used the modulator must have more than 2 levels.  If it is a 2 level modulation,
     split the modulation into two regressors instead.

    Calculates VIF for contrasts based on the ratio of the contrast variance estimate using the
    true design to the variance estimate where between condition correaltions are set to 0
    desmat : pandas DataFrame, design matrix
    contrasts : dictionary of contrasts, key=contrast name,  using the desmat column names to express the contrasts
    returns: pandas DataFrame with VIFs for each contrast
    """
    desmat_copy = desmat.copy()
    # find location of constant regressor and remove those columns (not needed here)
    desmat_copy = desmat_copy.loc[
        :, (desmat_copy.nunique() > 1) | (desmat_copy.isnull().any())
    ]
    # Scaling stabilizes the matrix inversion
    nsamp = desmat_copy.shape[0]
    desmat_copy = (desmat_copy - desmat_copy.mean()) / (
        (nsamp - 1) ** 0.5 * desmat_copy.std()
    )
    vifs_contrasts = {}
    for contrast_name, contrast_string in contrasts.items():
        try:
            contrast_cvec = expression_to_contrast_vector(
                contrast_string, desmat_copy.columns
            )
            true_var_contrast = (
                contrast_cvec
                @ np.linalg.inv(desmat_copy.transpose() @ desmat_copy)
                @ contrast_cvec.transpose()
            )
            # The folllowing is the "best case" scenario because the between condition regressor correlations are set to 0
            best_var_contrast = (
                contrast_cvec
                @ np.linalg.inv(
                    np.multiply(
                        desmat_copy.transpose() @ desmat_copy,
                        np.identity(desmat_copy.shape[1]),
                    )
                )
                @ contrast_cvec.transpose()
            )
            vifs_contrasts[contrast_name] = true_var_contrast / best_var_contrast
        except Exception as e:
            print(f"Error computing VIF for regressor '{contrast_name}': {e}")

    return vifs_contrasts


def gen_vifdf(designmat, contrastdict, nuisance_regressors):
    """
    Create a Pandas DataFrame with VIF values for contrasts and regressors.

    Parameters
    designmat: The design matrix used in the analysis.
    modconfig (dict): A dictionary containing model configuration, including:
        - 'nuisance_regressors': A regex pattern to filter out nuisance regressors.
           - 'contrasts': A dictionary of contrast definitions.

    Returns
    Returns contrasts & regressors vif dict & DataFrame of combined VIFs w/ columns ['type', 'name', 'value'].
    """
    filtered_columns = designmat.columns[~designmat.columns.isin(nuisance_regressors)]
    regressor_dict = {item: item for item in filtered_columns if item != "intercept"}

    
    # est VIFs for contrasts and regressors
    con_vifs = est_contrast_vifs(desmat=designmat, contrasts=contrastdict)
    reg_vifs = est_contrast_vifs(desmat=designmat, contrasts=regressor_dict)

    # convert to do
    df_con = pd.DataFrame(list(con_vifs.items()), columns=["name", "value"])
    df_con["type"] = "contrast"
    df_reg = pd.DataFrame(list(reg_vifs.items()), columns=["name", "value"])
    df_reg["type"] = "regressor"

    # combine & rename cols
    df = pd.concat([df_con, df_reg], ignore_index=True)
    df = df[["type", "name", "value"]]

    return con_vifs,reg_vifs,df


# functions for subjects and contrasts generic files
def create_subjects_json(subj_list, studyid, taskname, specpath):
    subjects_file_path = os.path.join(specpath, f'{studyid}-{taskname}_subjects.json')
    subjects_data = {
        "Subjects": subj_list
    }
    with open(subjects_file_path, 'w') as f:
        json.dump(subjects_data, f, indent=4)
    print(f"\t\tSaved subjects file for task {taskname} to {subjects_file_path}")

def create_gencontrast_json(studyid, taskname, specpath):
    contrasts_file_path = os.path.join(specpath, f'{studyid}-{taskname}_contrasts.json')
    contrasts_data = {
        "Contrasts": [
            {
                "Name": "AvB",
                "ConditionList": ["trial_type.a", "trial_type.b"],
                "Weights": [1, -1],
                "Test": "t"
            },
            {
                "Name": "FacesvPlaces",
                "ConditionList": ["trial_type.faces", "trial_type.places"],
                "Weights": [1, -1],
                "Test": "t"
            }
        ]
    }
    with open(contrasts_file_path, 'w') as f:
        json.dump(contrasts_data, f, indent=4)
    print(f"\t\tSaved contrasts file for task {taskname} to {contrasts_file_path}")


def calc_niftis_meanstd(path_imgs):
    """
    Calculate the mean & standard deviation across a list of NIfTI images.

    Parameters:
    - images_paths: List of paths to NIfTI images.

    Returns:
    NIfTI Image for mean and std in position 1 , 2
    """
    # if path_imgs list is empty
    assert len(path_imgs) > 0, "Error: The list of image paths is empty."

    # load ref image to obtain header info
    reference_img = load_img(path_imgs[0])
    
    # load / stack image data
    loaded_images = [load_img(img_path).get_fdata() for img_path in path_imgs]
    img_data_array = np.array(loaded_images)

    # data array is not empty and has more than 1 image
    assert img_data_array.size > 0, "Error: No valid image data were loaded."
    assert img_data_array.shape[0] > 1, "Error: At least two images are required for mean and std calculation."

    # calculate the mean and std across images (axis=0)
    mean_imgs = np.mean(img_data_array, axis=0)
    std_imgs = np.std(img_data_array, axis=0, ddof=1)  # ddof=1 for sample std

    # Create NIfTI images for the mean and std
    mean_nifti = new_img_like(reference_img, mean_imgs)
    std_nifti = new_img_like(reference_img, std_imgs)

    return mean_nifti, std_nifti

def pull_contrast_conditions_spec(spec_content):
    """
    get unique condition list values from spec content
    """
    condition_vals = set()
    
    # At node with contrasts, get condition lists
    for node in spec_content.get("Nodes", []):
        for contrast in node.get("Contrasts", []):
            condition_list = contrast.get("ConditionList", [])

            for condition in condition_list:
                condition_vals.add(condition)
                
    return sorted(list(condition_vals))


# FIGURES DIAGRAM SVG CREATED VIA CLAUDE ANDE MANUALLY MODIFIED
def create_bids_workflow_figure1():
    """
    Creates SVG for Figure 1: Basic BIDS Stats Model Workflow
    runLevel -> subjectLevel (if >1 run) -> datasetLevel (one sample average)
    Left to right flow, fixed arrows
    """
    
    svg_content = '''
    <svg width="800" height="500" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <style>
                .title { font: bold 16px Arial; text-anchor: middle; }
                .level-title { font: bold 14px Arial; text-anchor: middle; }
                .box-text { font: 12px Arial; text-anchor: middle; }
                .small-text { font: 10px Arial; text-anchor: middle; }
                .run-box { fill: #e1f5fe; stroke: #0277bd; stroke-width: 2; }
                .subject-box { fill: #f3e5f5; stroke: #7b1fa2; stroke-width: 2; }
                .dataset-box { fill: #e8f5e8; stroke: #388e3c; stroke-width: 2; }
                .arrow { stroke: #333; stroke-width: 2; fill: none; marker-end: url(#arrowhead); }
                .condition-text { font: 10px Arial; fill: #666; }
            </style>
            <marker id="arrowhead" markerWidth="10" markerHeight="7" 
                    refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#333" />
            </marker>
        </defs>
    
        <!-- Run Level -->
        <text x="100" y="70" class="level-title">Run Level</text>
        
        <!-- Subject 1 Runs -->
        <text x="100" y="90" class="small-text">Subject 1</text>
        <rect x="50" y="100" width="80" height="40" class="run-box" rx="5"/>
        <text x="90" y="115" class="box-text">Run 1</text>
        <text x="90" y="130" class="box-text">Contrast A</text>
        
        <rect x="50" y="150" width="80" height="40" class="run-box" rx="5"/>
        <text x="90" y="165" class="box-text">Run 2</text>
        <text x="90" y="180" class="box-text">Contrast A</text>
        
        <!-- Subject 2 Runs -->
        <text x="100" y="210" class="small-text">Subject 2</text>
        <rect x="50" y="220" width="80" height="40" class="run-box" rx="5"/>
        <text x="90" y="235" class="box-text">Run 1</text>
        <text x="90" y="250" class="box-text">Contrast A</text>
        
        <rect x="50" y="270" width="80" height="40" class="run-box" rx="5"/>
        <text x="90" y="285" class="box-text">Run 2</text>
        <text x="90" y="300" class="box-text">Contrast A</text>
        
        <!-- Subject N indicator -->
        <text x="90" y="330" class="small-text">...</text>
        

        <!-- Subject Level -->
        <text x="300" y="70" class="level-title">Subject Level</text>
        
        <rect x="250" y="130" width="100" height="50" class="subject-box" rx="5"/>
        <text x="300" y="150" class="box-text">Subject 1</text>
        <text x="300" y="165" class="box-text">Contrast A</text>
        <text x="300" y="175" class="small-text">(avg runs)</text>
        
        <rect x="250" y="220" width="100" height="50" class="subject-box" rx="5"/>
        <text x="300" y="240" class="box-text">Subject 2</text>
        <text x="300" y="255" class="box-text">Contrast A</text>
        <text x="300" y="265" class="small-text">(avg runs)</text>
        
        <text x="300" y="300" class="small-text">...</text>
        
        <!-- Dataset Level -->
        <text x="550" y="70" class="level-title">Dataset Level</text>
        
        <rect x="480" y="170" width="120" height="60" class="dataset-box" rx="5"/>
        <text x="540" y="190" class="box-text">One Sample</text>
        <text x="540" y="205" class="box-text">Average</text>
        <text x="540" y="220" class="box-text">Contrast A</text>
        
        <!-- Arrows from runs to subject level (fixed spacing) -->
        <line x1="130" y1="120" x2="250" y2="150" class="arrow"/>
        <line x1="130" y1="170" x2="250" y2="160" class="arrow"/>
        <line x1="130" y1="240" x2="250" y2="240" class="arrow"/>
        <line x1="130" y1="290" x2="250" y2="250" class="arrow"/>
        
        <!-- Arrows from subject to dataset level -->
        <line x1="350" y1="155" x2="480" y2="190" class="arrow"/>
        <line x1="350" y1="245" x2="480" y2="210" class="arrow"/>
        
        <!-- Alternative path for single run -->
        <text x="150" y="350" class="condition-text">
            <tspan x="175" dy="0">If only 1 run per subject,</tspan>
            <tspan x="175" dy="20">subject level is skipped</tspan>
        </text>
        <path d="M 130 320 Q 300 380 480 220" class="arrow" stroke-dasharray="5,5"/>
        
    </svg>
    '''
    
    return svg_content

def create_bids_workflow_figure2():
    """
    Creates SVG for Figure 2: Extended BIDS Stats Model Workflow
    Run/Subject level feeds into BOTH dataset levels independently
    Dataset levels stacked vertically to reduce arrow overlap
    """
    
    svg_content = '''
    <svg width="800" height="700" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <style>
                .title { font: bold 16px Arial; text-anchor: middle; }
                .level-title { font: bold 14px Arial; text-anchor: middle; }
                .box-text { font: 12px Arial; text-anchor: middle; }
                .small-text { font: 10px Arial; text-anchor: middle; }
                .run-box { fill: #e1f5fe; stroke: #0277bd; stroke-width: 2; }
                .subject-box { fill: #f3e5f5; stroke: #7b1fa2; stroke-width: 2; }
                .dataset-box1 { fill: #e8f5e8; stroke: #388e3c; stroke-width: 2; }
                .dataset-box2 { fill: #fff3e0; stroke: #f57c00; stroke-width: 2; }
                .arrow { stroke: #333; stroke-width: 2; fill: none; marker-end: url(#arrowhead); }
                .condition-text { font: 10px Arial; fill: #666; }
                .group-text { font: 11px Arial; fill: #444; }
            </style>
            <marker id="arrowhead" markerWidth="10" markerHeight="7" 
                    refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#333" />
            </marker>
        </defs>

        <!-- Run Level -->
        <text x="100" y="70" class="level-title">Run Level</text>
        
        <!-- Group A Subjects -->
        <text x="100" y="90" class="group-text">Group A</text>
        <text x="100" y="105" class="small-text">Subject 1</text>
        <rect x="50" y="115" width="60" height="35" class="run-box" rx="3"/>
        <text x="80" y="130" class="box-text">Run 1</text>
        <text x="80" y="140" class="small-text">Contrast A</text>
        
        <rect x="120" y="115" width="60" height="35" class="run-box" rx="3"/>
        <text x="150" y="130" class="box-text">Run 2</text>
        <text x="150" y="140" class="small-text">Contrast A</text>
        
        <text x="100" y="170" class="small-text">Subject 2</text>
        <rect x="50" y="180" width="60" height="35" class="run-box" rx="3"/>
        <text x="80" y="195" class="box-text">Run 1</text>
        <text x="80" y="205" class="small-text">Contrast A</text>
        
        <rect x="120" y="180" width="60" height="35" class="run-box" rx="3"/>
        <text x="150" y="195" class="box-text">Run 2</text>
        <text x="150" y="205" class="small-text">Contrast A</text>
        
        <!-- Group B Subjects -->
        <text x="100" y="240" class="group-text">Group B</text>
        <text x="100" y="255" class="small-text">Subject 3</text>
        <rect x="50" y="265" width="60" height="35" class="run-box" rx="3"/>
        <text x="80" y="280" class="box-text">Run 1</text>
        <text x="80" y="290" class="small-text">Contrast A</text>
        
        <rect x="120" y="265" width="60" height="35" class="run-box" rx="3"/>
        <text x="150" y="280" class="box-text">Run 2</text>
        <text x="150" y="290" class="small-text">Contrast A</text>
        
        <text x="100" y="320" class="small-text">Subject 4</text>
        <rect x="50" y="330" width="60" height="35" class="run-box" rx="3"/>
        <text x="80" y="345" class="box-text">Run 1</text>
        <text x="80" y="355" class="small-text">Contrast A</text>
        
        <rect x="120" y="330" width="60" height="35" class="run-box" rx="3"/>
        <text x="150" y="345" class="box-text">Run 2</text>
        <text x="150" y="355" class="small-text">Contrast A</text>
        
        <!-- Subject Level -->
        <text x="300" y="70" class="level-title">Subject Level</text>
        
        <!-- Group A Subject Level -->
        <text x="300" y="90" class="group-text">Group A</text>
        <rect x="250" y="115" width="80" height="40" class="subject-box" rx="5"/>
        <text x="290" y="130" class="box-text">Subject 1</text>
        <text x="290" y="145" class="box-text">Contrast A</text>
        
        <rect x="250" y="180" width="80" height="40" class="subject-box" rx="5"/>
        <text x="290" y="195" class="box-text">Subject 2</text>
        <text x="290" y="210" class="box-text">Contrast A</text>
        
        <!-- Group B Subject Level -->
        <text x="300" y="240" class="group-text">Group B</text>
        <rect x="250" y="265" width="80" height="40" class="subject-box" rx="5"/>
        <text x="290" y="280" class="box-text">Subject 3</text>
        <text x="290" y="295" class="box-text">Contrast A</text>
        
        <rect x="250" y="330" width="80" height="40" class="subject-box" rx="5"/>
        <text x="290" y="345" class="box-text">Subject 4</text>
        <text x="290" y="360" class="box-text">Contrast A</text>
        
        <!-- Arrows from runs to subjects -->
        <line x1="180" y1="132" x2="250" y2="135" class="arrow"/>
        <line x1="180" y1="197" x2="250" y2="200" class="arrow"/>
        <line x1="180" y1="282" x2="250" y2="285" class="arrow"/>
        <line x1="180" y1="347" x2="250" y2="350" class="arrow"/>
        
        <!-- Dataset Level (Stacked vertically) -->
        <text x="550" y="70" class="level-title">Dataset Level</text>
        
        <!-- Dataset Level 1: One Sample Average (Top) -->
        <text x="550" y="90" class="small-text">One Sample Average</text>
        
        <rect x="500" y="115" width="100" height="50" class="dataset-box1" rx="5"/>
        <text x="550" y="135" class="box-text">Group A</text>
        <text x="550" y="150" class="box-text">One Sample</text>
        <text x="550" y="160" class="small-text">Contrast A</text>
        
        <rect x="500" y="180" width="100" height="50" class="dataset-box1" rx="5"/>
        <text x="550" y="200" class="box-text">Group B</text>
        <text x="550" y="215" class="box-text">One Sample</text>
        <text x="550" y="225" class="small-text">Contrast A</text>
        
        <!-- Dataset Level 2: Between Group (Bottom) -->
        <text x="560" y="270" class="small-text">Between Group Differences</text>
        
        <rect x="500" y="290" width="120" height="60" class="dataset-box2" rx="5"/>
        <text x="560" y="310" class="box-text">Group A vs B</text>
        <text x="560" y="325" class="box-text">Between Group</text>
        <text x="560" y="340" class="box-text">Contrast A</text>
        
        <!-- Clean arrows from subject to both dataset levels -->
        <!-- To Dataset Level 1 (One Sample) -->
        <line x1="330" y1="135" x2="500" y2="140" class="arrow"/>
        <line x1="330" y1="200" x2="500" y2="140" class="arrow"/>
        <line x1="330" y1="285" x2="500" y2="205" class="arrow"/>
        <line x1="330" y1="350" x2="500" y2="205" class="arrow"/>
        
        <!-- To Dataset Level 2 (Between Group) -->
        <line x1="330" y1="135" x2="500" y2="310" class="arrow"/>
        <line x1="330" y1="200" x2="500" y2="315" class="arrow"/>
        <line x1="330" y1="285" x2="500" y2="325" class="arrow"/>
        <line x1="330" y1="350" x2="500" y2="330" class="arrow"/>
        
        <!-- Alternative path notation -->
        <text x="275" y="450" class="condition-text">
            <tspan x="275" dy="0">If only 1 run per subject,</tspan>
            <tspan x="275" dy="20">subject level is skipped</tspan>
        </text>
        <path d="M 180 390 Q 340 430 500 230" class="arrow" stroke-dasharray="5,5"/>
        <path d="M 180 390 Q 330 480 480 340" class="arrow" stroke-dasharray="5,5"/>
        
    </svg>
    '''
    
    return svg_content