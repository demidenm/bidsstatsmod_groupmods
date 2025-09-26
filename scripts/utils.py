import os
import nbformat
import re
import json
import pandas as pd
import numpy as np
from IPython.display import display, Markdown
from bids.modeling import BIDSStatsModelsGraph
from bids.layout import BIDSLayout, BIDSLayoutIndexer


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
            <marker id="arrowhead" markerWidth="10" markerHeight="7" 
                    refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#333" />
            </marker>
        </defs>
    
        <!-- Run Level -->
        <text x="100" y="70" style="font: bold 14px Arial; text-anchor: middle;">Run Level</text>
        
        <!-- Subject 1 Runs -->
        <text x="100" y="90" style="font: 10px Arial; text-anchor: middle;">Subject 1</text>
        <rect x="50" y="100" width="80" height="40" style="fill: #e1f5fe; stroke: #0277bd; stroke-width: 2;" rx="5"/>
        <text x="90" y="115" style="font: 12px Arial; text-anchor: middle;">Run 1</text>
        <text x="90" y="130" style="font: 12px Arial; text-anchor: middle;">Contrast A</text>
        
        <rect x="50" y="150" width="80" height="40" style="fill: #e1f5fe; stroke: #0277bd; stroke-width: 2;" rx="5"/>
        <text x="90" y="165" style="font: 12px Arial; text-anchor: middle;">Run 2</text>
        <text x="90" y="180" style="font: 12px Arial; text-anchor: middle;">Contrast A</text>
        
        <!-- Subject 2 Runs -->
        <text x="100" y="210" style="font: 10px Arial; text-anchor: middle;">Subject 2</text>
        <rect x="50" y="220" width="80" height="40" style="fill: #e1f5fe; stroke: #0277bd; stroke-width: 2;" rx="5"/>
        <text x="90" y="235" style="font: 12px Arial; text-anchor: middle;">Run 1</text>
        <text x="90" y="250" style="font: 12px Arial; text-anchor: middle;">Contrast A</text>
        
        <rect x="50" y="270" width="80" height="40" style="fill: #e1f5fe; stroke: #0277bd; stroke-width: 2;" rx="5"/>
        <text x="90" y="285" style="font: 12px Arial; text-anchor: middle;">Run 2</text>
        <text x="90" y="300" style="font: 12px Arial; text-anchor: middle;">Contrast A</text>
        
        <!-- Subject N indicator -->
        <text x="90" y="330" style="font: 10px Arial; text-anchor: middle;">...</text>
        

        <!-- Subject Level -->
        <text x="300" y="70" style="font: bold 14px Arial; text-anchor: middle;">Subject Level</text>
        
        <rect x="250" y="130" width="100" height="50" style="fill: #f3e5f5; stroke: #7b1fa2; stroke-width: 2;" rx="5"/>
        <text x="300" y="148" style="font: 12px Arial; text-anchor: middle;">Subject 1</text>
        <text x="300" y="162" style="font: 12px Arial; text-anchor: middle;">Contrast A</text>
        <text x="300" y="172" style="font: 10px Arial; text-anchor: middle;">(avg runs)</text>
        
        <rect x="250" y="220" width="100" height="50" style="fill: #f3e5f5; stroke: #7b1fa2; stroke-width: 2;" rx="5"/>
        <text x="300" y="238" style="font: 12px Arial; text-anchor: middle;">Subject 2</text>
        <text x="300" y="252" style="font: 12px Arial; text-anchor: middle;">Contrast A</text>
        <text x="300" y="262" style="font: 10px Arial; text-anchor: middle;">(avg runs)</text>
        
        <text x="300" y="300" style="font: 10px Arial; text-anchor: middle;">...</text>
        
        <!-- Dataset Level -->
        <text x="550" y="70" style="font: bold 14px Arial; text-anchor: middle;">Dataset Level</text>
        
        <rect x="480" y="170" width="120" height="60" style="fill: #e8f5e8; stroke: #388e3c; stroke-width: 2;" rx="5"/>
        <text x="540" y="190" style="font: 12px Arial; text-anchor: middle;">One Sample</text>
        <text x="540" y="205" style="font: 12px Arial; text-anchor: middle;">Average</text>
        <text x="540" y="220" style="font: 12px Arial; text-anchor: middle;">Contrast A</text>
        
        <!-- Arrows from runs to subject level (fixed spacing) -->
        <line x1="130" y1="120" x2="250" y2="150" style="stroke: #333; stroke-width: 2; fill: none; marker-end: url(#arrowhead);"/>
        <line x1="130" y1="170" x2="250" y2="160" style="stroke: #333; stroke-width: 2; fill: none; marker-end: url(#arrowhead);"/>
        <line x1="130" y1="240" x2="250" y2="240" style="stroke: #333; stroke-width: 2; fill: none; marker-end: url(#arrowhead);"/>
        <line x1="130" y1="290" x2="250" y2="250" style="stroke: #333; stroke-width: 2; fill: none; marker-end: url(#arrowhead);"/>
        
        <!-- Arrows from subject to dataset level -->
        <line x1="350" y1="155" x2="480" y2="190" style="stroke: #333; stroke-width: 2; fill: none; marker-end: url(#arrowhead);"/>
        <line x1="350" y1="245" x2="480" y2="210" style="stroke: #333; stroke-width: 2; fill: none; marker-end: url(#arrowhead);"/>
        
        <!-- Alternative path for single run -->
        <text x="150" y="350" style="font: 10px Arial; fill: #666;">
            <tspan x="175" dy="0">If only 1 run per subject,</tspan>
            <tspan x="175" dy="10">subject level is skipped</tspan>
        </text>
        <path d="M 130 320 Q 300 380 480 220" style="stroke: #333; stroke-width: 2; fill: none; marker-end: url(#arrowhead); stroke-dasharray: 5,5;"/>
        
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
            <marker id="arrowhead" markerWidth="10" markerHeight="7" 
                    refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#333" />
            </marker>
        </defs>

        <!-- Run Level -->
        <text x="100" y="70" style="font: bold 14px Arial; text-anchor: middle;">Run Level</text>
        
        <!-- Group A Subjects -->
        <text x="100" y="90" style="font: 11px Arial; fill: #444;">Group A</text>
        <text x="100" y="105" style="font: 10px Arial; text-anchor: middle;">Subject 1</text>
        <rect x="50" y="115" width="60" height="35" style="fill: #e1f5fe; stroke: #0277bd; stroke-width: 2;" rx="3"/>
        <text x="80" y="130" style="font: 12px Arial; text-anchor: middle;">Run 1</text>
        <text x="80" y="140" style="font: 10px Arial; text-anchor: middle;">Contrast A</text>
        
        <rect x="120" y="115" width="60" height="35" style="fill: #e1f5fe; stroke: #0277bd; stroke-width: 2;" rx="3"/>
        <text x="150" y="130" style="font: 12px Arial; text-anchor: middle;">Run 2</text>
        <text x="150" y="140" style="font: 10px Arial; text-anchor: middle;">Contrast A</text>
        
        <text x="100" y="170" style="font: 10px Arial; text-anchor: middle;">Subject 2</text>
        <rect x="50" y="180" width="60" height="35" style="fill: #e1f5fe; stroke: #0277bd; stroke-width: 2;" rx="3"/>
        <text x="80" y="195" style="font: 12px Arial; text-anchor: middle;">Run 1</text>
        <text x="80" y="205" style="font: 10px Arial; text-anchor: middle;">Contrast A</text>
        
        <rect x="120" y="180" width="60" height="35" style="fill: #e1f5fe; stroke: #0277bd; stroke-width: 2;" rx="3"/>
        <text x="150" y="195" style="font: 12px Arial; text-anchor: middle;">Run 2</text>
        <text x="150" y="205" style="font: 10px Arial; text-anchor: middle;">Contrast A</text>
        
        <!-- Group B Subjects -->
        <text x="100" y="240" style="font: 11px Arial; fill: #444;">Group B</text>
        <text x="100" y="255" style="font: 10px Arial; text-anchor: middle;">Subject 3</text>
        <rect x="50" y="265" width="60" height="35" style="fill: #e1f5fe; stroke: #0277bd; stroke-width: 2;" rx="3"/>
        <text x="80" y="280" style="font: 12px Arial; text-anchor: middle;">Run 1</text>
        <text x="80" y="290" style="font: 10px Arial; text-anchor: middle;">Contrast A</text>
        
        <rect x="120" y="265" width="60" height="35" style="fill: #e1f5fe; stroke: #0277bd; stroke-width: 2;" rx="3"/>
        <text x="150" y="280" style="font: 12px Arial; text-anchor: middle;">Run 2</text>
        <text x="150" y="290" style="font: 10px Arial; text-anchor: middle;">Contrast A</text>
        
        <text x="100" y="320" style="font: 10px Arial; text-anchor: middle;">Subject 4</text>
        <rect x="50" y="330" width="60" height="35" style="fill: #e1f5fe; stroke: #0277bd; stroke-width: 2;" rx="3"/>
        <text x="80" y="345" style="font: 12px Arial; text-anchor: middle;">Run 1</text>
        <text x="80" y="355" style="font: 10px Arial; text-anchor: middle;">Contrast A</text>
        
        <rect x="120" y="330" width="60" height="35" style="fill: #e1f5fe; stroke: #0277bd; stroke-width: 2;" rx="3"/>
        <text x="150" y="345" style="font: 12px Arial; text-anchor: middle;">Run 2</text>
        <text x="150" y="355" style="font: 10px Arial; text-anchor: middle;">Contrast A</text>
        
        <!-- Subject Level -->
        <text x="290" y="70" style="font: bold 14px Arial; text-anchor: middle;">Subject Level</text>
        
        <!-- Group A Subject Level -->
        <text x="270" y="90" style="font: 11px Arial; fill: #444;">Group A</text>
        <rect x="250" y="115" width="80" height="40" style="fill: #f3e5f5; stroke: #7b1fa2; stroke-width: 2;" rx="5"/>
        <text x="290" y="130" style="font: 12px Arial; text-anchor: middle;">Subject 1</text>
        <text x="290" y="145" style="font: 12px Arial; text-anchor: middle;">Contrast A</text>
        
        <rect x="250" y="180" width="80" height="40" style="fill: #f3e5f5; stroke: #7b1fa2; stroke-width: 2;" rx="5"/>
        <text x="290" y="195" style="font: 12px Arial; text-anchor: middle;">Subject 2</text>
        <text x="290" y="210" style="font: 12px Arial; text-anchor: middle;">Contrast A</text>
        
        <!-- Group B Subject Level -->
        <text x="270" y="240" style="font: 11px Arial; fill: #444;">Group B</text>
        <rect x="250" y="265" width="80" height="40" style="fill: #f3e5f5; stroke: #7b1fa2; stroke-width: 2;" rx="5"/>
        <text x="290" y="280" style="font: 12px Arial; text-anchor: middle;">Subject 3</text>
        <text x="290" y="295" style="font: 12px Arial; text-anchor: middle;">Contrast A</text>
        
        <rect x="250" y="330" width="80" height="40" style="fill: #f3e5f5; stroke: #7b1fa2; stroke-width: 2;" rx="5"/>
        <text x="290" y="345" style="font: 12px Arial; text-anchor: middle;">Subject 4</text>
        <text x="290" y="360" style="font: 12px Arial; text-anchor: middle;">Contrast A</text>
        
        <!-- Arrows from runs to subjects -->
        <line x1="180" y1="132" x2="250" y2="135" style="stroke: #333; stroke-width: 2; fill: none; marker-end: url(#arrowhead);"/>
        <line x1="180" y1="197" x2="250" y2="200" style="stroke: #333; stroke-width: 2; fill: none; marker-end: url(#arrowhead);"/>
        <line x1="180" y1="282" x2="250" y2="285" style="stroke: #333; stroke-width: 2; fill: none; marker-end: url(#arrowhead);"/>
        <line x1="180" y1="347" x2="250" y2="350" style="stroke: #333; stroke-width: 2; fill: none; marker-end: url(#arrowhead);"/>
        
        <!-- Dataset Level (Stacked vertically) -->
        <text x="550" y="70" style="font: bold 14px Arial; text-anchor: middle;">Dataset Level</text>
        
        <!-- Dataset Level 1: One Sample Average (Top) -->
        <text x="550" y="90" style="font: 10px Arial; text-anchor: middle;">One Sample Average</text>
        
        <rect x="500" y="115" width="100" height="50" style="fill: #e8f5e8; stroke: #388e3c; stroke-width: 2;" rx="5"/>
        <text x="550" y="130" style="font: 12px Arial; text-anchor: middle;">Group A</text>
        <text x="550" y="145" style="font: 12px Arial; text-anchor: middle;">One Sample</text>
        <text x="550" y="155" style="font: 10px Arial; text-anchor: middle;">Contrast A</text>
        
        <rect x="500" y="180" width="100" height="50" style="fill: #e8f5e8; stroke: #388e3c; stroke-width: 2;" rx="5"/>
        <text x="550" y="198" style="font: 12px Arial; text-anchor: middle;">Group B</text>
        <text x="550" y="210" style="font: 12px Arial; text-anchor: middle;">One Sample</text>
        <text x="550" y="220" style="font: 10px Arial; text-anchor: middle;">Contrast A</text>
        
        <!-- Dataset Level 2: Between Group (Bottom) -->
        <text x="560" y="270" style="font: 10px Arial; text-anchor: middle;">Between Group Differences</text>
        
        <rect x="500" y="290" width="120" height="60" style="fill: #fff3e0; stroke: #f57c00; stroke-width: 2;" rx="5"/>
        <text x="560" y="310" style="font: 12px Arial; text-anchor: middle;">Group A vs B</text>
        <text x="560" y="325" style="font: 12px Arial; text-anchor: middle;">Between Group</text>
        <text x="560" y="340" style="font: 12px Arial; text-anchor: middle;">Contrast A</text>
        
        <!-- Clean arrows from subject to both dataset levels -->
        <!-- To Dataset Level 1 (One Sample) -->
        <line x1="330" y1="135" x2="500" y2="140" style="stroke: #333; stroke-width: 2; fill: none; marker-end: url(#arrowhead);"/>
        <line x1="330" y1="200" x2="500" y2="140" style="stroke: #333; stroke-width: 2; fill: none; marker-end: url(#arrowhead);"/>
        <line x1="330" y1="285" x2="500" y2="205" style="stroke: #333; stroke-width: 2; fill: none; marker-end: url(#arrowhead);"/>
        <line x1="330" y1="350" x2="500" y2="205" style="stroke: #333; stroke-width: 2; fill: none; marker-end: url(#arrowhead);"/>
        
        <!-- To Dataset Level 2 (Between Group) -->
        <line x1="330" y1="135" x2="500" y2="310" style="stroke: #333; stroke-width: 2; fill: none; marker-end: url(#arrowhead);"/>
        <line x1="330" y1="200" x2="500" y2="315" style="stroke: #333; stroke-width: 2; fill: none; marker-end: url(#arrowhead);"/>
        <line x1="330" y1="285" x2="500" y2="325" style="stroke: #333; stroke-width: 2; fill: none; marker-end: url(#arrowhead);"/>
        <line x1="330" y1="350" x2="500" y2="330" style="stroke: #333; stroke-width: 2; fill: none; marker-end: url(#arrowhead);"/>
        
        <!-- Alternative path notation -->
        <text x="275" y="450" style="font: 10px Arial; fill: #666;">
            <tspan x="275" dy="0">If only 1 run per subject,</tspan>
            <tspan x="275" dy="10">subject level is skipped</tspan>
        </text>
        <path d="M 180 390 Q 340 430 500 230" style="stroke: #333; stroke-width: 2; fill: none; marker-end: url(#arrowhead); stroke-dasharray: 5,5;"/>
        <path d="M 180 390 Q 330 480 480 340" style="stroke: #333; stroke-width: 2; fill: none; marker-end: url(#arrowhead); stroke-dasharray: 5,5;"/>
        
    </svg>
    '''
    
    return svg_content