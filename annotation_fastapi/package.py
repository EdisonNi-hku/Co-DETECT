import asyncio
import json
import sys
from pathlib import Path
import pandas as pd
from models import AnnotationRequest
from main import annotate_texts, annotate_one

########################################################################
# REQUEST BUILDER

# Builder class for constructing AnnotationRequest objects.
class AnnotationRequestBuilder:
    """Builder class for constructing AnnotationRequest objects."""
    
    def __init__(self):
        """Initialize the builder with default values."""
        self._examples = []
        self._annotation_guideline = ""
        self._task_id = ""
        self._reannotate_round = 0

    def set_examples_file_path(self, file_path: str, column_name: str = "text_to_annotate"):
        """
        Set examples from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            column_name: Name of the column containing text to annotate (default: "text_to_annotate")
        """
        df = pd.read_csv(file_path)
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in CSV. Available columns: {list(df.columns)}")
        self._examples = df[column_name].dropna().astype(str).tolist()

    def set_examples_existing_results(self, existing_results_path: str):
        """
        Set examples from existing annotation results JSON file.
        
        Args:
            existing_results_path: Path to the existing annotation results JSON file
        """
        with open(existing_results_path, "r") as f:
            data = json.load(f)
        annotations = data.get("annotations", []) if isinstance(data, dict) else data
        self._examples = [entry["text_to_annotate"] for entry in annotations if "text_to_annotate" in entry]
    
    def set_examples(self, examples: list):
        """Set examples for the request."""
        self._examples = examples
    
    def set_annotation_guideline(self, guideline: str):
        """Set the annotation guideline."""
        self._annotation_guideline = guideline
    
    def set_task_id(self, task_id: str):
        """Set the task ID."""
        self._task_id = task_id
    
    def set_reannotate_round(self, round_num: int = -1):
        """Set the reannotation round number."""
        if round_num < 0:
            self._reannotate_round += 1
        else:
            self._reannotate_round = round_num
    
    def get_request(self) -> AnnotationRequest:
        """Build and return the AnnotationRequest."""
        return AnnotationRequest(
            examples=self._examples,
            annotation_guideline=self._annotation_guideline,
            task_id=self._task_id,
            reannotate_round=self._reannotate_round
        )


########################################################################
# ANNOTATION CALL TOOLS  

# obtain annotation results 
async def annotation_inside_call(request: AnnotationRequest, singular: bool = False, verbose: bool = False) -> dict | None:
    
    # Call annotate_texts directly
    try:
        if verbose:
            print("Calling annotate_texts...")
        
        if singular:
            response = await annotate_one(request)
        else:
            response = await annotate_texts(request)
        
        result_dict = response.body.decode('utf-8')
        result_json = json.loads(result_dict)
        
        if verbose:
            print("\nAnnotation Results:")
            print(json.dumps(result_json, indent=2, ensure_ascii=False))

        return result_json
    except Exception as e:
        print(f"Error during annotation: {e}")
        return None


# what to call when running annotation results
# specifically does the annotation call, properly formats the results, and saves/prints them in a readable format 
def run_annotation(request: AnnotationRequest, out_json_path: str, out_txt_path: str, singular: bool = False, verbose: bool = False) -> None:
    results = asyncio.run(annotation_inside_call(request, singular=singular, verbose=verbose))

    handle_annotation_output(results, request.reannotate_round, out_json_path, out_txt_path, singular=singular, verbose=verbose)

    if verbose and out_txt_path and Path(out_txt_path).exists():
        with open(out_txt_path, "r", encoding="utf-8") as f:
            summary_text = f.read()
            print("\n" + summary_text)



########################################################################
# POST-ANNOTATION OUTPUT HANDLING FUNCTIONS    


def write_to_json(data: dict, output_json_path: str, msg: str = "Results saved to") -> None:
    """Write data to a JSON file."""
    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n{msg} {output_json_path}")


def json_handler(annotation_results: list, output_json_path: str, singular: bool = False):
    """
    handles all 3 of the following cases:
    1. No existing file (or new iteration), create new file with the annotation result
    2. Existing file with same uid, replace the old annotation with the new one
    3. Existing file without same uid, append the new annotation to the existing file
    """
    if singular:
        try:
            with open(output_json_path, "r") as f:
                existing_data = json.load(f)
        except FileNotFoundError:
            existing_data = {"annotations": []}
            print(f"No existing file found at {output_json_path}. A new file will be created.")

        annotation_results = annotation_results['annotations']
        new_uid = annotation_results[0]["uid"]
        existing_uids = {entry["uid"] for entry in existing_data["annotations"]}
        replaced = False
        if new_uid in existing_uids:
            for i, entry in enumerate(existing_data["annotations"]):
                if entry["uid"] == new_uid:
                    existing_data["annotations"][i] = annotation_results[0]
                    replaced = True
                    break
        if not replaced: 
            existing_data["annotations"].append(annotation_results[0])

    write_to_json(existing_data if singular else annotation_results, output_json_path)
    
 
def save_to_txt(output_json_path: str, output_txt_path: str, round: int = 0, verbose: bool = False): 
    """
    Save annotation results from JSON to a readable TXT file.
    
    Organizes annotations into categories:
    - Data points that are edge cases (you can tell from the 'new_edge_case' flag)
    - All other clusters (grouped by their cluster number)
    
    Each annotation entry contains all fields except: uid, raw_annotations, pca_x, pca_y
    
    Args:
        output_json_path: Path to the input JSON file
        output_txt_path: Path to save the output TXT file
    """
    try:
        # Load the JSON data
        with open(output_json_path, "r") as f:
            data = json.load(f)
        
        # Categorize annotations
        edge_cases = []
        other_clusters = {}  # Dictionary to group by cluster number
        
        annotations = data.get("annotations", []) if isinstance(data, dict) else data
        
        for annotation in annotations:
            # Skip certain fields
            fields_to_exclude = {"uid", "raw_annotations", "pca_x", "pca_y"}
            cleaned_annotation = {k: v for k, v in annotation.items() if k not in fields_to_exclude}
            
            # Check if this is an edge case using the 'new_edge_case' flag
            is_edge_case = annotation.get("new_edge_case", False)
            
            if is_edge_case:
                edge_cases.append(cleaned_annotation)
            else:
                # Group by cluster number
                cluster_num = annotation.get("cluster", annotation.get("cluster_num", "other"))
                if cluster_num not in other_clusters:
                    other_clusters[cluster_num] = []
                other_clusters[cluster_num].append(cleaned_annotation)
        
        # Write to TXT file
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write("Round " + str(round) + "\n")
            f.write("=" * 80 + "\n")
            f.write("ANNOTATION SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            # Edge cases section
            f.write("DATA POINTS THAT ARE EDGE CASES\n")
            f.write("-" * 80 + "\n")
            if edge_cases:
                for i, ann in enumerate(edge_cases, 1):
                    f.write(f"\n[{i}] {json.dumps(ann, indent=2, ensure_ascii=False)}\n")
            else:
                f.write("(None)\n")
            
            f.write("\n\n")
            
            # All other clusters grouped by cluster number
            f.write("ALL OTHER CLUSTERS\n")
            f.write("-" * 80 + "\n")
            if other_clusters:
                for cluster_num in sorted(other_clusters.keys(), key=lambda x: (isinstance(x, str), x)):
                    f.write(f"\n--- Cluster {cluster_num} ---\n")
                    for i, ann in enumerate(other_clusters[cluster_num], 1):
                        f.write(f"\n[{i}] {json.dumps(ann, indent=2, ensure_ascii=False)}\n")
            else:
                f.write("(None)\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Summary: {len(edge_cases)} edge cases, {sum(len(v) for v in other_clusters.values())} other annotations\n")
            f.write("=" * 80 + "\n")

        print(f"Summary saved to {output_txt_path}")
        if verbose:
            print(f"- Edge cases: {len(edge_cases)}")
            print(f"- Other annotations: {sum(len(v) for v in other_clusters.values())}")
        
    except Exception as e:
        print(f"Error saving to TXT: {e}")


def handle_annotation_output(annotation_results: list, round: int, output_json_path: str, output_txt_path: str, singular: bool = False, verbose: bool = False):
    """
    Handle annotation output by saving to JSON and TXT files.
    
    Args:
        annotation_results: The annotation results to save
        output_json_path: Path to save the JSON file
        output_txt_path: Path to save the TXT summary file
        singular: Whether the annotation was for a single example
    """
    json_handler(annotation_results, output_json_path, singular=singular)
    save_to_txt(output_json_path, output_txt_path, round=round, verbose=verbose)


########################################################################
# IDEA FOR MULTIPLE ANNOTATIONS PER CALL 

def multiple_annotation_handler(requests: list[AnnotationRequest], output_json_paths: list[str], output_txt_paths: list[str], final_out_json_path: str, verbose: bool = False) -> None:

    # ensure that the right values are the same (and different) across the requests  
    # examples and reannotate_round same
    assert all(req.examples == requests[0].examples for req in requests), "All requests must have the same examples"
    assert all(req.reannotate_round == requests[0].reannotate_round for req in requests), "All requests must have the same reannotate_round number"

    # annotation_guideline and task_id different
    assert len(set(req.annotation_guideline for req in requests)) == len(requests), "All requests must have different task IDs"
    assert len(set(req.task_id for req in requests)) == len(requests), "All requests must have different task IDs"

    # output paths different
    assert len(set(output_json_paths)) == len(output_json_paths), "Output JSON paths must be unique"
    assert len(set(output_txt_paths)) == len(output_txt_paths), "Output TXT paths must be unique"

    # run the annotations for each request 
    for i, request in enumerate(requests):
        run_annotation(request, output_json_paths[i], output_txt_paths[i], singular=False, verbose=verbose)

    labels = [req.task_id for req in requests]
    output_json = {"labels": labels, "annotations": []}
    to_keep = ["text_to_annotate", "uid", "cluster", "pca_x", "pca_y"] # only fields same across results 

    first = True
    for output_json_path in output_json_paths:
        with open(output_json_path, "r") as f:
            data = json.load(f)
        for i, entry in enumerate(data.get("annotations", [])):
            if first: 
                to_add = {}
            for key in entry:
                if first:
                    if key in to_keep:
                        to_add[key] = entry[key]
                    else: 
                        to_add[key] = [entry[key]]
                elif key not in to_keep:
                    output_json["annotations"][i][key].append(entry[key])
            if first:
                output_json["annotations"].append(to_add)
        first = False

    # save output_json to final_out_json_path
    write_to_json(output_json, final_out_json_path, msg="Combined results saved to")


# for later steps in the iteration 
def multiple_annotations_reannotate(request: AnnotationRequest, output_json_paths: list[str], output_txt_paths: list[str], final_out_json_path: str, singular: bool = False, verbose: bool = False):
    # first, need to read final_out_json_path["labels"] to see which task id index corresponds to the request task id
    with open(final_out_json_path, "r") as f:
        data = json.load(f)
    labels = data.get("labels", [])
    if request.task_id not in labels:
        raise ValueError(f"Task ID {request.task_id} not found in existing results. Cannot re-annotate.")
    task_index = labels.index(request.task_id)

    # then need to reannotate
    run_annotation(request, output_json_paths[task_index], output_txt_paths[task_index], singular=singular, verbose=verbose)

    # after that, need to update the final_out_json_path with the new annotation results
    with open(final_out_json_path, "r") as f:
        data = json.load(f)
    with open(output_json_paths[task_index], "r") as f:
        new_data = json.load(f)
    found_uids = []

    for i, entry in enumerate(data.get("annotations", [])):
        found_uids.append(entry['uid'])
        for key in entry:
            if type(entry[key]) == list:
                data["annotations"][i][key][task_index] = new_data["annotations"][i][key]

    # if any annotations are new, find it and append it to the data key by key
    if len(data["annotations"]) < len(new_data["annotations"]):
        for i, entry in enumerate(new_data.get("annotations", [])):
            if entry['uid'] not in found_uids:
                to_add = {}
                for key in entry:
                    try:
                        if type(data["annotations"][0][key]) == list:
                            to_add[key] = len(output_json_paths) * [None]
                            to_add[key][task_index] = new_data["annotations"][i][key]
                        else:
                            to_add[key] = new_data["annotations"][i][key]
                    # throwing away keys like edge_case_pca_x
                    except:
                        continue
                data["annotations"].append(to_add)

    write_to_json(data, final_out_json_path, msg="Updated combined results saved to")