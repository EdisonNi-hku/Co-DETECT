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
    
    def set_examples(self, examples: list):
        """Set examples for the request."""
        self._examples = examples
    
    def set_annotation_guideline(self, guideline: str):
        """Set the annotation guideline."""
        self._annotation_guideline = guideline
    
    def set_task_id(self, task_id: str):
        """Set the task ID."""
        self._task_id = task_id
    
    def set_reannotate_round(self, round_num: int):
        """Set the reannotation round number."""
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
# EXAMPLE REQUEST BUILDER

# code to build an importable example request 
def build_example_request() -> AnnotationRequest:

    builder = AnnotationRequestBuilder()
    
    builder.set_examples([
        "This product is amazing! I love it.",
        "Terrible service, never coming back.",
        "The quality is okay, nothing special.",
        "Best purchase I've ever made!",
        "Waste of money and time."
    ])
    builder.set_annotation_guideline("""Analyze customer sentiment in the given text.

Labels:
- positive
- negative
- neutral
""")
    builder.set_task_id("sentiment_analysis_demo")
    builder.set_reannotate_round(0)
   
    return builder.get_request()

example_request = build_example_request()


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
def get_annotation(request: AnnotationRequest, singular: bool = False, verbose: bool = False) -> dict | None:
    return asyncio.run(annotation_inside_call(request, singular=singular, verbose=verbose))


