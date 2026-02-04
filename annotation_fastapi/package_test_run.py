import package
import json

# An example of using labelled_data_small.csv to run an annotation request
# (Also good for guiding the agent for how to run the package methods)

# Define file paths and parameters
input_csv_path = "labeled_data_small_cleaned.csv"
column_name = "text_to_annotate"
out_json_path = "annotation_results.json"
out_txt_path = "annotation_summary.txt"

task_id = "hate_speech_v2"
reannotate_round = 0

#################### First example: build the annotation request ##########################
builder = package.AnnotationRequestBuilder()

builder.set_examples_file_path(
    file_path=input_csv_path,
    column_name=column_name
)

# Set annotation guideline for hate speech classification
builder.set_annotation_guideline("""A post contains hate speech if it contains any of the following aspects:
- Assaults on Human Dignity: Does the post demean or degrade individuals or groups based on race, ethnicity, gender, religion, sexual orientation, or other protected characteristics?
- Calls for Violence: Does the post incite or encourage physical harm or violence against individuals or groups?
- Vulgarity and/or Offensive Language: Does the post contain profanity, slurs, or other offensive language that may or may not be directed at individuals or groups?
""")

# Set task ID
builder.set_task_id(task_id)
builder.set_reannotate_round(reannotate_round)

# Get the request
request = builder.get_request()

# Run start annotation 
package.run_annotation(request, out_json_path, out_txt_path, singular=False, verbose=False)


#################### Second example: re-annotation ##########################
builder.set_annotation_guideline("""A post contains hate speech if it contains any of the following aspects:
- Assaults on Human Dignity: Does the post demean or degrade individuals or groups based on race, ethnicity, gender, religion, sexual orientation, or other protected characteristics?
- Calls for Violence: Does the post incite or encourage physical harm or violence against individuals or groups?
- Vulgarity and/or Offensive Language: Does the post contain profanity, slurs, or other offensive language that may or may not be directed at individuals or groups?
- Evangelical language: in our guidelines this is considered to be hate speech.""")
request = builder.get_request()

text_to_annotate = "Jesus is the final solution."
builder.set_examples([text_to_annotate])
request = builder.get_request()
package.run_annotation(request, out_json_path, out_txt_path, singular=True, verbose=False) 


#################### Third example: include a new point ##########################
# (another edge case example)
builder.set_examples(["I have a love/hate relationship with autists."])
request = builder.get_request()
package.run_annotation(request, out_json_path, out_txt_path, singular=True, verbose=False)



#################### Fourth example: iterate ##########################
builder.set_examples_existing_results(out_json_path)
builder.set_annotation_guideline("""A post contains hate speech if it contains any of the following aspects (though be a little more lenient this round):
- Assaults on Human Dignity: Does the post demean or degrade individuals or groups based on race, ethnicity, gender, religion, sexual orientation, or other protected characteristics?
- Calls for Violence: Does the post incite or encourage physical harm or violence against individuals or groups?
- Evangelical language: in our guidelines, specifically for cultish religions, this is considered to be hate speech.""")
builder.set_reannotate_round()
print("Examples set from existing results:", builder._examples)

request = builder.get_request()
package.run_annotation(request, out_json_path, out_txt_path, singular=False, verbose=True)

