import package
import json

# An example of using labelled_data_small.csv to run an annotation request
# (Not intended for guiding the agent -- yet)

# Define file paths and parameters
input_csv_path = "labeled_data_small_cleaned.csv"
column_name = "text_to_annotate"
final_out_json_path = "annotation_results_comb.json"
out_json_paths = ["annotation_results_0.json", "annotation_results_1.json"]
out_txt_paths = ["annotation_summary_0.txt", "annotation_summary_1.txt"]
requests = []
reannotate_round = 0

# build the annotation requests 
builder = package.AnnotationRequestBuilder()

# same things for both requests 
builder.set_examples_file_path(
    file_path=input_csv_path,
    column_name=column_name
)
builder.set_reannotate_round(reannotate_round)

#################### First example: build the annotation request ##########################

# the first one (hate speech task)
builder.set_annotation_guideline("""A post contains hate speech if it contains any of the following aspects:
- Assaults on Human Dignity: Does the post demean or degrade individuals or groups based on race, ethnicity, gender, religion, sexual orientation, or other protected characteristics?
- Calls for Violence: Does the post incite or encourage physical harm or violence against individuals or groups?
- Vulgarity and/or Offensive Language: Does the post contain profanity, slurs, or other offensive language that may or may not be directed at individuals or groups?
""")
builder.set_task_id("hate_speech_v2")

# Get the request
request = builder.get_request()
requests.append(request)

# the second one (sentiment analysis task -- just used chatgpt for this example guideline)
builder.set_annotation_guideline("""A post expresses sentiment based on the overall emotional tone and evaluative stance of the author. Annotate sentiment according to the following aspects:
- Emotional Valence: Does the post convey a primarily positive, negative, or neutral emotional tone (e.g., approval, anger, frustration, satisfaction, or factual neutrality)?
- Target of Sentiment: Is the sentiment directed toward a person, group, institution, event, idea, or the speaker themselves? Label sentiment based on the expressed attitude, not the identity of the target.
- Intensity and Context: How strong or explicit is the sentiment, and how does context (e.g., sarcasm, humor, exaggeration) affect its interpretation?
Do not base annotations on the presence of profanity, offensive language, or social acceptability; these elements may appear in posts of any sentiment category.
""")
builder.set_task_id("sentiment_analysis_v2")

# Get the request 
request = builder.get_request()
requests.append(request)

# Run start annotation 
package.multiple_annotation_handler(requests, out_json_paths, out_txt_paths, final_out_json_path, verbose=False)

#################### Second example: re-annotation ##########################
builder.set_task_id("hate_speech_v2")
builder.set_annotation_guideline("""A post contains hate speech if it contains any of the following aspects:
- Assaults on Human Dignity: Does the post demean or degrade individuals or groups based on race, ethnicity, gender, religion, sexual orientation, or other protected characteristics?
- Calls for Violence: Does the post incite or encourage physical harm or violence against individuals or groups?
- Vulgarity and/or Offensive Language: Does the post contain profanity, slurs, or other offensive language that may or may not be directed at individuals or groups?
- Evangelical language: in our guidelines this is considered to be hate speech.""")

text_to_annotate = "Jesus is the final solution."
builder.set_examples([text_to_annotate])
request = builder.get_request()
package.multiple_annotations_reannotate(request, out_json_paths, out_txt_paths, final_out_json_path, singular=True, verbose=False)



#################### Third example: include a new point ##########################
builder.set_examples(["I have a love/hate relationship with autists."])
request = builder.get_request()
package.multiple_annotations_reannotate(request, out_json_paths, out_txt_paths, final_out_json_path, singular=True, verbose=False)



#################### Fourth example: iteration ##########################

builder.set_examples_existing_results(final_out_json_path)
builder.set_annotation_guideline("""A post contains hate speech if it contains any of the following aspects (though be a little more lenient this round):
- Assaults on Human Dignity: Does the post demean or degrade individuals or groups based on race, ethnicity, gender, religion, sexual orientation, or other protected characteristics?
- Calls for Violence: Does the post incite or encourage physical harm or violence against individuals or groups?
- Evangelical language: in our guidelines, specifically for cultish religions, this is considered to be hate speech.""")
builder.set_reannotate_round()
print("Examples set from existing results:", builder._examples)

request = builder.get_request()
package.multiple_annotations_reannotate(request, out_json_paths, out_txt_paths, final_out_json_path, singular=False, verbose=True)

