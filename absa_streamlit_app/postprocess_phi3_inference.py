# https://github.com/benjaminzwhite/kaggle-notebooks/blob/main/absa-test-inference-with-finetuned.ipynb

import json
import re

example_review = "This place's burgers are the absolute best in town, and even though the service is incredibly slow I'd definitely come back - I want to try the tomato sauce that my friend had which looked delicious!"

example_raw_model_output = " [{'opinion term': 'burgers', 'aspect category': 'food quality','sentiment': 'positive', 'justification': 'best'},{'opinion term':'service', 'aspect category':'service general','sentiment': 'negative', 'justification':'slow'},{'opinion term': 'tomato sauce', 'aspect category': 'food quality','sentiment': 'positive', 'justification': 'delicious'}]"



RGB_GREEN = "#50C878"
RGB_RED = "#D22B2B"
RGB_YELLOW = "#E4D00A"
RGB_BLUE = "#6F8FAF"
RGB_GRAY = "#899499"

ASPECT_CATEGORIES = [
    "food quality",
    "service general",
    "restaurant general",
    "ambience general",
    "food style_options",
    "restaurant miscellaneous",
    "food prices",
    "restaurant prices",
    "drinks quality",
    "drinks style_options",
    "location general",
    "drinks prices",
    "food general"
]
SENTIMENTS = ["positive", "negative", "neutral"]

COLOR_LOOKUP = {"opinion term": RGB_BLUE}
for sent in SENTIMENTS:
    for ac in ASPECT_CATEGORIES:
        if sent == "positive":
            COLOR_LOOKUP[f"{sent} {ac}"] = RGB_GREEN
        elif sent == "negative":
            COLOR_LOOKUP[f"{sent} {ac}"] = RGB_RED
        elif sent == "neutral":
            COLOR_LOOKUP[f"{sent} {ac}"] = RGB_YELLOW
        else:
            COLOR_LOOKUP[f"{sent} {ac}"] = RGB_GRAY

def process_raw_output(input_text, raw_model_output):
    try:
        # load string to json
        
        # UPDATE DEBUG --
        # NEED TO USE DOUBLE QUOTES FOR JSON -.-
        raw_model_output = raw_model_output.replace("\'", "\"")
        
        data = json.loads(raw_model_output)
        # go over the dicts in the data, make sure you stock the "label"
        # use 3 colors to encode the sentiment
        processed_ents = []
        for quad_pred in data:
            # 1 - regex search for the opinion term if it is non NULL
            if quad_pred["opinion term"] != "NULL":
                x = re.search(quad_pred["opinion term"], input_text)
                tmp1 = {"start": x.start(), "end": x.end(), "label": "opinion term"}
                processed_ents.append(tmp1)
            # 2 - regex search for the aspect category/justification
            xx = re.search(quad_pred["justification"], input_text)
            # 2.1 == WIP == CREATE COLOR BASED ON SENT+ASPECTCATEGORY FUSED LABEL O_o
            fused_label = f"{quad_pred['sentiment']} {quad_pred['aspect category']}"
            tmp2 = {"start": xx.start(), "end": xx.end(), "label": fused_label}
            processed_ents.append(tmp2)

        dic_ents = {
            "text": input_text,
            "ents": processed_ents,
            "title": None
        }

        return dic_ents
    
    except Exception as e:
        print("PARSE ERROR", e)
        return {
            "text": input_text,
            "ents": [],
            "title": None
        }
    
print(process_raw_output(example_review, example_raw_model_output))


