# https://github.com/benjaminzwhite/kaggle-notebooks/blob/main/absa-test-inference-with-finetuned.ipynb

import json
import re

# -------------
# +++ FUZZY MATCH CODE CREDIT: 
# https://stackoverflow.com/a/73298537/22268217

import regex
from typing import Optional

def fuzzy_substring_search(major: str, minor: str, errs: int = 5) -> Optional[regex.Match]:
    """Find the closest matching fuzzy substring.

    Args:
        major: the string to search in
        minor: the string to search with
        errs: the total number of errors

    Returns:
        Optional[regex.Match] object
    """
    # TODO - ADJUST THIS LOGIC : my intuition is to allow up to the length of the
    # LLM detection as a fuzzy match ???? 
    # UPDATE IN TESTING : seems too permissive ??
    #errs = len(minor)

    errs_ = 0
    s = regex.search(f"({minor}){{e<={errs_}}}", major)
    while s is None and errs_ <= errs:
        errs_ += 1
        s = regex.search(f"({minor}){{e<={errs_}}}", major)

    # TODO - I NOTICED THAT SOMETIMES CAN GET MATCH THAT DOES NOT MATCH
    # WORD BOUNDARIES -> maybe adjust another postprocessing here to ensure end of full word
    return s
#----------------

example_review = "This place's burgers are the absolute best in town, and even though the service is incredibly slow I'd definitely come back - I want to try the tomato sauce that my friend had which looked delicious!"

example_raw_model_output = " [{'opinion term': 'burgers', 'aspect category': 'food quality','sentiment': 'positive', 'justification': 'best'},{'opinion term':'service', 'aspect category':'service general','sentiment': 'negative', 'justification':'slow'},{'opinion term': 'tomato sauce', 'aspect category': 'food quality','sentiment': 'positive', 'justification': 'delicious'}]"

# ----

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
        unparsable_llm_results = [] # UPDATE: store the bad results

        for quad_pred in data:
            # 1 - regex search for the opinion term if it is non NULL
            # UPDATE : use regex FUZZY SEARCH if there are no EXACT matches
            if quad_pred["opinion term"] != "NULL":
                x = re.search(quad_pred["opinion term"], input_text)

                if x is None:
                    x = fuzzy_substring_search(major=input_text, minor=quad_pred["opinion term"])

                    # if x is still none here, this quad_pred is unparsable so store as BAD result
                    if x is None:
                        unparsable_llm_results.append(quad_pred)
                        continue # TODO: check this skips all below and goes to next quad_pred in data


                tmp1 = {"start": x.start(), "end": x.end(), "label": "opinion term"}
                processed_ents.append(tmp1)

            # 2 - regex search for the aspect category/justification
            xx = re.search(quad_pred["justification"], input_text)
            # UPDATE : use regex FUZZY SEARCH if there are no EXACT matches
            if xx is None:
                xx = fuzzy_substring_search(major=input_text, minor=quad_pred["justification"])
                # if xx is still none here, this quad_pred is unparsable so store as BAD result
                if xx is None:
                    unparsable_llm_results.append(quad_pred) # should POP the previous item since it has the tmp1 from above???
                    continue


            # 2.1 == WIP == CREATE COLOR BASED ON SENT+ASPECTCATEGORY FUSED LABEL O_o
            fused_label = f"{quad_pred['sentiment']} {quad_pred['aspect category']}"
            tmp2 = {"start": xx.start(), "end": xx.end(), "label": fused_label}
            processed_ents.append(tmp2)

        dic_ents = {
            "text": input_text,
            "ents": processed_ents,
            "title": None
        }

        return dic_ents, unparsable_llm_results
    
    except Exception as e:
        print("PARSE ERROR", e)
        empty_ents = {
            "text": input_text,
            "ents": [],
            "title": None
        }

        return empty_ents, [] # [] is to match signature of unparsable_llm_results list being returned also
    
print(process_raw_output(example_review, example_raw_model_output))


