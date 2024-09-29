#import torch
import streamlit as st
#from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from spacy import displacy

from postprocess_phi3_inference import process_raw_output, COLOR_LOOKUP

import time # FOR DEBUGGING
DEBUG = True

# ----------------------
# --- Model loading ---
# CREDITS:
# https://huggingface.co/spaces/fabiochiu/text-to-kb/blob/main/app.py

# MODEL_NAME = "benjaminzwhite/phi-3-mini-4k-instruct-ABSA-QUAD"
# device = "cuda" if torch.cuda.is_available() else "cpu"

# st_model_load = st.text("Loading model ...")

# ==== TODO: CACHE MODEL LOAD ====
# https://docs.streamlit.io/get-started/fundamentals/advanced-concepts
#
# @st.cache(allow_output_mutation=True) # <----------- DOCS SAY USE st.cache_resource https://docs.streamlit.io/get-started/fundamentals/advanced-concepts
# def load_model():
#     print("Loading model...")
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#     model = AutoModelForCausalLM.from_pretrained( 
#         MODEL_NAME,  
#         device_map=device,  
#         torch_dtype="auto",  
#         trust_remote_code=True,  
#     ) 
#     print("Model loaded!")
#     return tokenizer, model

# if not DEBUG:
#     tokenizer, model = load_model()
#     st.success('Model loaded!')
#     st_model_load.text("")

# ----------------------




# TOGGLE WIDE MODE BY DEFAULT
st.set_page_config(layout="wide")




# ===========
def main():
    st.title("ABSA QUAD Demo")

    st.markdown("This is an app to visualize the predictions from the ABSA QUAD model. Currently works for restaurant reviews.")

    with st.expander("Show list of Sentiments and Aspect Categories"):
        st.markdown("""
                    ## Sentiments
                    
                    - positive
                    - negative
                    - neutral

                    ## Aspect Categories
                    
                    - food quality
                    - service general
                    - restaurant general
                    - ambience general
                    - food style_options
                    - restaurant miscellaneous
                    - food prices
                    - restaurant prices
                    - drinks quality
                    - drinks style_options
                    - location general
                    - drinks prices
                    - food general
                    """)

    option = st.selectbox(
    "Choose an example to analyse or select CUSTOM to enter custom example:",
    ("CUSTOM",
     "This place's burgers are the absolute best in town, and even though the service is incredibly slow I'd definitely come back - I want to try the tomato sauce that my friend had which looked delicious!",
     "We had to wait for 10 hours to get a table!!!"),
    )

    if option == "CUSTOM":
        input_text = st.text_input("Enter your text", "")
    else:
        input_text = option

    if st.button("Perform ABSA analysis"):
        with st.spinner("Running model..."):
            time.sleep(1)
            #result = model(input_text)
            #st.write("Prediction:", result[0]['label'], "| Score:", result[0]['score'])
            result = input_text + "!!!!!!"
            st.success('Model inference OK!', icon=":material/done_outline:")

        with st.container(border=True):
            st.subheader("Raw model output")
            st.write(result)

            # --- PARSE/PROCESS to spaCy visualizer ---

            # ==== PRETEND THIS IS THE RAW MODEL OUTPUT ===
            example_raw_model_output = " [{'opinion term': 'burgers', 'aspect category': 'food quality','sentiment': 'positive', 'justification': 'best'},{'opinion term':'service', 'aspect category':'service general','sentiment': 'negative', 'justification':'slow'},{'opinion term': 'tomato sauce', 'aspect category': 'food quality','sentiment': 'positive', 'justification': 'delicious'}]"

            dic_ents, unparsable_llm_results = process_raw_output(input_text, example_raw_model_output)

            ent_html = displacy.render(dic_ents, manual=True, style="ent", options={"colors":COLOR_LOOKUP})
            st.subheader("Processed output")
            st.markdown(ent_html, unsafe_allow_html=True)

            if unparsable_llm_results:
                st.subheader("Remaining unparsable results from model output")
                st.write(unparsable_llm_results)

if __name__ == "__main__":
    main()