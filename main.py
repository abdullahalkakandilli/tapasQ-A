import requests
import streamlit as st
from functionforDownloadButtons import download_button
from transformers import TapasTokenizer, TapasForQuestionAnswering

import pandas as pd

def _max_width_():
    max_width_str = f"max-width: 1800px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )

st.set_page_config(page_icon="✂️", page_title="Tapas")


c2, c3 = st.columns([6, 1])


with c2:
    c31, c32 = st.columns([12, 2])
    with c31:
        st.caption("")
        st.title("Tapas")
    with c32:
        st.image(
            "images/logo.png",
            width=200,
        )

uploaded_file = st.file_uploader(
    " ",
    key="1",
    help="To activate 'wide mode', go to the hamburger menu > Settings > turn on 'wide mode'",
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    uploaded_file.seek(0)
    table_ = df.to_dict()
    new_data = {}
    # loop through each key in the original dictionary
    for key in table_:
        # get the values from the inner dictionary and convert them to a list
        values = list(table_[key].values())
        # add the values to the new dictionary with the same key
        values = [str(value) if not isinstance(value, str) else value for value in values]
        new_data[key] = values


    file_container = st.expander("Check your uploaded .csv")
    file_container.write(df)


else:
    st.info(
        f"""
            👆 Upload a .csv file first. Sample to try: [biostats.csv](https://people.sc.fsu.edu/~jburkardt/data/csv/biostats.csv)
            """
    )

    st.stop()

answers = []

def get_values(question_input):
    def get_values(question):
        model_name = "google/tapas-base-finetuned-wtq"
        model = TapasForQuestionAnswering.from_pretrained(model_name)
        tokenizer = TapasTokenizer.from_pretrained(model_name)

        queries = [
            question
        ]

        table = pd.DataFrame.from_dict(new_data)

        inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="pt")

        outputs = model(**inputs)

        predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(

            inputs, outputs.logits.detach(), outputs.logits_aggregation.detach()

        )

        # let's print out the results:

        id2aggregation = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}

        aggregation_predictions_string = [id2aggregation[x] for x in predicted_aggregation_indices]

        answers = []

        for coordinates in predicted_answer_coordinates:

            if len(coordinates) == 1:

                # only a single cell:

                answers.append(table.iat[coordinates[0]])

            else:

                # multiple cells

                cell_values = []

                for coordinate in coordinates:
                    cell_values.append(table.iat[coordinate])

                answers.append(", ".join(cell_values))

        return answers

form = st.form(key="annotation")
with form:
    question_input = st.text_input("Enter your query here")

    submitted = st.form_submit_button(label="Submit")
result_df = pd.DataFrame()
if submitted:

    result = get_values(question_input)
    print(question_input)


c29, c30, c31 = st.columns([1, 1, 2])


with c29:

    st.text(answers)
