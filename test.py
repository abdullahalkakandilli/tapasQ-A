from transformers import TapasTokenizer, TapasForQuestionAnswering
import pandas as pd


df = pd.read_csv(r'C:\Users\alka\Masaüstü\testDataTapas.csv')

table_ = df.to_dict()
new_data = {}
# loop through each key in the original dictionary
for key in table_:
    # get the values from the inner dictionary and convert them to a list
    values = list(table_[key].values())
    # add the values to the new dictionary with the same key
    values = [str(value) if not isinstance(value, str) else value for value in values]
    new_data[key] = values


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
question_ = "What is the name of the first actor?"

answer_ = get_values(question_)
print(answer_[0])


#What is the name of the first actor?
#Predicted answer: Brad Pitt
