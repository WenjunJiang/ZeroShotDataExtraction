import pandas as pd
import json
import warnings
import json_metric
warnings.filterwarnings('ignore')

train_df = pd.read_csv('medical-note-extraction-h-2-o-gen-ai-world-ny/train.csv')
#print(train_df.columns)

with open("medical_records.json", 'r') as f:
    pred_json = json.load(f)


def predjson_to_dataframe(pred_json):
    """
    Convert a list of dicts with keys 'id' and 'medical_record'
    into a pandas DataFrame with columns 'ID' and 'json'.

    'json' column stores medical_record as a JSON string.
    """
    data = []
    for entry in pred_json:
        data.append({
            "ID": entry["id"],
            "pred_json": json.dumps(entry["medical_record"], ensure_ascii=False)
        })

    return pd.DataFrame(data)


pred_sample_df = predjson_to_dataframe(pred_json)
sample_ID = pred_sample_df["ID"].tolist()


train_sample_df = train_df.merge(pred_sample_df, on="ID", how="right").drop(columns=["Note", "pred_json"])


similarity = json_metric.score(train_sample_df, pred_sample_df.rename(columns={"pred_json":"json"}), "ID")
print("The score for evaluation metric is:", similarity)