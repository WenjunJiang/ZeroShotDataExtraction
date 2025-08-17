from ollama import chat
from ollama import ChatResponse
import pandas as pd
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

system_prompt = "As a medical professional, please analyze the following medical notes carefully and extract relevant information in the specified JSON format. Do not include any values that are not mentioned in the notes."
# Define the template with a placeholder
prompt_template = """
For each entry:
- Accurately identify and extract patient data, visit motivation, symptoms, and vital signs.
- Only include keys that have corresponding information in the notes, omitting any keys that are not mentioned.
- Follow this JSON structure:

```json
{
  "patient_info": {
    "age": <number>,
    "gender": <string>
  },
  "visit_motivation": "",
  "symptoms": [<string>],
  "vital_signs": {
    "heart_rate": {
      "value": <number>,
      "unit": "bpm"
    },
    "oxygen_saturation": {
      "value": <number>,
      "unit": "%"
    },
    "cholesterol_level": {
      "value": <number>,
      "unit": "mg/dL"
    },
    "glucose_level": {
      "value": <number>,
      "unit": "mg/dL"
    },
    "temperature": {
      "value": <number>,
      "unit": "°C"
    },
    "respiratory_rate": {
      "value": <number>,
      "unit": "breaths/min"
    },
    "blood_pressure": {
      "systolic": {
        "value": <number>,
        "unit": "mmHg"
      },
      "diastolic": {
        "value": <number>,
        "unit": "mmHg"
      }
    }
  }
}

Here are the clinical notes to review:

{medical_notes}

"""
test_df = pd.read_csv("medical-note-extraction-h-2-o-gen-ai-world-ny/test.csv")
test_df.head()
# Try a small subset
trial_df = test_df.iloc[:10].copy()
# Build prompts
all_prompts = []
for index, row in trial_df.iterrows():
    medical_notes = row.Note
    filled_prompt = f"{prompt_template}".replace("{medical_notes}", medical_notes)
    message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": filled_prompt}
    ]
    all_prompts.append(message)

print(all_prompts[0])
def print_outputs(outputs):
    for output in outputs:
        print(f"Generated text: {output!r}")
        print("-" * 80)

model = "gpt-oss:20b"  # or your model name
options = {
    "temperature": 0.2,
    "top_p": 0.9,
    "seed": 777,
    "num_predict": 2048,  # vLLM's max_tokens
}

def run_batch(all_prompts):
    outputs = []
    for messages in all_prompts:
        resp: ChatResponse = chat(model=model, messages=messages, options=options)
        outputs.append(resp["message"]["content"])
    return outputs

##### Parallel batch
def _chat_one(idx, messages):
    resp = chat(model=model, messages=messages, options=options)
    return idx, resp["message"]["content"]

def run_batch_parallel(all_prompts, max_workers=4):
    results = [None] * len(all_prompts)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_chat_one, i, msgs) for i, msgs in enumerate(all_prompts)]
        for fut in as_completed(futs):
            i, text = fut.result()
            results[i] = text  # preserve input order
    return results

from time import time
start = time()

# responses = llm.generate(
#     all_prompts[0],
#     sampling_params,
#     use_tqdm = True
# )

## run multiple messages one by one
# outputs = run_batch(all_prompts)
## run multiple messages in parallel
outputs = run_batch_parallel(all_prompts, max_workers=5)

end = time()
elapsed = (end-start)/60. #minutes
print(f"Inference took {elapsed} minutes!")
print_outputs(outputs)


def clean_preds(preds: str):
    try:
        preds = preds.split('```json\n')[1].split('\n```')[0]
    except Exception as e:

        print(e)
        try:
            preds = preds.split('\n\n')[1]
        except Exception as e:
            print(e)
            preds = preds

    return preds


# %%
json_list = []

for output in outputs:
    res = output

    try:
        clean_pred = clean_preds(res)
        print("pred successfully")
        clean_pred_json = json.loads(clean_pred)
        print("json load successfully")
        json_list.append(clean_pred_json)
    except:
        print("Error")
        json_list.append(res)

# print json_list
for elem in json_list:
    print(elem)

# trial_df['json'] = json_list
# sub_df = trial_df[['ID', 'json']]
#
# sub_df.iloc[2]['json']