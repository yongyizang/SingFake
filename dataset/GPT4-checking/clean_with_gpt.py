import openai, json
import pandas as pd
from tqdm import tqdm

openai.api_key = None # fill in your API key
assert openai.api_key is not None, "You must provide an API key"
systemPrompt = open('systemPrompt.txt', 'r').read().strip()
csv_path = None # fill in the path to the csv file you want to clean
assert csv_path is not None, "You must provide the path to the csv file you want to clean"

csv_file = pd.read_csv(csv_path)

for i in tqdm(range(len(csv_file)), desc="Cleaning"):
    # get url
    url = csv_file.iloc[i]['Url'].strip()
    singer_name = csv_file.iloc[i]['Singer']
    title = csv_file.iloc[i]['Title'].strip()
    try:
        HTMLContent = None # Implement HTMLContent.
        assert HTMLContent is not None, "You must provide an HTML content. This part of the script need to be implemented by yourself."

        response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
                {"role": "system", "content": systemPrompt},
                {"role": "user", "content": HTMLContent},
            ]
        )
        # get response text
        response_json = json.loads(response['choices'][0]['message']['content'])

        cleaned_spoof_tag = response_json["type"]
        cleaned_singer_tag = response_json["singer"]
        cleaned_model_tag = response_json["model"]
        cleaned_correctness = str(response_json["correct"])

        
        csv_file.iloc[i]['Bonafide Or Spoof'] = cleaned_spoof_tag
        csv_file.iloc[i]['Singer'] = cleaned_singer_tag
        csv_file.iloc[i]['Model'] = cleaned_model_tag
        if cleaned_correctness.strip().lower() != "true":
            print("WARNING: Correctness is not True at {}".format(csv_file.iloc[i]['Url']))
            print("Singer wrote as {}, but should be {}".format(singer_name, cleaned_singer_tag))
    except Exception as e:
        print(title, e)
        continue