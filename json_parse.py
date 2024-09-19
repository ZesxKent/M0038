import pandas as pd
import json

data = pd.read_csv('dogbot430_2022.csv')

def parse_json(data):
    try:
        parsed = json.loads(data)
        if isinstance(parsed, dict):
            return parsed
        else:
            return {}
    except json.JSONDecodeError:
        return {}

parsed_data = data['data'].apply(parse_json)
parsed_df = pd.json_normalize(parsed_data)

data = pd.concat([data, parsed_df], axis=1)
data.drop(columns=['data'])
print(data.head())

data.to_csv('processed_data_430_2022.csv', index=False)