import pandas as pd
from datetime import datetime

file_path = "data.csv"  
data = pd.read_csv(file_path)

# 1. Normalize date format
def normalize_date(date):
    try:
        return datetime.strptime(date, '%m-%d-%Y %H:%M').strftime('%m-%d-%Y %H:%M')
    except:
        return None  # Mark invalid dates for review

data['START_DATE*'] = data['START_DATE*'].apply(normalize_date)
data['END_DATE*'] = data['END_DATE*'].apply(normalize_date)

# 2. Handle missing values
# Fill missing values for optional fields with "Unknown"
data['PURPOSE'] = data['PURPOSE'].fillna('Unknown')

# Flag or drop rows with critical missing values
data = data.dropna(subset=['START_DATE', 'END_DATE', 'CATEGORY', 'START', 'STOP', 'MILES'])

# 3. Ensure numeric consistency for MILES
data['MILES'] = pd.to_numeric(data['MILES'], errors='coerce')
data = data[data['MILES'] > 0]  # Remove rows with invalid or zero miles

# 4. Ensure START_DATE is earlier than END_DATE
data['START_DATE'] = pd.to_datetime(data['START_DATE'], format='%m-%d-%Y %H:%M')
data['END_DATE'] = pd.to_datetime(data['END_DATE'], format='%m-%d-%Y %H:%M')
data = data[data['END_DATE'] > data['START_DATE']]

# 5. Ensure uniform capitalization
data['CATEGORY'] = data['CATEGORY'].str.title()
data['START'] = data['START'].str.title()
data['STOP'] = data['STOP'].str.title()
data['PURPOSE'] = data['PURPOSE'].str.title()

# Save the cleaned data
output_file = "cleaned_data.csv"
data.to_csv(output_file, index=False)
print(f"Cleaned data saved to {output_file}")
