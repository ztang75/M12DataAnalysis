# Calculate average fear scores per participant
fear_scores = df.groupby('participant_id')['Self-assessment Fear'].mean().reset_index()
# Calculate maximum fear scores (alternative metric)
max_fear = df.groupby('participant_id')['Self-assessment Fear'].max().reset_index()
fear_scores['max_fear'] = max_fear['Self-assessment Fear']

# Merge features and fear scores
participant_data = pd.merge(behavior_counts, fear_scores, on='participant_id')

# Create fear level categories (4=high, 3=medium, 1-2=low)
participant_data['fear_level'] = pd.cut(
    participant_data['max_fear'],
    bins=[0, 2.1, 3.1, 4.1],
    labels=['Low', 'Medium', 'High']
)

# Binary high fear classification (4=high fear, 1-3=not high fear)
participant_data['high_fear'] = (participant_data['max_fear'] == 4).astype(int)

# List all behavior columns
behavior_columns = participant_data.columns.difference(
    ['participant_id', 'Self-assessment Fear', 'max_fear', 'high_fear', 'fear_level']
).tolist()
