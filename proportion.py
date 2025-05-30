# Calculate behavior proportion by fear level
# First, add a fear level category to the original dataframe
df['fear_level'] = pd.cut(
    df['Self-assessment Fear'],
    bins=[0, 2.1, 3.1, 4.1],
    labels=['Low', 'Medium', 'High']
)

# Create a crosstab of fear levels and behaviors
fear_behavior = pd.crosstab(df['fear_level'], df['event_type'], normalize='index')

# Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(fear_behavior, cmap='YlOrRd', annot=True, fmt='.2f')
plt.title('Proportion of Behaviors by Fear Level')
plt.xlabel('Behavior Type')
plt.ylabel('Fear Level')
plt.tight_layout()
plt.savefig('fear_behavior_heatmap.png')

# Boxplot for main behaviors by fear category
plt.figure(figsize=(12, 8))
melted = pd.melt(participant_data, 
                 id_vars=['fear_level'], 
                 value_vars=selected_behaviors,
                 var_name='Behavior Type', 
                 value_name='Count')
                
sns.boxplot(x='Behavior Type', y='Count', hue='fear_level', data=melted)
plt.title('Behavior Comparison by Fear Level')
plt.xlabel('Behavior Type')
plt.ylabel('Behavior Count')
plt.legend(title='Fear Level')
plt.tight_layout()
plt.savefig('behavior_by_fear_level.png')

# Create mosaic plot for scene types and fear levels
scene_fear_data = df[['scene_type', 'fear_level']].copy()
plt.figure(figsize=(20, 20))
mosaic(scene_fear_data, ['scene_type', 'fear_level'])
plt.title('Relationship Between Scene Type and Fear Level')
plt.tight_layout()
plt.savefig('scene_fear_mosaic.png')
