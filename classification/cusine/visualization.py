import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your CSV file
df = pd.read_csv('../data/Cleaned_Indian_Food_Dataset.csv')  # Replace with your actual file name

# Check the first few rows to ensure the column exists
print(df.head())

# Count frequency of each cuisine
cuisine_counts = df['Cuisine'].value_counts()

# Plotting using seaborn
plt.figure(figsize=(10, 6))
sns.barplot(x=cuisine_counts.values, y=cuisine_counts.index, palette='viridis')
plt.title('Cuisine Distribution')
plt.xlabel('Number of Occurrences')
plt.ylabel('Cuisine')
plt.tight_layout()
plt.savefig('cuisine_distribution.png', dpi=300, bbox_inches='tight') 
plt.show()
