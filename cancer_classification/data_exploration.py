import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("dataset.csv")

# Display basic statistics about the dataset
print("Dataset Statistics:")
print(df.describe())

# Display the first few rows of the dataset
print("\nFirst Few Rows of the Dataset:")
print(df.head())

# Plot distribution of classes
plt.figure(figsize=(12, 6))
sns.countplot(x='class_name', data=df, order=df['class_name'].value_counts().index)
plt.title('Distribution of Classes')
plt.xlabel('Class Name')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.show()

# Display a few sample images
plt.figure(figsize=(12, 6))
for i in range(6):
    plt.subplot(2, 3, i+1)
    img_path = df.iloc[i, 0]
    img = plt.imread(img_path)
    plt.imshow(img)
    plt.title(f"Class: {df.iloc[i, 1]}")
    plt.axis("off")

plt.show()
