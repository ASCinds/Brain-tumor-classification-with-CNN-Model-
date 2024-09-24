import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

df_results = pd.read_csv("results.csv")
df_results.columns = ["actual_class", "predicted_class"]

# Calculate metrics
accuracy = accuracy_score(df_results['actual_class'], df_results['predicted_class'])
precision = precision_score(df_results['actual_class'], df_results['predicted_class'], average='weighted')
recall = recall_score(df_results['actual_class'], df_results['predicted_class'], average='weighted')
f1 = f1_score(df_results['actual_class'], df_results['predicted_class'], average='weighted')

print(f"Accuracy: {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1 Score: {f1:.2%}")

# Confusion Matrix
conf_matrix = confusion_matrix(df_results['actual_class'], df_results['predicted_class'])

# Plot Confusion Matrix using Seaborn
plt.figure(figsize=(16, 13))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='coolwarm', cbar=True,
            xticklabels=sorted(df_results['actual_class'].unique()),
            yticklabels=sorted(df_results['actual_class'].unique()))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
plt.savefig("confusion_matrix.png")

# Classification Report
plt.figure(figsize=(16, 13))
report = classification_report(df_results['actual_class'], df_results['predicted_class'])
sns.heatmap(pd.DataFrame.from_dict(classification_report(df_results['actual_class'], df_results['predicted_class'], output_dict=True)).iloc[:-1, :].T, annot=True, fmt=".2%", cmap='Blues')
plt.title('Classification Report')

plt.tight_layout()
plt.savefig("classification_report.png")
