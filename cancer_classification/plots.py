import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("performance.csv")

# Plotting
plt.figure(figsize=(12, 6))

# Plot Training and Validation Loss
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
plt.plot(df.index, df['TL'], label='Train Loss', marker='o')
plt.plot(df.index, df['VL'], label='Validation Loss', marker='o')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Training and Validation Accuracy
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
plt.plot(df.index, df['TA'], label='Train Accuracy', marker='o')
plt.plot(df.index, df['VA'], label='Validation Accuracy', marker='o')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()  # Adjust layout to prevent overlapping
plt.savefig('plots.png')  # Save the figure as a PNG image
