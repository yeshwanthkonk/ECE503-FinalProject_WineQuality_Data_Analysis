import matplotlib.pyplot as plt

# Combine X and y into a single DataFrame for convenience
df = pd.concat([X, y], axis=1)

# Set up the figure with subplots for each column
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))  # Adjust grid size (3x4 for 12 features)
fig.suptitle('Distribution of Features', fontsize=16)

# Flatten axes to access them easily in a loop
axes = axes.flatten()

# Plot histogram for each column
for i, col in enumerate(df.columns):
    ax = axes[i]
    ax.hist(df[col], bins=20, color='skyblue', edgecolor='black')
    ax.set_title(col)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')

# Hide unused subplots (if any)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.subplots_adjust(top=0.9)  # Adjust spacing for the title
plt.show()
