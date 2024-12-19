import pandas as pd
import seaborn as sns

from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt

# fetch dataset
wine_quality = fetch_ucirepo(id=186)

# data (as pandas dataframes)
X = wine_quality.data.features
y = wine_quality.data.targets

# metadata
print(wine_quality.metadata)

# variable information
print(wine_quality.variables)

wine_data = pd.concat([X,y], axis=1)
print(wine_data.isnull().sum())

# Visualize Quality Distribution
ax = sns.countplot(x="quality", data=wine_data, color="#FFCC99")  # Light orange
plt.title("Distribution of Wine Quality")

# Add count numbers on top of bars
for bar in ax.patches:
    ax.text(
        bar.get_x() + bar.get_width() / 2,  # X-coordinate: center of the bar
        bar.get_height() + 0.5,            # Y-coordinate: slightly above the bar
        int(bar.get_height()),             # Text: height of the bar
        ha="center",                       # Center align text horizontally
        va="bottom"                        # Bottom align text vertically
    )

plt.savefig("label_distribution.png")

plt.figure(figsize=(10, 8))
sns.heatmap(wine_data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
