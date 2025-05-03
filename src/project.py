import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("D:\CSE\Financial-Profile-Analysis-and-Default-Risk-Insights-project/dataset/LoanData_Preprocessed_v1.1.csv")

## checking the data types
print(df.info())

# data balanced or imbalanced

default_counts = df['default'].value_counts()
print("default variable distribution:\n", default_counts)

default_percentages = df['default'].value_counts(normalize=True) * 100
print("\nDefault variable percentage distribution:\n", default_percentages)

threshold = 10

percentage_0 = default_percentages.get(0, 0)
percentage_1 = default_percentages.get(1, 0)

percentage_difference = abs(percentage_0 - percentage_1)
balance_status = "balanced" if percentage_difference < threshold else "not balanced"
print("\nBalance status:", balance_status)
print(f"Percentage difference between '0' and '1': {percentage_difference:.2f}%")


default_str = df['default'].astype(str)

default_grouped = default_str.apply(lambda x: x if x in ['0', '1'] else 'Garbage')

default_percentages = default_grouped.value_counts(normalize=True) * 100

order = ['0', '1', 'Garbage']
default_percentages = default_percentages.reindex([x for x in order if x in default_percentages.index])

# Plot
fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
default_percentages.plot(
    kind='pie',
    ax=ax,
    labels=default_percentages.index,
    autopct='%1.1f%%',
    startangle=270,
    fontsize=14,
    colors=plt.cm.Paired.colors,
    wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
    labeldistance=1.2,
    pctdistance=0.7
)

ax.set_ylabel('')
ax.set_title('Default Variable Distribution', fontsize=16)
plt.tight_layout()
plt.show()

for column in df.columns:
    if column == 'ed':
        category_count = df[column].value_counts(dropna=False)
        category_percentages = df[column].value_counts(normalize=True, dropna=False) * 100

        print(f"--- Stats for {column} ---")
        print("Value counts:\n", category_count)
        print("\nPercentage:\n", category_percentages.round(2))

    elif df[column].dtypes == 'float':
        average = df[column].mean()
        std_dev = df[column].std()
        variance = df[column].var()
        median = df[column].median()
        mode = df[column].mode()
        
        print(f"--- Numerical Stats for {column} ---")
        print(f"Average (Mean): {average:.2f}")
        print(f"Standard Deviation: {std_dev:.2f}")
        print(f"Variance: {variance:.2f}")
        print(f"Median: {median:.2f}\n\n")
    else:
        category_count = df[column].value_counts()
        category_percentages = df[column].value_counts(normalize=True) * 100

        print(f"--- Stats for {column} ---")
        print("Value counts:\n", category_count)
        print("\nPercentage:\n", category_percentages)

# Check for missing values in each column
df['age'] = df['age'].fillna(df['age'].mean())
df['income'] = df['income'].fillna(df['income'].mean())
df['ed'] = df['ed'].fillna(df['ed'].mode()[0])

print(df.isnull().sum())
print(df[df.isnull().any(axis=1)])



# correlation heat-map with each column. 
corr_matrix = df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='RdBu', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()

for column in df.columns:
    plt.figure(figsize=(8, 6))
    
    if df[column].dtype in ['float64']:
        if df[column].nunique() > 20:
            sns.histplot(df[column], kde=True)
            plt.title(f"{column} Distribution (Histogram)")
            plt.xlabel(column)
            plt.ylabel("Frequency")
        else:
            sns.countplot(x=column, data=df)
            plt.title(f"{column} Distribution (Bar Chart)")
            plt.xlabel(column)
            plt.ylabel("Count")
    
    else:
        sns.countplot(x=column, data=df)
        plt.title(f"{column} Distribution (Bar Chart)")
        plt.xlabel(column)
        plt.ylabel("Count")
    
    plt.tight_layout()
    plt.show()

df_sorted = df.sort_values('income')
plt.figure(figsize=(10, 8))

# Plot Income vs Age (first plot)
plt.subplot(2, 1, 1)
sns.lineplot(x='age', y='income', data=df_sorted)
plt.title('Income vs Age')
plt.xlabel('Age')
plt.ylabel('Income (in thousands)')
plt.tight_layout()
plt.show()

# Plot Income vs Age (second plot)
plt.subplot(2, 1, 2)
sns.lineplot(x='income', y='debtinc', data=df_sorted, label='Debt-to-Income Ratio')
sns.lineplot(x='income', y='creddebt', data=df_sorted, label='Credit Debt')
sns.lineplot(x='income', y='othdebt', data=df_sorted, label='Other Debt')
plt.title('Income vs Debt')
plt.xlabel('Income (in thousands)')
plt.ylabel('Debt (in thousands)')
plt.tight_layout()
plt.show()

for column in df.columns:
    if df[column].dtype in ['int64', 'float64']:
        col_min = df[column].min()
        col_max = df[column].max()
        col_range = col_max - col_min
        print(f"--- Range for {column} ---")
        print(f"Min: {col_min:.2f}")
        print(f"Max: {col_max:.2f}")
        print(f"Range: {col_range:.2f}\n")

column = 'ed'
frequency_table = df[column].value_counts()
percentage_table = df[column].value_counts(normalize=True) * 100

freq_df = pd.DataFrame({
    'Frequency': frequency_table,
    'Percentage (%)': percentage_table.round(2)
})

print(freq_df)
