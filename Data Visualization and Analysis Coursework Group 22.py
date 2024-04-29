import numpy as np
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import chi2_contingency
print ('setup complete')

data = pd.read_csv("C:/Users/Ramim/Downloads/mental health data set/Mental Health Dataset.csv")
print(data.head())

data.info()

data= data.dropna()
print (data.isnull().sum())

data = data.drop_duplicates()
print (data.duplicated().sum())

sample_data = data.sample(n=20000)
print(len(sample_data))

sample_data['Timestamp'] = pd.to_datetime(sample_data['Timestamp'])
sample_data.describe()

data = sample_data

sns.set(style="whitegrid")
plt.figure(figsize=(12, 18))

plt.subplot(3, 2, 1)
sns.countplot(y='Gender', data=data, order = data['Gender'].value_counts().index)
plt.title('Gender Distribution')

plt.subplot(3, 2, 2)
sns.countplot(y='treatment', data=data)
plt.title('Treatment Distribution')


plt.subplot(3, 2, 3)
sns.countplot(y='Growing_Stress', data=data)
plt.title('Growing Stress Reports')

plt.subplot(3, 2, 4)
sns.countplot(y='Mental_Health_History', data=data)
plt.title('Self-reported Mental Health History')

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(data=data, x='Gender')
plt.title('Distribution of Gender')
plt.xticks(rotation=45)
plt.show()

treatment_by_gender = pd.crosstab(data['Gender'], data['treatment'])
print("\nTreatment by Gender:\n", treatment_by_gender)

treatment_by_gender_proportions = treatment_by_gender.div(treatment_by_gender.sum(1), axis=0)
print("\nProportions of Treatment by Gender:\n", treatment_by_gender_proportions)

treatment_by_gender_proportions.plot(kind='bar', stacked=True)
plt.title('Proportion of Individuals Seeking Treatment by Gender')
plt.ylabel('Proportion')
plt.xlabel('Gender')
plt.xticks(rotation=45)
plt.legend(title='Treatment', labels=['No', 'Yes'])

occupation_treatment_ct = pd.crosstab(data['Occupation'], data['treatment'])

occupation_treatment_prop = occupation_treatment_ct.div(occupation_treatment_ct.sum(1), axis=0)
occupation_treatment_prop.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Treatment-Seeking Behavior by Occupation')
plt.xlabel('Occupation')
plt.ylabel('Proportion Seeking Treatment')
plt.legend(title='Treatment', labels=['No', 'Yes'])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


chi2, p, dof, expected = chi2_contingency(occupation_treatment_ct)
print(f"Chi-square Statistic: {chi2}, p-value: {p}")

cross_tab_history = pd.crosstab(data['Occupation'], data['Mental_Health_History'])

cross_tab_treatment = pd.crosstab(data['family_history'], data['treatment'])

cross_tab_history.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Personal Mental Health History vs. Occupation')
plt.xlabel('Occupation')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Personal Mental Health History')

cross_tab_treatment.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Treatment Seeking Behavior vs. Family History of Mental Health Issues')
plt.xlabel('Family History of Mental Health Issues')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Treatment Seeking Behavior')

cross_tab_stress = pd.crosstab(data['Work_Interest'], data['Growing_Stress'])
cross_tab_moodswings = pd.crosstab(data['Days_Indoors'], data['Mood_Swings'])

cross_tab_stress.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Reported Stress Levels vs. Work Interest')
plt.xlabel('Work Interest')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Reported Stress')

cross_tab_moodswings.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Mood swings vs. Days Indoors')
plt.xlabel('Days Indoors')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Mood Swings')
plt.show()

plt.figure(figsize=(20, 10))

plt.subplot(2, 2, 1)
data['Gender'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Gender Distribution')

plt.subplot(2, 2, 2)
data['Mood_Swings'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Mood Swings Distribution')

plt.subplot(2, 2, 3)
data['Changes_Habits'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Change in Habits Distribution')

plt.subplot(2, 2, 4)
data['family_history'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Family History of Mental Health Issues Distribution')

plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 10))

plt.subplot(2, 2, 1)
sns.violinplot(x='Mental_Health_History', y='Days_Indoors', data=data)
plt.title('Mental Health History Distribution by Days Indoors')

plt.subplot(2, 2, 2)
sns.violinplot(x='family_history', y='Days_Indoors', data=data)
plt.title('Family History of Mental Health Issues Distribution by Days Indoors')

plt.subplot(2, 2, 3)
sns.violinplot(x='treatment', y='Days_Indoors', data=data)
plt.title('Treatment Distribution by Days Indoors')

plt.subplot(2, 2, 4)
sns.violinplot(x='Gender', y='Days_Indoors', data=data)
plt.title('Gender Distribution by Days Indoors')

plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 10))

plt.subplot(2, 2, 1)
sns.boxplot(x='Mental_Health_History', y='Occupation', data=data)
plt.title('Mental Health History Distribution by Occupation')

plt.subplot(2, 2, 2)
sns.boxplot(x='family_history', y='Occupation', data=data)
plt.title('Family History of Mental Health Issues Distribution by Occupation')

plt.subplot(2, 2, 3)
sns.boxplot(x='Coping_Struggles', y='Occupation', data=data)
plt.title('Coping to struggles Distribution by Occupation')

plt.subplot(2, 2, 4)
sns.boxplot(x='Gender', y='Occupation', data=data)
plt.title('Gender Distribution by Occupation')

plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 10))

plt.subplot(2, 2, 1)
sns.lineplot(x='care_options', y='Days_Indoors', data=data)
plt.title('Care Options Distribution by Days Indoors')

plt.subplot(2, 2, 2)
sns.lineplot(x='family_history', y='Days_Indoors', data=data)
plt.title('Family History of Mental Health Issues Distribution by Days Indoors')

plt.subplot(2, 2, 3)
sns.lineplot(x='Social_Weakness', y='Days_Indoors', data=data)
plt.title('Social Weakness Distribution by Days Indoors')

plt.subplot(2, 2, 4)
sns.lineplot(x='Gender', y='Days_Indoors', data=data)
plt.title('Gender Distribution by Days Indoors')

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.countplot(x='Occupation', data=data)
plt.title('Occupation Distribution')
plt.xlabel('Occupation')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.countplot(x='Mood_Swings', data=data)
plt.title('Mood Swings Distribution')
plt.xlabel('Mood Swings')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Grouped bar chart to compare Gender and Mood_Swings
plt.figure(figsize=(8, 6))
sns.countplot(x='Gender', hue='Mood_Swings', data=data)
plt.title('Comparison of Mood Swings by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(title='Mood Swings')
plt.show()

# Grouping the data by country and care options and counting occurrences
country_care_counts = data.groupby(['Country', 'care_options']).size().unstack(fill_value=0)

# Create a cross-tabulation of Work_Interest and Social_Weakness
cross_tab = pd.crosstab(data['Work_Interest'], data['Social_Weakness'])

# Heatmap of Work Interest vs Social Weakness
plt.figure(figsize=(8, 6))
sns.heatmap(cross_tab, annot=True, fmt='d', cmap='Blues')
plt.title('Heatmap of Work Interest vs Social Weakness')
plt.xlabel('Social Weakness')
plt.ylabel('Work Interest')
plt.show()

# Create a cross-tabulation of Occupation and Days Indoors
cross_tab_occ_days = pd.crosstab(data['Occupation'], data['Days_Indoors'])

# Heatmap of Occupation vs Days Indoors
plt.figure(figsize=(10, 8))
sns.heatmap(cross_tab_occ_days, annot=True, fmt='d', cmap='coolwarm')
plt.title('Heatmap of Occupation vs Days Indoors')
plt.xlabel('Days Indoors')
plt.ylabel('Occupation')
plt.show()

# Create a cross-tabulation of Mental Health Interview and Care Options
cross_tab_interview_care = pd.crosstab(data['mental_health_interview'], data['care_options'])

# Heatmap of Mental Health Interview vs Care Options
plt.figure(figsize=(10, 8))
sns.heatmap(cross_tab_interview_care, annot=True, fmt='d', cmap='coolwarm')
plt.title('Heatmap of Mental Health Interview vs Care Options')
plt.xlabel('Care Options')
plt.ylabel('Mental Health Interview')
plt.show()
