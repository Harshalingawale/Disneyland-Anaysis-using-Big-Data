#!/usr/bin/env python
# coding: utf-8

# ![Big data.jpg](attachment:e1b2c67b-f2cb-4c40-930d-27377e8396ed.jpg)

# # Problem statement
# Link to Dataset is [Here](https://www.kaggle.com/datasets/arushchillar/disneyland-reviews) 

# ### A company named Disneyland wants to analyze customers sentiments for review to understand visitors and to identify areas for improvement.

# ## Business Problem

# ### A company wants to analyze customer reviews to understand the sentiments of the visitors. Specifically, they aim to determine whether the reviews are positive, negative, or neutral.

# #### Importance:
# * Customer satisfaction can directly impact business success. 
# * Analyzing sentiment can help in prioritizing improvements, addressing concerns, and enhancing overall customer experience.

# ## Importing Library

# In[22]:


import pandas as hi
import nltk
import matplotlib.pyplot as mpl
import seaborn as sen
import warnings
warnings.filterwarnings('ignore')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

dis = hi.read_csv('DisneylandReviews.csv', encoding='latin1')


# ##### Here we have imported libraries and loaded our dataset.

# ## Data Exploration

# In[23]:


print(dis.tail(2))


# ##### From the above we can see first few rows

# ## Data Cleaning

# In[24]:


print("Null values in each column before removing:")
print(dis.isnull().sum().sum())


# In[25]:


dis.dropna(inplace=True)


# In[26]:


print(dis.isnull().sum().sum())


# ##### Here we have cleaned our data.

# ## Data preprocessing

# In[27]:


st_wd = set(stopwords.words('english'))
l_em = WordNetLemmatizer()

def pre_pro(tex):
    tex = tex.lower()
    tokens = nltk.word_tokenize(tex)
    tokens = [token for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if token not in st_wd]
    tokens = [l_em.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

dis['clean_text'] = dis['Review_Text'].apply(pre_pro)
print(dis['clean_text'].tail(2))


# ##### Here we have first converted tex to lowercase first,then done some tokenization tex and removed punctuations. After applying preprocessing we have displayed clean tex.

# In[28]:


tid_v = TfidfVectorizer(max_features=50)
X = tid_v.fit_transform(dis['clean_text']).toarray()
y = dis['Rating'].apply(lambda x: 'positive' if x > 3 else 'negative' if x < 3 else 'neutral')

print(f'Shape of X: {X.shape}')
print(f'Shape of y: {y.shape}')


# ##### Here we have done TF-IDF vectorization and then displayed shape.

# ## Data Visualization

# In[29]:


dis['Year_Month'] = hi.to_datetime(dis['Year_Month'], errors='coerce')
dis = dis.dropna(subset=['Year_Month'])

mpl.figure(figsize=(8, 5))
sen.countplot(x='Rating', data=dis, palette='viridis')
mpl.title('Distribution of Ratings')
mpl.xlabel('Rating')
mpl.ylabel('Count')
mpl.show()


# ##### From the above plot we can see visualize distribution of ratings.

# In[30]:


if 'Year_Month' in dis.columns:
    dis['Year_Month'] = hi.to_datetime(dis['Year_Month'], errors='coerce')
    dis = dis.dropna(subset=['Year_Month'])
else:
    print("The 'Year_Month' column is not present in the dataset.")

if 'Year_Month' in dis.columns:
    dis.set_index('Year_Month', inplace=True)
    monthly_reviews = dis['Review_ID'].resample('M').count()

    mpl.figure(figsize=(12, 6))
    monthly_reviews.plot()
    mpl.title('Number of Reviews Per Month')
    mpl.xlabel('Year-Month')
    mpl.ylabel('Number of Reviews')
    mpl.show()


# ##### Here we have visualize the number of reviews per month/year.

# In[37]:


def gen_wor(tex, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(tex)
    mpl.figure(figsize=(10, 5))
    mpl.imshow(wordcloud, interpolation='bilinear')
    mpl.axis('off')
    mpl.title(title)
    mpl.show()


# ##### From the above we have define a function to generate word clouds.

# In[39]:


positive_reviews = dis[dis['Rating'] > 3]['clean_text']
positive_text = ' '.join(positive_reviews)
gen_wor(positive_text, 'Positive Reviews Word Cloud')


# ##### We have separate the reviews into positive and combine the cleaned tex for each sentiment and displayed it.

# In[40]:


negative_reviews = dis[dis['Rating'] < 3]['clean_text']
negative_text = ' '.join(negative_reviews)
gen_wor(negative_text, 'Negative Reviews Word Cloud')


# ##### We have separate the reviews into negative and combine the cleaned tex for each sentiment and displayed it.

# In[41]:


neutral_reviews = dis[dis['Rating'] == 3]['clean_text']
neutral_text = ' '.join(neutral_reviews)
gen_wor(neutral_text, 'Neutral Reviews Word Cloud')


# ##### We have separate the reviews into neutral and combine the cleaned tex for each sentiment and displayed it.

# ## Model Training

# In[42]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ##### Know we have done model training.

# In[43]:


s_cl = SVC(kernel='linear')
s_cl.fit(X_train, y_train)


# In[44]:


s_sc = cross_val_score(s_cl, X, y, cv=5)
print(f'SVM Cross_validation Scores: {s_sc}')
print(f'Mean SVM Cross-validation Score: {s_sc.mean():.2f}')

y_pred_svm = s_cl.predict(X_test)
print('SVM Classification Report:')
print(classification_report(y_test, y_pred_svm))
print(f'SVM Accuracy: {accuracy_score(y_test, y_pred_svm):.2f}')


# In[59]:


l_g_cl = LogisticRegression(max_iter=100)
l_g_cl.fit(X_train, y_train)


# In[60]:


lgr_scr = cross_val_score(l_g_cl, X, y, cv=5)
print(f'Logistic Regression Cross-validation Scores: {lgr_scr}')
print(f'Mean Logistic Regression Cross-validation Score: {lgr_scr.mean():.2f}')

y_p_lo = l_g_cl.predict(X_test)

print('Logistic Regression Classification Report:')
print(classification_report(y_test, y_p_lo))
print(f'Logistic Regression Accuracy: {accuracy_score(y_test, y_p_lo):.2f}')


# In[61]:


ra_f_c = RandomForestClassifier(random_state=42)
ra_f_c.fit(X_train, y_train)


# In[62]:


r_sco = cross_val_score(ra_f_c, X, y, cv=5)
print(f'Random Forest Cross-validation Scores: {r_sco}')
print(f'Mean Random Forest Cross-validation Score: {r_sco.mean():.2f}')

y_pred_rf = ra_f_c.predict(X_test)

print('Random Forest Classification Report:')
print(classification_report(y_test, y_pred_rf))
print(f'Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}')


# In[63]:


pa_cla = PassiveAggressiveClassifier(max_iter=100, random_state=42)
pa_cla.fit(X_train, y_train)


# In[64]:


p_scor = cross_val_score(pa_cla, X, y, cv=5)
print(f'Passive Aggressive Classifier Cross-validation Scores: {p_scor}')
print(f'Mean Passive Aggressive Classifier Cross-validation Score: {p_scor.mean():.2f}')

y_pred_pac = pa_cla.predict(X_test)

print('Passive Aggressive Classifier Classification Report:')
print(classification_report(y_test, y_pred_pac))
print(f'Passive Aggressive Classifier Accuracy: {accuracy_score(y_test, y_pred_pac):.2f}')


# In[65]:


n_b_cla = GaussianNB()
n_b_cla.fit(X_train, y_train)


# In[66]:


n_sco = cross_val_score(n_b_cla, X, y, cv=5)
print(f'Naive Bayes Cross-validation Scores: {n_sco}')
print(f'Mean Naive Bayes Cross-validation Score: {n_sco.mean():.2f}')

y_pred_nb = n_b_cla.predict(X_test)

print('Naive Bayes Classification Report:')
print(classification_report(y_test, y_pred_nb))
print(f'Naive Bayes Accuracy: {accuracy_score(y_test, y_pred_nb):.2f}')


# In[67]:


models = ['SVM', 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'Passive Aggressive']
mean_scores = [s_sc.mean(), lgr_scr.mean(), r_sco.mean(), n_sco.mean(), p_scor.mean()]

for model, score in zip(models, mean_scores):
    print(f'{model}: Mean Cross-validation Score = {score:.2f}')

be_st_m = mean_scores.index(max(mean_scores))
be_st_n = models[be_st_m]
be_st_sc = mean_scores[be_st_m]

print(f'\nBest Model: {be_st_n}')
print(f'Best Mean Cross-validation Score: {be_st_sc:.2f}')


# ##### Here we have compared all models based on mean cross-validation scores and displayed best model.

# ## Hyperparameter Tunning for best model

# In[68]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lg_p_g = {
    'penalty': ['l1', 'l2', 'elasticnet', None],
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['lbfgs', 'liblinear', 'saga'],
    'max_iter': [100, 200, 300]
}
l_g_cl = LogisticRegression()

logreg_grid_search = GridSearchCV(estimator=l_g_cl, param_grid=lg_p_g, cv=5, scoring='accuracy')
logreg_grid_search.fit(X_train, y_train)
be_lo_c = logreg_grid_search.best_estimator_

y_p_lo = be_lo_c.predict(X_test)

co_ma_l = confusion_matrix(y_test, y_p_lo)

mpl.figure(figsize=(10, 7))
sen.heatmap(co_ma_l, annot=True, fmt='d', cmap='Blues', xticklabels=be_lo_c.classes_, yticklabels=be_lo_c.classes_)
mpl.xlabel('Predicted Labels')
mpl.ylabel('True Labels')
mpl.title('Confusion Matrix for Logistic Regression')
mpl.show()


# In[69]:


print('Logistic Regression Classification Report:')
print(classification_report(y_test, y_p_lo))
print(f'Logistic Regression Accuracy: {accuracy_score(y_test, y_p_lo):.2f}')


# ## Conclusion

# #### By completing these objectives, we will deliver an end-to-end NLP pipeline that can help Disneyland analyze customer sentiment effectively, leading to actionable insights for improving the overall customer experience.

# In[ ]:




