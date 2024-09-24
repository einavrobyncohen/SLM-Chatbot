import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer

#######################################################################

## GENERATING THE MOCK DATASET## 
#This is mock dataset, assuming some features that might appear in the actual data. This includes both structured and unstructured data. 
data = {
    'Ticket ID': [1234, 5678, 9101, 1121],
    'Ticket Type': ['Network', 'Software', 'Hardware', 'Security'], 
    'Priority': [1,3,2,5],
    'Description': [
        'Issue with network connectivity.', 
        'Software update caused a bug',
        'Hardware malfunction in server',
        'Security breach detected'
    ],
    'Approved': ['yes', 'no', 'yes', 'no']
}

#######################################################################

## Creating a csv file ##
#df = pd.DataFrame(data)
#df.to_csv('../data/ticket_data.csv', index=False)

#######################################################################

## Text Vectorization for Unstructured Data ## 

df = pd.read_csv('../data/ticket_data.csv')
print(f'Original dataset: {df}')

vectorizer = TfidfVectorizer()
#Vectotizing the description column in the mock dataset
tfidf_matrix = vectorizer.fit_transform(df['Description'])
#Converting the TF-IDF matrix to a dataframe for easier inspection and then adding the TF-IDF features back to the original df
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
df_tfidf = pd.concat([df, tfidf_df], axis=1)
df_tfidf.drop(columns=['Description'], inplace=True)

print(f'dataset with TF-IDF vectors: {df_tfidf}')

df_tfidf.to_csv('../data/ticket_data_vectorized.csv', index=False)