# Step2: Data_Preprocessing
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def data_preprocessing(df):
    
    #----------------------------------------------------------------------------------------
    #Step1: Drop the duplicated values
    #Step2: Feature Engineering (optional)**
    #Step3: Split the dataset into X (seen) and y (unseen)
    #Step4: Split the dataset into train and test
    #Step5: Sagregate data into numerical and categorical
    #Step6: Build numerical_pipeline : Imputation and Scalling
    #Step7: Build categorical_pipeline : Imputation and encoding
    #Step8: Concat numerical and categorical pipeline using column transformer
    #Step9: Fit the transformer on train and test
    #Step10: Fit_resample the data so that the target column will get balanced
    #Step11: Return the preprocessed X_train,X_test,Y_train,Y_test
    #---------------------------------------------------------------------------------------
    
    #Step1: Drop the duplicated values
    df.drop_duplicates(inplace = True)
    
    #Step2: Feature Engineering (optional)**
    
    #Step3: Split the dataset into X (seen) and y (unseen)
    X = df.drop(columns = ['customerID','Churn'])
    y = df['Churn']
    
    
    #step4: Split the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = 0.3,
                                                        random_state = 1)
    

    #Step5: segregate the pipeline into numerical and categorical
        #segregate numerical columns and categorical columns    
    numerical_col = X_train.select_dtypes(exclude = 'object').columns
    categorical_col = X_train.select_dtypes(include = 'object').columns
    
    
    #Step6: Build numerical_pipeline : Imputation and Scalling
    numerical_pipeline = Pipeline([('Imputation', SimpleImputer(strategy = 'median')),
                                    ('Scaling', MinMaxScaler())
                                    ])

    
    #Step7: Build categorical_pipeline : Imputation and Encoding
    categorical_pipeline = Pipeline([('Imputation', SimpleImputer(strategy = 'most_frequent')),
                                      ('Encoding', OneHotEncoder(handle_unknown='ignore'))
                                      ])
    
    
    #Step8: Concat numerical and categorical pipeline using column transformer
    Transformer = ColumnTransformer([('numerical_pipeline',
                                    numerical_pipeline,
                                    numerical_col),
                                  ('categorical_pipeline',
                                    categorical_pipeline,
                                    categorical_col)
                                  ])
    

    #Step9: Fit the transformer on train and test
    X_train = Transformer.fit_transform(X_train)   #Seen data
    X_test = Transformer.transform(X_test)         #Unseen data
    

    #Step10: Fit_resample the data so that the target column will get balanced
    
    '''
    *Generally used with classification model
    *target column balance technique basically classified into two...
    
       - Over_sampling technique
       
       - Under_sampling technique
       

    '''
    
    X_train_resample,y_train_resample = SMOTE().fit_resample(X_train,y_train)
    
    #Step11: Return preprocess X_train,X_test,y_train,y_test
    return X_train,X_test,y_train,y_test