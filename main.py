from src.data_ingestion import data_ingestion 
from src.data_preprocessing import data_preprocessing



def main():
    
    
    #Step1: Data Ingestion
    df = data_ingestion()
    print(df.columns)
    
    
    #Step2: Data Preprocessing
    X_train, X_test, y_train, y_test  =  data_preprocessing(df)
    print(X_train)
    
main()


