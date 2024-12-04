## Imports
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from xgboost import XGBRegressor, XGBClassifier
from statistics import mean
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder

# Missing values imputer function with a given algorithm
def impute_missing_values(data, target_column, algorithm):
    # Separating the missing values from the non missing values
    available_data = data[data[target_column].notna()]
    missing_data = data[data[target_column].isna()]

    # Diagnóstico inicial
    print(f"\nImputando valores para coluna: {target_column}")
    print(f"Linhas disponíveis para treino: {len(available_data)}")
    print(f"Linhas com valores ausentes: {len(missing_data)}")

    # Verificar se há dados suficientes para imputação
    if len(available_data) == 0 or len(missing_data) == 0:
        print(f"Sem dados suficientes para imputar valores na coluna {target_column}")
        return data

    # Separating the target column from the rest
    X_available = available_data.drop(columns=[target_column])
    y_available = available_data[target_column]

    # Garantir consistência entre colunas
    X_available = X_available.select_dtypes(include=["number"])
    X_missing = missing_data.drop(columns=[target_column]).select_dtypes(include=["number"])
    common_columns = X_available.columns.intersection(X_missing.columns)
    X_available = X_available[common_columns]
    X_missing = X_missing[common_columns]

    # Verificar se ainda há colunas suficientes após alinhamento
    if X_available.shape[1] == 0:
        print(f"Sem colunas disponíveis para imputar na coluna {target_column}")
        return data

    # Training the model with the available data
    model = algorithm
    model.fit(X_available, y_available)

    # Prediting the missing values
    predicted_values = model.predict(X_missing)

    # Inputing the missing values with the predictions
    data.loc[data[target_column].isna(), target_column] = predicted_values

    return data

def impute_missing_values_with_fallback(data, target_column, algorithm):
    available_data = data[data[target_column].notna()]
    missing_data = data[data[target_column].isna()]

    if len(available_data) == 0 or len(missing_data) == 0:
        print(f"Sem dados suficientes para imputar a coluna '{target_column}', usando mediana.")
        data[target_column].fillna(data[target_column].median(), inplace=True)
        return data

    X_available = available_data.drop(columns=[target_column]).select_dtypes(include=["number"])
    y_available = available_data[target_column]
    X_missing = missing_data.drop(columns=[target_column]).select_dtypes(include=["number"])

    common_columns = X_available.columns.intersection(X_missing.columns)
    X_available = X_available[common_columns]
    X_missing = X_missing[common_columns]

    if X_available.shape[1] == 0 or X_missing.shape[1] == 0:
        print(f"Sem colunas suficientes para imputar '{target_column}', usando mediana.")
        data[target_column].fillna(data[target_column].median(), inplace=True)
        return data

    model = algorithm
    model.fit(X_available, y_available)
    predicted_values = model.predict(X_missing)
    data.loc[data[target_column].isna(), target_column] = predicted_values
    return data

def check_missing_values(data, step_name):
    print(f"\n{step_name}: Valores ausentes restantes:")
    print(data.isnull().sum()[data.isnull().sum() > 0])


# Outlier removal function
def handle_outliers(data, target_column):
    lower_quantile = 0.25
    upper_quantile = 0.75
    multiplier = 1.5
    # Calculate Q1, Q3, and IQR
    Q1 = data[target_column].quantile(lower_quantile)
    Q3 = data[target_column].quantile(upper_quantile)
    IQR = Q3 - Q1
    
    # Define outlier bounds
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    # Replace outliers with NaN
    data[target_column] = data[target_column].apply(
        lambda x: np.nan if x < lower_bound or x > upper_bound else x
    )
    
    return data

def scale_numerical(column, X_train, X_val, scaler):
    # Certifique-se de que a coluna é numérica
    if not pd.api.types.is_numeric_dtype(X_train[column]):
        print(f"A coluna '{column}' não é numérica e será ignorada.")
        return

    # Escalona os dados e substitui os valores na coluna
    try:
        X_train[column] = scaler.fit_transform(X_train[[column]].values.reshape(-1, 1))
        X_val[column] = scaler.transform(X_val[[column]].values.reshape(-1, 1))
    except ValueError as e:
        print(f"Erro ao escalonar a coluna '{column}': {e}")


# Ordinal encoder function
def categorical_ordinal_encode(X_train, X_val,column):
    # Define a function to categorize each carrier based on its claim count
    count = X_train['Carrier Name'].value_counts()
    def categorize_claims(count):
        if count >= 40000:
            return 2
        elif 4000 <= count < 40000:
            return 1
        else:
            return 0

    # Apply the categorization to create a mapping dictionary
    carrier_category_map = count.apply(categorize_claims)

    # Map the `Carrier Name` to the new `Carrier Claim Category`
    X_train['Carrier Claim Category'] = X_train['Carrier Name'].map(carrier_category_map)


# Categorical encoder function
def categorical_prop_encode(X_train, X_val, features):
    for col in features:
        proportion = X_train[col].value_counts(normalize = True)  # Get the porportion of each category
        X_train[col] = X_train[col].map(proportion)  # Map the porportions in the column
        X_val[col] = X_val[col].map(proportion) # Do the same for the valid dataset
        X_val[col] = X_val[col].fillna(0)  # Handle categories in X_val not seen in X_train with 0

# Structure TODO:
# SKFold Loop
    # Split into current train/valid

    # Impute missing values

    # Outlier removal on train (only numerical)

    # Scaling (only numerical)

    # Categorical Prop Encoding
    # Categorical Ordinal Encoding

    # Fit model on train
    # Predict valid
def cv_scores(model, X, y, num_features, cat_features, num_imputing_algorithm= XGBRegressor() , cat_imputing_algorithm = XGBClassifier(), scalling_outlier = True, scaler = MinMaxScaler()):
    ''' Takes as argument the model used, the predictors and the target, the models used for imputing numerical and categorical 
      features, if any scaling and outlier removed should be performed, and what scaling method should be used.
     Then it returns the results obtained from the stratified cross validation for the given model'''
    
    skf = StratifiedKFold(n_splits=5)
    
    # Generating the lists to store our results
    precision_scores_train = []
    precision_scores_val = []   
    recall_scores_train = []  
    recall_scores_val = []
    f1_scores_train = []    
    f1_scores_val = []
    index = [f'Fold {i}' for i in range(1,6)]
    index.append("Average")
    
    for train_index, test_index in skf.split(X, y):
        # Dividing our data in validation and train
        X_train, X_val = X.iloc[train_index], X.iloc[test_index]
        y_train, y_val = y.iloc[train_index], y.iloc[test_index]

        #Filling num missing values
        for column in num_features:
            impute_missing_values(X_train, column, num_imputing_algorithm)
            impute_missing_values(X_val, column, num_imputing_algorithm)

        for column in cat_features:
            #Filling cat missing values
            impute_missing_values(X_train, column, cat_imputing_algorithm)
            impute_missing_values(X_val, column, cat_imputing_algorithm)

        # Removing inconsistencies on the train
        inconsistent = X_train[(X_train['Age at Injury'] > 80) | (X_train["Age at Injury"] < 16)].index
        X_train.drop(inconsistent, inplace=True)
        y_train.drop(inconsistent, inplace=True)

        #Performing scaling and outlier treatment dependent on the boolean
        if scalling_outlier:
            for column in num_features:
                handle_outliers(X_train, column)
                scale_numerical(column,X_train, X_val, scaler)

        """# Creating an ordinal variable
        categorical_ordinal_encode(X_train, X_val, num_features)"""

        # Categorical Prop Encoding
        for cat_feature in cat_features:
            categorical_prop_encode(X_train, X_val, cat_feature)

        # Training the classification model
        model.fit(X_train, y_train)

        
        # Making the predictions for the training and validation data
        pred_train = model.predict(X_train)
        pred_val = model.predict(X_val)
    
        
        # Calculating and storing the scores
        precision_scores_train.append(precision_score(y_train, pred_train, average='macro'))
        precision_scores_val.append(precision_score(y_val, pred_val, average='macro'))
        recall_scores_train.append(recall_score(y_train, pred_train, average='macro'))
        recall_scores_val.append(recall_score(y_val, pred_val, average='macro'))
        f1_scores_train.append(f1_score(y_train, pred_train, average='macro'))
        f1_scores_val.append(f1_score(y_val, pred_val, average='macro'))

    
    precision_scores_train.append(mean(precision_scores_train))
    precision_scores_val.append(mean(precision_scores_val))
    recall_scores_train.append(mean(recall_scores_train))
    recall_scores_val.append(mean(recall_scores_val))
    f1_scores_train.append(mean(f1_scores_train))
    f1_scores_val.append(mean(f1_scores_val))


    # Storing the results in a dataframe
    model_results = pd.DataFrame(data={
        'Train_precision': precision_scores_train,
        'Test_precision': precision_scores_val,
        'Train_recall': recall_scores_train,
        'Test_recall': recall_scores_val,
        'Train_f1_score': f1_scores_train,
        'Test_f1_score': f1_scores_val,
    }, index=index)
    
    return model_results


le = LabelEncoder()

def test_prediction(model, X, y, num_features, cat_features, data_test, 
                    num_inputing_algorithm=XGBRegressor(), 
                    cat_inputing_algorithm=XGBClassifier(), 
                    scaler=MinMaxScaler()):
    
    for column in num_features:
        impute_missing_values_with_fallback(X, column, num_inputing_algorithm)
        impute_missing_values_with_fallback(data_test, column, num_inputing_algorithm)
    impute_missing_values_with_fallback(X, "Alternative Dispute Resolution", cat_inputing_algorithm)
    impute_missing_values_with_fallback(data_test, "Alternative Dispute Resolution", cat_inputing_algorithm)

    # Etapa 2: Remoção de inconsistências
    inconsistent = X[(X['Age at Injury'] > 80) | (X["Age at Injury"] < 16)].index
    X.drop(inconsistent, inplace=True)
    y.drop(inconsistent, inplace=True)

    # Etapa 3: Tratamento de outliers
    for column in num_features:
        handle_outliers(X, column)
        handle_outliers(data_test, column)

    # Etapa 4: Reimputação Após Outliers
    for column in num_features:
        impute_missing_values_with_fallback(X, column, num_inputing_algorithm)
        impute_missing_values_with_fallback(data_test, column, num_inputing_algorithm)

    # Etapa 5: Escalonamento
    for column in num_features:
        # Diagnóstico antes do escalonamento
        if X[column].isnull().sum() > 0:
            impute_missing_values_with_fallback(X, column, num_inputing_algorithm)
        scale_numerical(column, X, data_test, scaler)

    """# Creating an ordinal variable
    for num_feature in num_features:
        categorical_ordinal_encode(X, data_test,num_feature)"""

    # Categorical Prop Encoding
    for cat_feature in cat_features:
        categorical_prop_encode(X, data_test, cat_feature)
    
    # Fitting the model
    model.fit(X, y)

    # Using the model to make prediction on the test dataset
    pred_test = model.predict(data_test)

    # Inversing the encoding of our target variable 
    pred_test = le.inverse_transform(pred_test)

    # Making a dataframe with the indexes of data_test and predictions converted back to strings
    submission_df = pd.DataFrame({
        "Claim Injury Type": pred_test
    }, index=data_test.index)
    
    return submission_df


    