import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def split_input_output(data, target_col):
    X = data.drop(columns = target_col,
                  axis = 1)

    y = data[target_col]

    return X, y

def split_train_test(X, y, test_size, random_state=None):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size = test_size,
        stratify = y,
        random_state = random_state
    )

    return X_train, X_test, y_train, y_test

def split_num_cat(X, num_col, cat_col):
    X_num = X[num_col]
    X_cat = X[cat_col]

    return X_num, X_cat

def fit_num_imputer(X_num):
    num_imputer = SimpleImputer(missing_values = np.nan,
                                strategy = 'median')
    num_imputer.fit(X_num)

    return num_imputer

def transform_num_imputer(X_num, num_imputer):
    X_num_imputed = X_num.copy()

    X_num_imputed = pd.DataFrame(
        num_imputer.transform(X_num_imputed),
        columns = X_num.columns,
        index = X_num.index
    )

    return X_num_imputed

def fit_cat_imputer(X_cat):
    cat_imputer = SimpleImputer(missing_values = np.nan,
                                strategy = 'constant',
                                fill_value = 'KOSONG')

    cat_imputer.fit(X_cat)

    return cat_imputer

def transform_cat_imputer(X_cat, cat_imputer):
    X_cat_imputed = X_cat.copy()

    X_cat_imputed = pd.DataFrame(
        cat_imputer.transform(X_cat_imputed),
        columns = X_cat.columns,
        index = X_cat.index
    )

    return X_cat_imputed

def split_cat_data(X_cat, ohe_col, le_col):
    X_cat_ohe = X_cat[ohe_col]
    X_cat_le = X_cat[le_col]

    return X_cat_ohe, X_cat_le

def fit_ohe_encoder(X_cat_ohe):
    categories = []
    for col in X_cat_ohe.columns:
        unique_value_raw = list(set(X_cat_ohe[col]))

        unique_value = [val for val in unique_value_raw if val != 'KOSONG']

        categories.append(unique_value)

    ohe_encoder = OneHotEncoder(categories = categories,
                                handle_unknown = 'ignore')

    ohe_encoder.fit(X_cat_ohe)

    return ohe_encoder

def transform_ohe_encoder(X_cat_ohe, ohe_encoder):
    X_cat_ohe_encoded = X_cat_ohe.copy()

    ohe_col = []
    for cols in ohe_encoder.categories_:
        ohe_col.extend(cols)

    X_cat_ohe_encoded = pd.DataFrame(
        ohe_encoder.transform(X_cat_ohe_encoded).toarray(),
        columns = ohe_col,
        index = X_cat_ohe.index
    )

    return X_cat_ohe_encoded

def transform_le_encoder(X_cat_le):
    loan_grade_col = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'KOSONG']
    loan_grade_mapper = {val:i+1 for i, val in enumerate(loan_grade_col)}

    cb_person_col = ['N', 'KOSONG', 'Y']
    cb_person_col_mapper = {val:i+1 for i, val in enumerate(cb_person_col)}

    mapper = {'loan_grade': loan_grade_mapper,
              'cb_person_default_on_file': cb_person_col_mapper}

    X_cat_le_encoded = X_cat_le.copy()
    for col in X_cat_le_encoded.columns:
        X_cat_le_encoded[col] = X_cat_le_encoded[col].map(mapper[col])

    return X_cat_le_encoded

def fit_scaler(X_concat):
    scaler = StandardScaler()
    scaler.fit(X_concat)
    return scaler

def transform_scaler(X_concat, scaler):
    X_concat = X_concat.copy()

    X_concat_scaled = pd.DataFrame(
        scaler.transform(X_concat),
        columns = X_concat.columns,
        index = X_concat.index
    )

    return X_concat_scaled

def fit_preprocess_data(X_train):
    NUMERICAL_COL = ['person_age', 'person_income', 'person_emp_length',
                    'loan_amnt', 'loan_int_rate', 'loan_percent_income',
                    'cb_person_cred_hist_length']

    CATEGORICAL_COL = ['person_home_ownership', 'loan_intent',
                    'loan_grade', 'cb_person_default_on_file']

    OHE_COL = ['person_home_ownership', 'loan_intent']

    LE_COL = ['loan_grade', 'cb_person_default_on_file']

    X_train_num, X_train_cat = split_num_cat(
        X = X_train,
        num_col = NUMERICAL_COL,
        cat_col = CATEGORICAL_COL
    )

    num_imputer = fit_num_imputer(X_num = X_train_num)
    
    X_train_num_imputed = transform_num_imputer(
        X_num = X_train_num,
        num_imputer = num_imputer
    )

    cat_imputer = fit_cat_imputer(X_cat = X_train_cat)

    X_train_cat_imputed = transform_cat_imputer(
        X_cat = X_train_cat,
        cat_imputer = cat_imputer
    )

    X_train_cat_ohe, X_train_cat_le = split_cat_data(
        X_cat = X_train_cat_imputed,
        ohe_col = OHE_COL,
        le_col = LE_COL
    )

    ohe_encoder = fit_ohe_encoder(X_cat_ohe = X_train_cat_ohe)

    X_train_cat_ohe_encoded = transform_ohe_encoder(
        X_cat_ohe = X_train_cat_ohe,
        ohe_encoder = ohe_encoder
    )

    X_train_cat_le_encoded = transform_le_encoder(X_cat_le = X_train_cat_le)

    X_train_cat_encoded = pd.concat((X_train_cat_ohe_encoded, X_train_cat_le_encoded), axis=1)

    X_train_concat = pd.concat((X_train_num_imputed, X_train_cat_encoded), axis=1)

    scaler = fit_scaler(X_concat = X_train_concat)

    return num_imputer, cat_imputer, ohe_encoder, scaler

def transform_preprocess_data(X,
                    num_imputer, cat_imputer,
                    ohe_encoder,
                    scaler):
    num_col = ['person_age', 'person_income', 'person_emp_length',
                    'loan_amnt', 'loan_int_rate', 'loan_percent_income',
                    'cb_person_cred_hist_length']

    cat_col = ['person_home_ownership', 'loan_intent',
                    'loan_grade', 'cb_person_default_on_file']

    ohe_col = ['person_home_ownership', 'loan_intent']

    le_col = ['loan_grade', 'cb_person_default_on_file']

    X = X.copy()

    X_num, X_cat = split_num_cat(X = X,
                                 num_col = num_col,
                                 cat_col = cat_col)

    X_num_imputed = transform_num_imputer(X_num = X_num,
                                          num_imputer = num_imputer)

    X_cat_imputed = transform_cat_imputer(X_cat = X_cat,
                                          cat_imputer = cat_imputer)

    X_cat_ohe, X_cat_le = split_cat_data(X_cat = X_cat_imputed,
                                         ohe_col = ohe_col,
                                         le_col = le_col)

    X_cat_ohe_encoded = transform_ohe_encoder(X_cat_ohe = X_cat_ohe,
                                              ohe_encoder = ohe_encoder)

    X_cat_le_encoded = transform_le_encoder(X_cat_le = X_cat_le)

    X_cat_encoded = pd.concat((X_cat_ohe_encoded, X_cat_le_encoded), axis=1)

    X_concat = pd.concat((X_num_imputed, X_cat_encoded), axis=1)

    X_clean = transform_scaler(X_concat = X_concat,
                               scaler = scaler)

    return X_clean