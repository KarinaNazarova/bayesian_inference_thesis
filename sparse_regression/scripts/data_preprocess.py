import pandas as pd

import numpy as np
import pandas as pd
import nest_asyncio
nest_asyncio.apply()
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd


###### PREPARE T1 DATA ######
def prepare_t1_data(input_file, output_file):
    output_dir = 'sparse_regression/data/processed/'
    os.makedirs(output_dir, exist_ok=True)
  
    # Load the data
    data_original = pd.read_excel(input_file, index_col=None, header=0)  
    print('Downloaded Dataset')
    print('')

    # Access data for T1 only 
    data_t1 = data_original.loc[:, :'VAR00005']

    # Remove empty columns from T1 data
    data_t1.drop(['VAR00006', 'VAR00001', 'VAR00002', 'VAR00003', 'VAR00004', 'VAR00005'], axis=1, inplace=True)


    # Replace missing values with 999
    mark_for_missing_values = 999
    data_t1.fillna(mark_for_missing_values, inplace=True)
    total_missing_values = data_t1.isin([mark_for_missing_values]).sum().sum()
    print(f'Number of missing values: {total_missing_values}')

    # Replace missing values with nan
    data_t1.replace(999, np.nan, inplace=True)
    total_missing_values = data_t1.isna().sum().sum()

    print(f'Number of missing values: {total_missing_values}')

    ################# COMBINE STRESSFUL EVENTS AND THEIR IMPACT ###########################
    # Multiply the event occurence with corresponding impact and combine it into a new column
    new_columns = {}
    for i in range(1, 61):
        eev_fre = f'eev{i}fre'
        eev_imp = f'eev{i}imp'
        eev_product_col = f'eev{i}_product'
        
        # if event didn't happen, fill in the corresponding impact with 0
        data_t1.loc[data_t1[eev_fre] == 0, eev_imp] = 0

        new_columns[eev_product_col] = data_t1[eev_fre] * data_t1[eev_imp]

    new_columns_df = pd.DataFrame(new_columns)
    data_t1 = pd.concat([data_t1, new_columns_df], axis=1)
    # defrag the df
    data_t1 = data_t1.copy()

    # Drop redundant ISECA columns
    data_t1.drop('Id', axis=1, inplace=True)
    # Drop the columns for events and their impact as they are combined into a new column
    data_t1.drop(columns=[f'eev{i}fre' for i in range(1, 61)] + [f'eev{i}imp' for i in range(1, 61)], inplace=True)

    total_missing_values = data_t1.isna().sum().sum()
    print(f'Number of missing values after combining ISECA: {total_missing_values}')


    # DROP MISSING ENTRIES FROM CDI
    data_t1.dropna(subset=[f'cdi{i}' for i in range(1, 28)], inplace=True)
    total_missing_values = data_t1.isna().sum().sum()

    print(f'Number of missing values after removing incomplete CDI cases: {total_missing_values}')


    ### LOGICAL IMPUTATION #### 
    # For children who do not go to institution, fill in missing values for institution_name and time_in_institution as 0
    data_t1.loc[data_t1['group_type'].isin([1]), 'institution_name'] = 0
    data_t1.loc[data_t1['group_type'].isin([1]), 'time_in_institution'] = 0

    total_missing_values = data_t1.isna().sum().sum()
    print(f'Number of missing values after substituting institution name/time with 0 for those who don\'t go to institution: {total_missing_values}')

    # for kids who don't go to school, fill in missing values for school_name as 0
    data_t1.loc[data_t1['goes_to_school'].isin([2]), 'school_name'] = 0

    total_missing_values = data_t1.isna().sum().sum()
    print(f'Number of missing values after substituting school name with 0 for those who don\'t go to school: {total_missing_values}')

    # for kids who don't have job, fill in missing values for earnings as 0
    data_t1.loc[data_t1['has_job'].isin([2]), 'earnings'] = 0

    total_missing_values = data_t1.isna().sum().sum()
    print(f'Number of missing values after substituting earnings with 0 for those who don\'t have a job: {total_missing_values}')


    # CALCULATE CDI LABEL
    data_t1['cdi_label'] = (data_t1.loc[:, 'cdi1':'cdi27'].sum(axis=1) >= 19).astype(int)

    output_file = os.path.join(output_dir, output_file)
    data_t1.to_excel(output_file, index=False)

    print(f"Data saved to {output_file}")
   

###### END PREPARE T1 DATA ######

###### COMBINE THE QUESTIONNAIRES ######
def combine_questionnaires(input_file, output_file):
    # Load the data
    data_t1 = pd.read_excel(input_file, index_col=None, header=0)  
    print('Loaded The Clean Dataset. Ready to Combine Tests.')
    print('')

    ################# COMBINE TESTS ###########################
    # Life satisfaction
    # Subscale self (10 items): 1,6,11,15,20,25,29,35,40,44  
    # Comparative self (8 items): 2,7,12,16,21,30,36,45  
    # Non-violence (4 items): 8,22,31,47  
    # Family (11 items): 3,9,13,17,23,26,32,37,41,46,50  
    # Friendship (10 items): 4,10,18,24,27,33,38,39,42,48  
    # Satisfaction with school (7 items): 5,14,19,28,34,43,49  
    ls_self_items = [1, 6, 11, 15, 20, 25, 29, 35, 40, 44]
    ls_comparative_self = [2, 7, 12, 16, 21, 30, 36, 45]
    ls_non_violence = [8, 22, 31, 47]
    ls_family = [3, 9, 13, 17, 23, 26, 32, 37, 41, 46, 50]
    ls_friendship = [4, 10, 18, 24, 27, 33, 38, 39, 42, 48]
    ls_school = [5, 14, 19, 28, 34, 43, 49]

    ls_columns_self = [f'emsv{suffix}' for suffix in ls_self_items]
    ls_columns_comparative_self = [f'emsv{suffix}' for suffix in ls_comparative_self]
    ls_columns_non_violence = [f'emsv{suffix}' for suffix in ls_non_violence]
    ls_columns_family = [f'emsv{suffix}' for suffix in ls_family]
    ls_columns_friendship = [f'emsv{suffix}' for suffix in ls_friendship]
    ls_columns_school = [f'emsv{suffix}' for suffix in ls_school]

    # Calculate total score for each subcategory 
    data_t1['ls_self'] = data_t1[ls_columns_self].sum(axis=1)
    data_t1['ls_comparative_self'] = data_t1[ls_columns_comparative_self].sum(axis=1)
    data_t1['ls_non_violence'] = data_t1[ls_columns_non_violence].sum(axis=1)
    data_t1['ls_family'] = data_t1[ls_columns_family].sum(axis=1)
    data_t1['ls_friendship'] = data_t1[ls_columns_friendship].sum(axis=1)
    data_t1['ls_school'] = data_t1[ls_columns_school].sum(axis=1)

    # PANAS 
    # Calculate total score for negative and positive affects
    panas_positive_values = [1, 3, 4, 5, 7, 8, 10, 11, 12, 14, 15, 18, 21, 22, 23, 24, 29, 35, 36, 39]
    panas_negative_values = [2,6,9,13,16,17,19,20,25,26,27,28,30,31,32,33,34,37,38,40]

    panas_positive_columns = [f'ea{pos}' for pos in panas_positive_values]
    panas_negative_columns = [f'ea{neg}' for neg in panas_negative_values]

    data_t1['panas_positive'] = data_t1[panas_positive_columns].sum(axis=1)
    data_t1['panas_negative'] = data_t1[panas_negative_columns].sum(axis=1)


    # CDI 
    # F-I(Social withdrawal): 23,15,26,3,27,5.  
    # F-II(Anhedonia-Asthenia): 10,16,18,11,1,6,17,9.  
    # F-III(Incompetence/Maladjustment): 7,4,8,25,14.  
    # F-IV(Negative Self-Esteem): 22,21,12,20.  
    # F-V(Sleep and Appetite Disturbances): 2,19,13,24.

    # Define indices for each subcategory 
    social_withdrawal = [23, 15, 26, 3, 27, 5]
    anhedonia_asthenia = [10, 16, 18, 11, 1, 6, 17, 9]
    incompetence_maladjustment = [7, 4, 8, 25, 14]
    negative_self_esteem = [22, 21, 12, 20]
    sleep_appetite_disturbances = [2, 19, 13, 24]

    # Construct full names for each subcategory
    columns_social_withdrawal = [f'cdi{idx}' for idx in social_withdrawal]
    columns_anhedonia_asthenia = [f'cdi{idx}' for idx in anhedonia_asthenia]
    columns_incompetence_maladjustment = [f'cdi{idx}' for idx in incompetence_maladjustment]
    columns_negative_self_esteem = [f'cdi{idx}' for idx in negative_self_esteem]
    columns_sleep_appetite_disturbances = [f'cdi{idx}' for idx in sleep_appetite_disturbances]

    # Calculate total value for each subcategory
    data_t1['cdi_social_withdrawal'] = data_t1[columns_social_withdrawal].sum(axis=1)
    data_t1['cdi_anhedonia_asthenia'] = data_t1[columns_anhedonia_asthenia].sum(axis=1)
    data_t1['cdi_incompetence_maladjustment'] = data_t1[columns_incompetence_maladjustment].sum(axis=1)
    data_t1['cdi_negative_self_esteem'] = data_t1[columns_negative_self_esteem].sum(axis=1)
    data_t1['cdi_sleep_appetite_disturbances'] = data_t1[columns_sleep_appetite_disturbances].sum(axis=1)
    ################# END COMBINE TESTS ###########################

    ################# DROP REDUNDANT COLUMNS ######################
    data_t1.drop(data_t1.loc[:, 'group_type':'cdi27'].columns, axis=1, inplace=True)
    data_t1.drop('apoio', axis=1, inplace=True)

    data_t1.to_excel(output_file, index=False)
    print(f"Combined data saved to {output_file}")
    ################# END DROP REDUNDANT COLUMNS ###########################

###### END COMBINE THE QUESTIONNAIRES ######

#### PREPROCESS T1 DATA ####
def round_data(data, filename):
    data_rounded = data.select_dtypes(include=[np.number]).round().astype(int)
    
    # Combine the rounded numeric columns with the non-numeric columns
    data_rounded = pd.concat([data_rounded, data.select_dtypes(exclude=[np.number])], axis=1)
    
    # Ensure the column order is preserved
    data_rounded = data_rounded[data.columns]

    path = f'{filename}.xlsx'
    data_rounded.to_excel(path, index=False)
    print(f'round_data data saved to {path}')

    return data_rounded

def drop_cdi_columns_not_combined(data, filename):
    data.drop(columns=[f'cdi{i}' for i in range(1, 28)], inplace=True)
    path = f'{filename}.xlsx'
    data.to_excel(path, index=False)
    print(f'drop_cdi_columns_not_combined data saved to {path}')
    return data


def drop_cdi_columns_combined(data, filename):
    cdi_columns = ['cdi_social_withdrawal', 'cdi_anhedonia_asthenia', 'cdi_incompetence_maladjustment', 'cdi_negative_self_esteem', 'cdi_sleep_appetite_disturbances']
    data.drop(columns=cdi_columns, inplace=True)
    path = f'{filename}.xlsx'
    data.to_excel(path, index=False)
    print(f'drop_cdi_columns_combined data saved to {path}')
    return data


def encode_data(data, categorical_features, filename):
    data_encoded = pd.get_dummies(data, columns=categorical_features, drop_first=True)

    # Convert boolean columns to integers
    data_encoded = data_encoded.astype({col: int for col in data_encoded.select_dtypes(include=['bool']).columns})

    path = f'{filename}.xlsx'
    data_encoded.to_excel(path, index=False)
    print(f'encode_data data saved to {path}')
    return data_encoded 

# Standardise numerical features, one-hot encode categorical features, and recode binary feature
def standardise_encode_recode_features(df, filename, numerical_features, categorical_features, binary_features, leave_as_it_is_features):
    # Standardize numerical features
    scaler = StandardScaler()
    df_numerical = pd.DataFrame(scaler.fit_transform(df[numerical_features]), columns=numerical_features, index=df.index)

    # One-hot encode categorical features
    encoder = OneHotEncoder(drop='first', sparse_output=False)  # Use sparse_output instead of sparse
    df_categorical = pd.DataFrame(encoder.fit_transform(df[categorical_features]), 
                                  columns=encoder.get_feature_names_out(categorical_features),
                                  index=df.index)
    
    # Recode binary features to 0/1
    df_binary = df[binary_features].map(lambda x: 1 if x == 2 else 0)

    # Combine all the processed features together with the "leave as it is" features
    df_preprocessed = pd.concat([df_numerical, df_categorical, df_binary, df[leave_as_it_is_features]], axis=1)

    path = f'{filename}.xlsx'
    df_preprocessed.to_excel(path, index=False)
    print(f'standardise_encode_recode_features data saved to {filename}')
    return df_preprocessed


def create_dataset_for_test(df, test_score, filename):
    # list of all test scores
    test_scores = ['tdeescri', 'tdearitm', 'tdeleit', 'tdetotal']
    
    if test_score not in test_scores:
        raise ValueError(f"Invalid test_score. Must be one of {test_scores}")
    
    # remove the test score we want to keep from the list of all test scores
    test_scores_to_drop = [score for score in test_scores if score != test_score]
    
    # drop other tests and create a new dataset
    school_test_df = df.drop(columns=test_scores_to_drop)

    path = f'{filename}.xlsx'
    school_test_df.to_excel(path, index=False)
    print(f'school_test_df data saved to {path}')
    return school_test_df

#### END PREPROCESS T1 DATA ####