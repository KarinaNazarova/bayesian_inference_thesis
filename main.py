from sparse_regression.scripts.missingness import impute_missing_values_and_save_df
from sparse_regression.scripts.copulas_synthetic_data import generate_synthetic_data

from sparse_regression.experiments.A_run_experiments import run_experiment_A
from sparse_regression.experiments.B_run_experiments import run_experiment_B
from sparse_regression.experiments.C_2Multi_run_experiments import run_experiment_C
from sparse_regression.experiments.D_run_experiments import run_experiment_D
from sparse_regression.experiments.E_run_experiments import run_experiment_E
from sparse_regression.experiments.F_run_experiments import run_experiment_F


from sparse_regression.scripts.data_preprocess import (
    create_dataset_for_test, round_data, drop_cdi_columns_not_combined, 
    drop_cdi_columns_combined, encode_data, prepare_t1_data, combine_questionnaires, standardise_encode_recode_features
)

import pandas as pd
import numpy as np

# ##### LOAD AND CLEAN DATA (data_clean.py) ######
# prepare_t1_data('sparse_regression/data/raw/data.xlsx', 'data_cleaned_T1.xlsx')

# ###### COMBINE DATA FOR T1 (data_combine.py) ######
# # This combines all questionnaires based on their subcategories 
# # combine_questionnaires('sparse_regression/data/processed/data_cleaned_T1.xlsx', 'sparse_regression/data/processed/data_combined_T1.xlsx')

# ###### GENERATE REAL DATASETS WITH IMPUTED/ENCODED/COMBINED/ROUNDED DATA ######
# # 1. OG imputed data - impute missing columns in the original data
# data_t1_not_combined = pd.read_excel('sparse_regression/data/processed/data_cleaned_T1.xlsx', index_col=None, header=0)  
# print('Loaded the Dataset (T1)')
# print('')
# impute_missing_values_and_save_df(data_t1_not_combined, 'sparse_regression/data/processed/imputed_data_not_combined_T1')
 
# # # 2. Imputed data combined - NO NEED probably
# # # imputes missing values in the combined data
# # data_t1_combined = pd.read_excel('sparse_regression/data/processed/data_combined_T1.xlsx', index_col=None, header=0)  
# # print('Loaded the Dataset (C)')
# # print('')

# # impute_missing_values_and_save_df(data_t1_combined, 'sparse_regression/data/processed/imputed_data_combined_T1')

# # 3. nocdi_imputed_data_not_combined_T1 - no combined questionnaires, no encoding, imputation (all values but cdi)
# # drop columns that are from cdi questionnaires with drop_cdi_columns_not_combined
# imputed_data_not_combined_T1 = pd.read_excel('sparse_regression/data/processed/imputed_data_not_combined_T1.xlsx', index_col=None, header=0)
# nocdi_imputed_data_not_combined_T1 = drop_cdi_columns_not_combined(imputed_data_not_combined_T1.copy(), 'sparse_regression/data/processed/nocdi_imputed_data_not_combined_T1')

# # # 4. nocdi_imputed_data_combined_T1 - combined questionnaires, no encoding, imputation (no cdi)  - NO NEED probably
# # # drop groups of columns that are from cdi questionnaires with drop_cdi_columns_combined
# # imputed_data_combined_T1 = pd.read_excel('sparse_regression/data/processed/imputed_data_combined_T1.xlsx', index_col=None, header=0)
# # nocdi_imputed_data_combined_T1 = drop_cdi_columns_combined(imputed_data_combined_T1.copy(), 'sparse_regression/data/processed/nocdi_imputed_data_combined_T1')

# # # 5. encoded_imputed_data_not_combined_T1 - no combined questionnaires, encoding, imputation  
# # # encode imputed_data_not_combined_T1 with encode_data
# # numeric_features = [
# #     'tdeescri', 'tdearitm', 'tdeleit', 'tdetotal', 'age', 'time_in_institution', 'number_siblings', 'people_in_house', 'earnings' 
# # ]
# # categorical_features = [col for col in imputed_data_not_combined_T1.columns if col not in numeric_features]
# # encoded_imputed_data_not_combined_T1 = encode_data(imputed_data_not_combined_T1.copy(), categorical_features, 'sparse_regression/data/processed/encoded_imputed_data_not_combined_T1')

# # # 6. rounded_imputed_data_combined_T1
# # imputed_data_combined_T1 = pd.read_excel('sparse_regression/data/processed/imputed_data_combined_T1.xlsx', index_col=None, header=0)
# # round_data(imputed_data_combined_T1, 'sparse_regression/data/processed/rounded_imputed_data_combined_T1')

# # 7. rounded_imputed_data_not_combined_T1
# imputed_data_not_combined_T1 = pd.read_excel('sparse_regression/data/processed/imputed_data_not_combined_T1.xlsx', index_col=None, header=0)
# round_data(imputed_data_not_combined_T1, 'sparse_regression/data/processed/rounded_imputed_data_not_combined_T1')

# # 8. one hot encode categorical features, standardise numeric features, recode binary features
# # standardise all numeric features 
# tde_features = ['tdeescri', 'tdearitm', 'tdeleit', 'tdetotal']
# numeric_features = ['age', 'time_in_institution', 'grade', 'number_siblings', 'people_in_house', 'earnings']
# numeric_features = numeric_features + tde_features

# # categorical features to do one hot encoding
# categorical_features_ohe = ['group_type', 'institution_type', 'institution_name', 'school_name', 'father_occupation', 'mother_occupation', 'apoio', 
#                             'family_contact_frequency', 'job_type']

# # recode binary features to 0/1
# binary_features = ['sex', 'knows_dob', 'goes_to_school', 'type_of_school', 'repeated_grade', 'abandon_school', 'expelled_from_school', 'has_children', 'got_pregnant',
#                    'has_siblings_in_institution', 'has_contact_w_family', 'has_job' ]

# # features to leave as they are
# emsv_features = [f'emsv{i}' for i in range(1, 51)]
# ea_features = [f'ea{i}' for i in range(1, 41)]
# cdi_features = [f'cdi{i}' for i in range(1, 28)]
# eev_product_features = [f'eev{i}_product' for i in range(1, 61)]
# ordinal_categorical_features = ['father_education', 'mother_education', 'times_repeated_grade']
# unsure_features = ['parents_live_together', 'job_satisfaction', 'skip_school_for_work']
# label = ['cdi_label']
# leave_as_it_is_features = ordinal_categorical_features + emsv_features + ea_features + cdi_features + eev_product_features + unsure_features + label
# #####

# leave_as_it_is_features = ordinal_categorical_features + emsv_features + ea_features + cdi_features + eev_product_features + unsure_features

# # Standardise numeric features from all questionnaires 
# numeric_features += leave_as_it_is_features
# leave_as_it_is_features = label

# #####
# rounded_imputed_data_not_combined_T1 = pd.read_excel('sparse_regression/data/processed/rounded_imputed_data_not_combined_T1.xlsx', index_col=None, header=0)
# standardise_encode_recode_features(rounded_imputed_data_not_combined_T1, 'sparse_regression/data/processed/standardise_encode_recode_rounded_data_T1', numeric_features, categorical_features_ohe, binary_features, leave_as_it_is_features)

# # 9. Remove CDI features from the preprocessed standardise_encode_recode_rounded_data_T1 data
# data = pd.read_excel('sparse_regression/data/processed/standardise_encode_recode_rounded_data_T1.xlsx', index_col=None, header=0)
# drop_cdi_columns_not_combined(data, 'sparse_regression/data/processed/nocdi_standardise_encode_recode_rounded_data_T1')


# # 10. For each school test, only keep that test data, removing other tests
# data = pd.read_excel('sparse_regression/data/processed/standardise_encode_recode_rounded_data_T1.xlsx', index_col=None, header=0)
# create_dataset_for_test(data, 'tdeescri', 'sparse_regression/data/processed/tdeescri_only_standardise_encode_recode_rounded_data_T1')


# # SYNTHETIC DATA GENERATION WITH COPULA (copulas_synthetic_data.py)
# generate_synthetic_data('sparse_regression/data/processed/rounded_imputed_data_combined_T1.xlsx', 'sparse_regression/data/processed/synthetic_data_copula.xlsx')

#### RUN EXPERIMENTS ######
# # Run experiment A 
# run_experiment_A() 

# # Run experiment B 
# run_experiment_B()

# # Run experiment C
# run_experiment_C()

# Run experiment D - tdeleit
# run_experiment_D()

# Run experiment E - tdearitm
# run_experiment_E()

# Run experiment F - tdeescri
run_experiment_F()