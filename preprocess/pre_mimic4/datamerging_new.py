from pathlib import Path
import numpy as np
import pandas as pd
from datetime import timedelta


p_project = Path(__file__).parents[2]
path_m4 = p_project/'data/mimic4'

lab_df = pd.read_csv(path_m4/'processed/tables/lab_processed.csv')[
    ['subject_id', 'hadm_id', 'charttime', 'valuenum', 'label']]
inputs_df = pd.read_csv(path_m4/'processed/tables/inputs_processed.csv')[
    ['subject_id', 'hadm_id', 'charttime', 'amount', 'label']]
outputs_df = pd.read_csv(path_m4/'processed/tables/outputs_processed.csv')[
    ['subject_id', 'hadm_id', 'charttime', 'value', 'label']]
presc_df = pd.read_csv(path_m4/'processed/tables/prescriptions_processed.csv')[
    ['subject_id', 'hadm_id', 'charttime', 'dose_val_rx', 'drug']]

# Change the name of amount. Valuenum for every table
inputs_df['valuenum'] = inputs_df['amount']
inputs_df = inputs_df.drop(columns=['amount']).copy()

outputs_df['valuenum'] = outputs_df['value']
outputs_df = outputs_df.drop(columns=['value']).copy()

presc_df['valuenum'] = presc_df['dose_val_rx']
presc_df = presc_df.drop(columns=['dose_val_rx']).copy()
presc_df['label'] = presc_df['drug']
presc_df = presc_df.drop(columns=['drug']).copy()

# Tag to distinguish between lab and inputs events
inputs_df['Origin'] = 'Inputs'
lab_df['Origin'] = 'Lab'
outputs_df['Origin'] = 'Outputs'
presc_df['Origin'] = 'Prescriptions'

merged_df = pd.concat((inputs_df, lab_df, outputs_df, presc_df)).reset_index()

# Check that all labels have different names.
assert(merged_df['label'].nunique() == (inputs_df['label'].nunique(
)+lab_df['label'].nunique()+outputs_df['label'].nunique()+presc_df['label'].nunique()))

# set the timestamp as the time delta between the first chart time for each admission
merged_df['charttime'] = pd.to_datetime(
    merged_df['charttime'], format='%Y-%m-%d %H:%M:%S')
ref_time = merged_df.groupby('hadm_id')['charttime'].min()
merged_df_1 = pd.merge(ref_time.to_frame(name='ref_time'),
                       merged_df, left_index=True, right_on='hadm_id')
merged_df_1['time_stamp'] = merged_df_1['charttime']-merged_df_1['ref_time']
assert(len(merged_df_1.loc[merged_df_1['time_stamp']
       < timedelta(hours=0)].index) == 0)

# Create a label code (int) for the labels.
label_dict = dict(zip(list(merged_df_1['label'].unique()), range(
    len(list(merged_df_1['label'].unique())))))
merged_df_1['label_code'] = merged_df_1['label'].map(label_dict)

label_dict_df = pd.Series(merged_df_1['label'].unique()).reset_index()
label_dict_df.columns = ['index', 'label']
label_dict_df['label_code'] = label_dict_df['label'].map(label_dict)
label_dict_df.drop(columns=['index'], inplace=True)
label_dict_df.to_csv(path_m4/'processed/tables/variable_name_dict.csv')

merged_df_short = merged_df_1[['hadm_id', 'valuenum', 'time_stamp', 'label_code']].rename(
    columns={'hadm_id': 'ID', 'time_stamp': 'Time'})

# select patients who have records in both the first 24 hours and the second 24 hours
ids_before_24 = merged_df_short.loc[(
    merged_df_short['Time'] < timedelta(hours=24))]['ID'].unique()

ids_after_24 = merged_df_short.loc[(merged_df_short['Time'] >= timedelta(hours=24)) &
                                   (merged_df_short['Time'] < timedelta(hours=48))]['ID'].unique()

merged_df_short = merged_df_short.loc[merged_df_short['ID'].isin(
    set(ids_before_24) & set(ids_after_24))]

# select 48h records
merged_df_short = merged_df_short.loc[merged_df_short['Time'] < timedelta(
    hours=48)]

# The sampling interval is 1 minute
merged_df_short['Time'] = merged_df_short['Time'].dt.total_seconds().div(
    60).astype(int)
assert(len(merged_df_short.loc[merged_df_short['Time'] > 2880].index) == 0)

value_df = pd.pivot_table(merged_df_short, values='valuenum', index=[
                          'ID', 'Time'], columns=['label_code'], aggfunc=np.max)
mask_df = value_df.notna()

d_values = {}
d_masks = {}
for i in value_df.columns:
    d_values[i] = "Value_" + str(i)
    d_masks[i] = "Mask_" + str(i)

value_df.rename(columns=d_values, inplace=True)
mask_df.rename(columns=d_masks, inplace=True)

value_df.fillna(0, inplace=True)
mask_df = mask_df.astype(int)

complete_df = pd.concat((value_df, mask_df), axis=1).reset_index()

complete_df['ID'] = complete_df['ID'].astype(int)

complete_df.sort_values(["ID", "Time"], inplace=True)

complete_df.to_csv(
    path_m4/'processed/mimic4_full_dataset.csv', index=False)
