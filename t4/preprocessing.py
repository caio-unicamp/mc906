import pandas as pd
import numpy as np
import os
import gc
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import pickle

DATA_DIR = 'data'
PROCESSED_DIR = 'processed'
os.makedirs(PROCESSED_DIR, exist_ok=True)

RANDOM_STATE = 42

def read_csv_safe(path, **kwargs):
    try:
        return pd.read_csv(path, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='latin-1', **kwargs)


def aggregate_supplementary_tables():
    if os.path.exists(f'{PROCESSED_DIR}/bureau_agg.pkl') and \
       os.path.exists(f'{PROCESSED_DIR}/prev_app_agg.pkl') and \
       os.path.exists(f'{PROCESSED_DIR}/pos_cash_agg.pkl') and \
       os.path.exists(f'{PROCESSED_DIR}/cc_balance_agg.pkl') and \
       os.path.exists(f'{PROCESSED_DIR}/installments_agg.pkl'):
        return

    print("Loading supplementary tables...")

    bureau = read_csv_safe(f'{DATA_DIR}/bureau.csv')
    bb = read_csv_safe(f'{DATA_DIR}/bureau_balance.csv')
    prev = read_csv_safe(f'{DATA_DIR}/previous_application.csv')
    pos = read_csv_safe(f'{DATA_DIR}/POS_CASH_balance.csv')
    cc = read_csv_safe(f'{DATA_DIR}/credit_card_balance.csv')
    ins = read_csv_safe(f'{DATA_DIR}/installments_payments.csv')

    num_aggs = {
        'AMT_CREDIT_SUM': ['mean', 'sum', 'min', 'max'],
        'AMT_CREDIT_SUM_DEBT': ['mean', 'sum', 'max'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean', 'max'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'AMT_ANNUITY': ['mean', 'max'],
        'DAYS_CREDIT': ['mean', 'min', 'max'],
        'CREDIT_DAY_OVERDUE': ['mean', 'max'],
        'DAYS_CREDIT_ENDDATE': ['mean', 'min', 'max'],
        'DAYS_ENDDATE_FACT': ['mean', 'min'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean', 'max'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'DAYS_CREDIT_UPDATE': ['mean', 'max'],
    }

    print("Aggregating bureau...")
    bureau_agg = bureau.groupby('SK_ID_CURR').agg(num_aggs)
    bureau_agg.columns = ['BUREAU_' + '_'.join(col) for col in bureau_agg.columns]
    bureau_agg['BUREAU_COUNT'] = bureau.groupby('SK_ID_CURR').size()

    cr_status = pd.get_dummies(bureau['CREDIT_ACTIVE'], prefix='BUREAU_ACTIVE')
    cr_status['SK_ID_CURR'] = bureau['SK_ID_CURR']
    cr_status = cr_status.groupby('SK_ID_CURR').mean()
    bureau_agg = bureau_agg.join(cr_status, how='left')

    cr_type = pd.get_dummies(bureau['CREDIT_TYPE'], prefix='BUREAU_TYPE')
    cr_type['SK_ID_CURR'] = bureau['SK_ID_CURR']
    cr_type = cr_type.groupby('SK_ID_CURR').mean()
    bureau_agg = bureau_agg.join(cr_type, how='left')

    bureau_agg.fillna(0, inplace=True)
    pickle.dump(bureau_agg, open(f'{PROCESSED_DIR}/bureau_agg.pkl', 'wb'))
    del bureau, cr_status, cr_type
    gc.collect()

    if bb is not None:
        print("Aggregating bureau_balance...")
        bb_agg = bb.groupby('SK_ID_BUREAU').agg({'MONTHS_BALANCE': ['size', 'min', 'max']})
        bb_agg.columns = ['BB_' + '_'.join(col) for col in bb_agg.columns]

        status_dummies = pd.get_dummies(bb['STATUS'], prefix='BB_STATUS')
        status_dummies['SK_ID_BUREAU'] = bb['SK_ID_BUREAU']
        status_dummies = status_dummies.groupby('SK_ID_BUREAU').mean()
        bb_agg = bb_agg.join(status_dummies, how='left')

        bureau_agg = bureau_agg.merge(bb_agg, left_index=True, right_index=True, how='left')
        bureau_agg.fillna(0, inplace=True)
        pickle.dump(bureau_agg, open(f'{PROCESSED_DIR}/bureau_agg.pkl', 'wb'))
        del bb, bb_agg, status_dummies
        gc.collect()

    print("Aggregating previous_application...")
    prev_num_aggs = {
        'AMT_ANNUITY': ['mean', 'min', 'max'],
        'AMT_APPLICATION': ['mean', 'min', 'max'],
        'AMT_CREDIT': ['mean', 'min', 'max'],
        'AMT_DOWN_PAYMENT': ['mean', 'min', 'max'],
        'AMT_GOODS_PRICE': ['mean', 'min', 'max'],
        'HOUR_APPR_PROCESS_START': ['mean', 'min', 'max'],
        'RATE_DOWN_PAYMENT': ['mean', 'max'],
        'RATE_INTEREST_PRIMARY': ['mean', 'max'],
        'RATE_INTEREST_PRIVILEGED': ['mean', 'max'],
        'DAYS_DECISION': ['mean', 'min', 'max'],
        'CNT_PAYMENT': ['mean', 'sum'],
        'DAYS_FIRST_DRAWING': ['mean', 'min'],
        'DAYS_FIRST_DUE': ['mean', 'min'],
        'DAYS_LAST_DUE_1ST_VERSION': ['mean', 'min', 'max'],
        'DAYS_LAST_DUE': ['mean', 'min'],
        'DAYS_TERMINATION': ['mean', 'min'],
        'NFLAG_INSURED_ON_APPROVAL': ['mean'],
        'SELLERPLACE_AREA': ['mean', 'max'],
        'NFLAG_LAST_APPL_IN_DAY': ['mean'],
    }

    prev_agg = prev.groupby('SK_ID_CURR').agg(prev_num_aggs)
    prev_agg.columns = ['PREV_' + '_'.join(col) for col in prev_agg.columns]
    prev_agg['PREV_COUNT'] = prev.groupby('SK_ID_CURR').size()

    prev_approved = prev[prev['NAME_CONTRACT_STATUS'] == 'Approved']
    prev_agg['PREV_APPROVED_COUNT'] = prev_approved.groupby('SK_ID_CURR').size()
    prev_agg['PREV_APPROVED_RATIO'] = prev_agg['PREV_APPROVED_COUNT'] / prev_agg['PREV_COUNT']

    prev_refused = prev[prev['NAME_CONTRACT_STATUS'] == 'Refused']
    prev_agg['PREV_REFUSED_RATIO'] = prev_refused.groupby('SK_ID_CURR').size() / prev_agg['PREV_COUNT']

    prev_agg.fillna(0, inplace=True)
    pickle.dump(prev_agg, open(f'{PROCESSED_DIR}/prev_app_agg.pkl', 'wb'))
    del prev, prev_approved, prev_refused
    gc.collect()

    print("Aggregating POS_CASH_balance...")
    pos_aggs = {
        'MONTHS_BALANCE': ['size', 'min', 'max'],
        'CNT_INSTALMENT': ['mean', 'min', 'max'],
        'CNT_INSTALMENT_FUTURE': ['mean', 'min', 'max', 'sum'],
        'SK_DPD': ['mean', 'max', 'sum'],
        'SK_DPD_DEF': ['mean', 'max', 'sum'],
    }
    pos_agg = pos.groupby('SK_ID_CURR').agg(pos_aggs)
    pos_agg.columns = ['POS_' + '_'.join(col) for col in pos_agg.columns]

    pos_agg['POS_COMPLETED_RATIO'] = pos[pos['NAME_CONTRACT_STATUS'] == 'Completed'].groupby('SK_ID_CURR').size() / pos_agg['POS_MONTHS_BALANCE_size']
    pos_agg['POS_ACTIVE_RATIO'] = pos[pos['NAME_CONTRACT_STATUS'] == 'Active'].groupby('SK_ID_CURR').size() / pos_agg['POS_MONTHS_BALANCE_size']

    pos_agg.fillna(0, inplace=True)
    pickle.dump(pos_agg, open(f'{PROCESSED_DIR}/pos_cash_agg.pkl', 'wb'))
    del pos
    gc.collect()

    print("Aggregating credit_card_balance...")
    cc_num_aggs = {
        'MONTHS_BALANCE': ['size', 'min'],
        'AMT_BALANCE': ['mean', 'max'],
        'AMT_CREDIT_LIMIT_ACTUAL': ['mean', 'max'],
        'AMT_DRAWINGS_ATM_CURRENT': ['mean', 'max', 'sum'],
        'AMT_DRAWINGS_CURRENT': ['mean', 'max', 'sum'],
        'AMT_DRAWINGS_OTHER_CURRENT': ['mean', 'max', 'sum'],
        'AMT_DRAWINGS_POS_CURRENT': ['mean', 'max', 'sum'],
        'AMT_INST_MIN_REGULARITY': ['mean', 'max'],
        'AMT_PAYMENT_CURRENT': ['mean', 'max', 'sum'],
        'AMT_PAYMENT_TOTAL_CURRENT': ['mean', 'max', 'sum'],
        'AMT_RECEIVABLE_PRINCIPAL': ['mean', 'max'],
        'AMT_RECIVABLE': ['mean', 'max'],
        'AMT_TOTAL_RECEIVABLE': ['mean', 'max'],
        'CNT_DRAWINGS_ATM_CURRENT': ['mean', 'max', 'sum'],
        'CNT_DRAWINGS_CURRENT': ['mean', 'max', 'sum'],
        'CNT_DRAWINGS_OTHER_CURRENT': ['mean', 'max', 'sum'],
        'CNT_DRAWINGS_POS_CURRENT': ['mean', 'max', 'sum'],
        'CNT_INSTALMENT_MATURE_CUM': ['mean', 'max', 'sum'],
        'SK_DPD': ['mean', 'max', 'sum'],
        'SK_DPD_DEF': ['mean', 'max', 'sum'],
    }

    cc_agg = cc.groupby('SK_ID_CURR').agg(cc_num_aggs)
    cc_agg.columns = ['CC_' + '_'.join(col) for col in cc_agg.columns]

    cc_agg['CC_UTILIZATION'] = cc_agg['CC_AMT_BALANCE_mean'] / (cc_agg['CC_AMT_CREDIT_LIMIT_ACTUAL_mean'] + 1)
    cc_agg['CC_PAYMENT_RATIO'] = cc_agg['CC_AMT_PAYMENT_CURRENT_sum'] / (cc_agg['CC_AMT_BALANCE_mean'] + 1)

    cc_agg.fillna(0, inplace=True)
    cc_agg.replace([np.inf, -np.inf], 0, inplace=True)
    pickle.dump(cc_agg, open(f'{PROCESSED_DIR}/cc_balance_agg.pkl', 'wb'))
    del cc
    gc.collect()

    print("Aggregating installments_payments...")
    ins_num_aggs = {
        'NUM_INSTALMENT_VERSION': ['mean', 'sum'],
        'NUM_INSTALMENT_NUMBER': ['mean', 'min', 'max'],
        'DAYS_INSTALMENT': ['mean', 'min', 'max'],
        'DAYS_ENTRY_PAYMENT': ['mean', 'min', 'max'],
        'AMT_INSTALMENT': ['mean', 'min', 'max', 'sum'],
        'AMT_PAYMENT': ['mean', 'min', 'max', 'sum'],
    }
    ins_agg = ins.groupby('SK_ID_CURR').agg(ins_num_aggs)
    ins_agg.columns = ['INS_' + '_'.join(col) for col in ins_agg.columns]

    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    ins_agg['INS_PAYMENT_DIFF_MEAN'] = ins.groupby('SK_ID_CURR')['PAYMENT_DIFF'].mean()
    ins_agg['INS_PAYMENT_PCT'] = ins.groupby('SK_ID_CURR')['AMT_PAYMENT'].sum() / (ins.groupby('SK_ID_CURR')['AMT_INSTALMENT'].sum() + 1)

    ins['DAYS_DIFF'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins_agg['INS_DAYS_LATE_MEAN'] = ins.groupby('SK_ID_CURR')['DAYS_DIFF'].mean()
    ins_agg['INS_LATE_RATIO'] = ins[ins['DAYS_DIFF'] > 0].groupby('SK_ID_CURR').size() / ins.groupby('SK_ID_CURR').size()

    ins_agg.fillna(0, inplace=True)
    ins_agg.replace([np.inf, -np.inf], 0, inplace=True)
    pickle.dump(ins_agg, open(f'{PROCESSED_DIR}/installments_agg.pkl', 'wb'))
    del ins
    gc.collect()

    print("Supplementary table aggregation complete.")


def preprocess_main_table():
    train_path = f'{PROCESSED_DIR}/train_processed.pkl'
    test_path = f'{PROCESSED_DIR}/test_processed.pkl'

    if os.path.exists(train_path) and os.path.exists(test_path):
        print("Loading preprocessed main tables...")
        train = pickle.load(open(train_path, 'rb'))
        test = pickle.load(open(test_path, 'rb'))
        return train, test

    print("Loading main tables and supplementary features...")
    train = read_csv_safe(f'{DATA_DIR}/application_train.csv')
    test = read_csv_safe(f'{DATA_DIR}/application_test.csv')
    sample_sub = read_csv_safe(f'{DATA_DIR}/sample_submission.csv')

    if not os.path.exists(f'{PROCESSED_DIR}/bureau_agg.pkl'):
        aggregate_supplementary_tables()

    print("Merging supplementary features...")

    for name, file in [
        ('bureau', 'bureau_agg.pkl'),
        ('prev', 'prev_app_agg.pkl'),
        ('pos', 'pos_cash_agg.pkl'),
        ('cc', 'cc_balance_agg.pkl'),
        ('ins', 'installments_agg.pkl'),
    ]:
        if os.path.exists(f'{PROCESSED_DIR}/{file}'):
            df = pickle.load(open(f'{PROCESSED_DIR}/{file}', 'rb'))
            train = train.merge(df, left_on='SK_ID_CURR', right_index=True, how='left')
            test = test.merge(df, left_on='SK_ID_CURR', right_index=True, how='left')
            del df
            gc.collect()
        else:
            print(f"Warning: {file} not found, skipping.")

    print(f"After merging: train shape {train.shape}, test shape {test.shape}")

    y = train['TARGET'].copy()
    train.drop(columns=['TARGET'], inplace=True)
    train_ids = train['SK_ID_CURR']
    test_ids = test['SK_ID_CURR']

    print("Feature processing...")

    categorical_cols = train.select_dtypes(include=['object']).columns.tolist()
    print(f"Categorical columns: {len(categorical_cols)}")

    for col in categorical_cols:
        le = LabelEncoder()
        combined = pd.concat([train[col].astype(str), test[col].astype(str)], axis=0)
        le.fit(combined)
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))

    train.replace([np.inf, -np.inf], np.nan, inplace=True)
    test.replace([np.inf, -np.inf], np.nan, inplace=True)

    imputer = SimpleImputer(strategy='median')
    train_imputed = imputer.fit_transform(train.drop(columns=['SK_ID_CURR']))
    test_imputed = imputer.transform(test.drop(columns=['SK_ID_CURR']))

    feature_names = train.drop(columns=['SK_ID_CURR']).columns.tolist()
    train = pd.DataFrame(train_imputed, columns=feature_names)
    test = pd.DataFrame(test_imputed, columns=feature_names)
    train['SK_ID_CURR'] = train_ids.values
    test['SK_ID_CURR'] = test_ids.values
    train['TARGET'] = y.values

    pickle.dump(train, open(train_path, 'wb'))
    pickle.dump(test, open(test_path, 'wb'))
    pickle.dump(sample_sub, open(f'{PROCESSED_DIR}/sample_submission.pkl', 'wb'))

    print(f"Final train shape: {train.shape}, test shape: {test.shape}")
    return train, test


if __name__ == '__main__':
    train, test = preprocess_main_table()
    print("Preprocessing complete.")
