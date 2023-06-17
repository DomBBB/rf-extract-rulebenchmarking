from data_generator import generate_imbalanced_data, make_noisy, generate_uniform_data
from os.path import exists
import os
from rule_generator import generate_interval_rule_set, generate_linear_rule_set, generate_conjunctive_rule, generate_conjunctive_rule2

CONFIG_DICT = {

    # Core Suite: Natural Business Prediction Problems

    'german_credit': {
        'DATA_SET': '../data/core_suite/german_credit_coded.csv',
        'DATA_TYPES': {'Duration in Month': float, 'Credit Amount': float, 'Installmentrate': float,
                       'PresentResidence': float, 'Age in years': float, 'Number existing Credits': float,
                       'Number people liable': float},
        'DROP': ['Index'],
        'TARGET_LABEL': 'Target',
        'TYPE': 'BINARY',
        'POS_CLASS': 0,
        'META_DATA': {'use_case': "credit",
                      'origin': 'organic',
                      'primary_source': 'https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)',
                      'actual_source': 'https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)',
                      'changes': 'inserted feature names according to the UCI documentation; LabelEncoded "Target" from categorical to 0 and 1 '}
    },

    'taiwan_credit': {
        'DATA_SET': '../data/core_suite/TaiwanCreditData.csv',
        'DATA_TYPES': {'Amount': float, 'Age': float, 'Bill_Sep': float, 'Bill_Aug': float, 'Bill_Jul': float,
                       'Bill_Jun': float, 'Bill_May': float, 'Bill_Apr': float, 'PayAmount_Sep': float,
                       'PayAmount_Aug': float, 'PayAmount_Jul': float, 'PayAmount_Jun': float, 'PayAmount_May': float,
                       'PayAmount_Apr': float},
        'DROP': ["Probabilities"],
        'TARGET_LABEL': 'DefaultNextMonth',
        'TYPE': 'BINARY',
        'POS_CLASS': 1,
        'META_DATA': {'use_case': "credit",
                      'origin': 'organic',
                      'primary_source': 'https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients',
                      'actual_source': 'https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset',
                      'changes': 'changed feature names Pay_1,... to the according month for readability'}
    },

    'compas': {
        'DATA_SET': '../data/core_suite/compas.csv',
        'DATA_TYPES': {},
        'DROP': ["Unnamed: 0"],
        'TARGET_LABEL': 'recidivate-within-two-years',
        'TYPE': 'BINARY',
        'POS_CLASS': 1,  # NOTE was originally 0
        'META_DATA': {'use_case': "recidivism",
                      'origin': 'organic',
                      'primary_source': 'https://www.propublica.org/datastore/dataset/compas-recidivism-risk-score-data-and-analysis',
                      'actual_source': 'https://github.com/corels/corels/tree/master/data',
                      'changes': 'merged train and test data'}
    },

    'fraud_detection': {
        'DATA_SET': '../data/core_suite/fraud_detection.csv',
        'DATA_TYPES': {'Amount': float, 'Age': float, 'Bill_Sep': float, 'Bill_Aug': float, 'Bill_Jul': float,
                       'Bill_Jun': float, 'Bill_May': float, 'Bill_Apr': float, 'PayAmount_Sep': float,
                       'PayAmount_Aug': float, 'PayAmount_Jul': float, 'PayAmount_Jun': float, 'PayAmount_May': float,
                       'PayAmount_Apr': float},
        'DROP': [],
        'TARGET_LABEL': 'Class',
        'TYPE': 'BINARY',
        'POS_CLASS': 1,
        'META_DATA': {'use_case': 'transaction-fraud',
                      'origin': 'organic',
                      'actual_source': 'https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud',
                      'notes': 'The original data set is PCA transformed for privacy.',
                      'changes': 'none'}
    },

    'fraud_oracle': {
        'NAME': 'fraud_oracle',
        'DATA_SET': '../data/core_suite/fraud_oracle_clean.csv',
        'DATA_TYPES': {'WeekOfMonth': float, 'WeekOfMonthClaimed': float, 'Age': float, 'PolicyNumber': float,
                       'Age': float, 'RepNumber': float, 'Deductible': float, 'DriverRating': float, 'Year': float},
        'DROP': ["Unnamed: 0"],
        'TARGET_LABEL': 'FraudFound_P',
        'TYPE': 'BINARY',
        'POS_CLASS': 1,
        'META_DATA': {'use_case': 'insurance-fraud',
                      'origin': 'organic',
                      'actual_source': 'https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection',
                      'changes': 'replaced "Age" entries with Value = 0 with the median '}
    },

    'bike_75': {
        'DATA_SET': '../data/core_suite/bike.csv',
        'DATA_TYPES': {'Rented Bike Count': float, 'Hour': float, 'Humidity_percent': float, 'Visibility_10m': float,
                       'target_75': str, 'target_mean': str},
        'DROP': ['Rented Bike Count', 'target_mean'],
        'TARGET_LABEL': 'target_75',
        'TYPE': 'BINARY',
        'POS_CLASS': 'True',
        'META_DATA': {'use_case': "demand-prediction",
                      'origin': 'organic',
                      'primary_source': 'https://archive.ics.uci.edu/ml/datasets/Seoul+Bike+Sharing+Demand',
                      'actual_source': '',
                      'changes': 'Introduced binary target as Rented Bike Count >= 75% of max of Rented Bike Count'}
    },

    'bike_mean': {
        'DATA_SET': '../data/core_suite/bike.csv',
        'DATA_TYPES': {'Rented Bike Count': float, 'Hour': float, 'Humidity_percent': float, 'Visibility_10m': float,
                       'target_75': str, 'target_mean': str},
        'DROP': ['Rented Bike Count', 'target_75'],
        'TARGET_LABEL': 'target_mean',
        'TYPE': 'BINARY',
        'POS_CLASS': 'True',
        'META_DATA': {'use_case': "demand-prediction",
                      'origin': 'organic',
                      'primary_source': 'https://archive.ics.uci.edu/ml/datasets/Seoul+Bike+Sharing+Demand',
                      'actual_source': '',
                      'changes': 'Introduced binary target as Rented Bike Count >= mean of Rented Bike Count'}
    },

    'orange_up': {
        'DATA_SET': '../data/core_suite/orange.csv',
        'DATA_TYPES': {'Var73': float, 'churn': str, 'appetency': str, 'upselling': str},
        'DROP': ['appetency', 'churn'],
        'TARGET_LABEL': 'upselling',
        'TYPE': 'BINARY',
        'POS_CLASS': '1',
        'META_DATA': {'use_case': "product recommendation",
                      'origin': 'organic',
                      'primary_source': 'https://kdd.org/kdd-cup/view/kdd-cup-2009/Data',
                      'actual_source': 'https://kdd.org/kdd-cup/view/kdd-cup-2009/Data',
                      'changes': 'Used only training data (b/c) only there labels available; joined with label files'}
    },

    # 'orange_app': {
    #     'DATA_SET': '../data/core_suite/orange.csv',
    #     'DATA_TYPES': {'Var73': float, 'churn': str, 'appetency': str, 'upselling': str},
    #     'DROP': ['upselling', 'churn'],
    #     'TARGET_LABEL': 'appetency',
    #     'TYPE': 'BINARY',
    #     'POS_CLASS': '1',
    #     'META_DATA': {'use_case': "product recommendation",
    #                   'origin': 'organic',
    #                   'primary_source': 'https://kdd.org/kdd-cup/view/kdd-cup-2009/Data',
    #                   'actual_source': 'https://kdd.org/kdd-cup/view/kdd-cup-2009/Data',
    #                   'changes': 'Used only training data (b/c) only there labels available; joined with label files'}
    # },

    'orange_churn': {
        'DATA_SET': '../data/core_suite/orange.csv',
        'DATA_TYPES': {'Var73': float, 'churn': str, 'appetency': str, 'upselling': str},
        'DROP': ['appetency', 'upselling'],
        'TARGET_LABEL': 'churn',
        'TYPE': 'BINARY',
        'POS_CLASS': '1',
        'META_DATA': {'use_case': "churn",
                      'origin': 'organic',
                      'primary_source': 'https://kdd.org/kdd-cup/view/kdd-cup-2009/Data',
                      'actual_source': 'https://kdd.org/kdd-cup/view/kdd-cup-2009/Data',
                      'changes': 'Used only training data (b/c) only there labels available; joined with label files'}
    },

    'adult': {
        'NAME': 'adult',
        'DATA_SET': '../data/core_suite/adult.csv',
        'DATA_TYPES': {"Age": float, "Capital_loss": float, "Hours_per_week": float, "Capital_gains": float},
        'DROP': ["Unnamed: 0"],
        'TYPE': 'BINARY',
        'TARGET_LABEL': 'class',
        'POS_CLASS': ' T',
        'META_DATA': {'use_case': "income_prediction",
                      'origin': 'organic',
                      'primary_source': 'https://archive.ics.uci.edu/ml/datasets/adult',
                      'actual_source': '',
                      'changes': 'none'}
    },

    'house': {
        'NAME': 'house',
        'DATA_SET': '../data/core_suite/house.csv',
        'DATA_TYPES': {"P1": float},
        'DROP': [],
        'TYPE': 'BINARY',
        'TARGET_LABEL': 'binaryClass',
        'POS_CLASS': "N",
        'META_DATA': {'use_case': "price_prediction",
                      'origin': 'organic',
                      'primary_source': 'https://www.cs.toronto.edu/~delve/data/census-house/desc.html',
                      'aux_source': 'https://www.openml.org/search?type=data&sort=runs&id=821&status=active',
                      'actual_source': 'https://github.com/Joeyonng/decision-rules-network/tree/master/datasets',
                      'notes': 'Mean was used as classification threshold',
                      'changes': 'none'}
    },

    'heloc': {
        'DATA_SET': '../data/core_suite/heloc.csv',
        'DATA_TYPES': {'ExternalRiskEstimate': float, 'MSinceOldestTradeOpen': float,
                       'MSinceMostRecentTradeOpen': float, 'AverageMInFile': float, 'NumSatisfactoryTrades': float,
                       'NumTrades60Ever2DerogPubRec': float, 'NumTrades90Ever2DerogPubRec': float,
                       'PercentTradesNeverDelq': float, 'MSinceMostRecentDelq': float,
                       'MaxDelq2PublicRecLast12M': float, 'MaxDelqEver': float, 'NumTotalTrades': float,
                       'NumTradesOpeninLast12M': float, 'PercentInstallTrades': float,
                       'MSinceMostRecentInqexcl7days': float, 'NumInqLast6M': float, 'NumInqLast6Mexcl7days': float,
                       'NetFractionRevolvingBurden': float, 'NetFractionInstallBurden': float,
                       'NumRevolvingTradesWBalance': float, 'NumInstallTradesWBalance': float,
                       'NumBank2NatlTradesWHighUtilization': float, 'PercentTradesWBalance': float},
        'DROP': ['Probabilities'],
        'TYPE': 'BINARY',
        'TARGET_LABEL': 'RiskPerformance',
        'POS_CLASS': "Good",
        'META_DATA': {'use_case': 'credit',
                      'origin': 'organic',
                      'primary_source': '',
                      'aux_source': 'https://www.openml.org/search?type=data&status=active&id=45023',
                      'actual_source': 'https://github.com/Joeyonng/decision-rules-network/tree/master/datasets',
                      'changes': 'none'}
    },

    'telco_churn': {
        'DATA_SET': '../data/core_suite/churn_telecom.csv',
        'DATA_TYPES': {'Account length': float, 'Area code': float, 'Number vmail messages': float, 'Total day calls': float, 'Total eve calls': float, 'Total night calls': float, 'Total intl calls': float, 'Customer service calls': float},
        'DROP': [],
        'TARGET_LABEL': 'Churn',
        'TYPE': 'BINARY',
        'POS_CLASS': 1,
        'META_DATA': {'use_case': "churn",
                      'origin': "organic",
                      'primary_source': '',
                      'source': 'https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets/discussion/235070',
                      'changes': ''
                      }
    },

    "electricity": {
        'DATA_SET': '../data/core_suite/electricity-normalized.csv',
        'DATA_TYPES': {'day': float},
        'DROP': [],
        'TYPE': 'BINARY',
        'TARGET_LABEL': 'class',
        'POS_CLASS': 'UP',
        'META_DATA': {'use_case': "price_sensitivity",
                      'origin': "organic",
                      'source': 'https://www.kaggle.com/datasets/yashsharan/the-elec2-dataset',
                      'changes': ''}
    },

    'bank_marketing': {
        'DATA_SET': '../data/core_suite/bank-additional-x.csv',
        'DATA_TYPES': {'age': float, 'duration': float, 'campaign': float, 'pdays': float, 'previous': float},
        'DROP': [],
        'TYPE': 'BINARY',
        'TARGET_LABEL': 'y',
        'POS_CLASS': "yes",
        'META_DATA': {'use_case': "product_recommendation",
                      'origin': "organic",
                      'primary_source': 'https://archive.ics.uci.edu/ml/datasets/bank+marketing',
                      'other_source': 'https://www.openml.org/search?type=data&sort=runs&id=1461&status=active',
                      'changes': 'row 3516 was removed because it was the only row where default==yes. \
                      This creates a problem for the quantile binarizer'}
    },

    'telco_churn_2': {
        'DATA_SET': '../data/core_suite/SouthAsianWirelessTelecomOperator(SATO2015).csv',
        'DATA_TYPES': {'network_age': float, 'Aggregate_Calls': float, 'Aggregate_ONNET_REV': float, 'Aggregate_OFFNET_REV': float, 'Aggregate_complaint_count': float,
                       'aug_user_type': str, 'sep_user_type': str, 'aug_fav_a': str, 'sep_fav_a': str},
        'DROP': [],
        'TARGET_LABEL': 'Class',
        'TYPE': 'BINARY',
        'POS_CLASS': "Churned",
        'META_DATA': {'use_case': "churn",
                      'origin': "organic",
                      'actual_source': 'https://www.kaggle.com/datasets/mahreen/sato2015',
                      'changes': ''}
    },

    "boston": {
        'DATA_SET': '../data/core_suite/boston.csv',
        'DATA_TYPES': {'CHAS': float, 'RAD': float, 'TAX': float},
        'DROP': ['idx', 'MEDV'],  # MEDV is original numeric target
        'TARGET_LABEL': 'Target',
        'TYPE': 'BINARY',
        'POS_CLASS': 1,
        'META_DATA': {'use_case': "price_prediction",
                      'origin': "organic",
                      'primary_source': 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/',
                      'actual_source': 'https://www.kaggle.com/code/shreayan98c/boston-house-price-prediction',
                      'changes': 'target mean used as threshold for binary prediction'}
    },

    # Other: Suite of natural data sets
    #
    # 'mushroom': {
    #     'DATA_SET': '../data/other/mushroom.csv',
    #     'DATA_TYPES': {},
    #     'DROP': [],
    #     'TYPE': 'BINARY',
    #     'TARGET_LABEL': 'Poisonous/Edible',
    #     'POS_CLASS': 'e',
    #     'META_DATA': {'use_case': 'other',
    #                   'origin': 'organic',
    #                   'source': 'https://archive.ics.uci.edu/ml/datasets/mushroom',
    #                   'changes': 'none'}
    # },
    #
    # 'mushroom_inv': {
    #     'DATA_SET': '../data/other/mushroom.csv',
    #     'DATA_TYPES': {},
    #     'DROP': [],
    #     'TYPE': 'BINARY',
    #     'TARGET_LABEL': 'Poisonous/Edible',
    #     'POS_CLASS': 'p',
    #     'META_DATA': {'use_case': 'other', 'origin': 'organic'}
    # },
    #
    # 'bike_imbalanced': {
    #     'DATA_SET': '../data/adjusted/imbalance/binary_bike_imbalanced.csv',
    #     'DATA_TYPES': {'Rented Bike Count': float, 'Hour': float, 'Humidity_percent': float, 'Visibility_10m': float},
    #     'DROP': ["Unnamed: 0", "Rented Bike Count"],
    #     'TARGET_LABEL': 'Target',
    #     'TYPE': 'BINARY',
    #     'POS_CLASS': 1,
    #     'META_DATA': {'use_case': "demand-prediction", 'origin': 'adjusted'}
    # },
    #
    # 'binary_churn': {
    #     'DATA_SET': '../data/synthetic/churn_prob_out_35.csv',
    #     'DATA_TYPES': {'Children': float, 'RatePlan': str},
    #     'DROP': ['Id', 'pChurn', '3_Class', '5_Class', 'is_test_set'],
    #     'TARGET_LABEL': 'CHURN',
    #     'TYPE': 'BINARY',
    #     'POS_CLASS': 'T',
    #     'META_DATA': {'use_case': "churn", 'origin': 'synthetic'}
    # },
    #
    # 'miniloan': {
    #     'DATA_SET': '../data/synthetic/miniloan-decisions-100K.csv',
    #     'DATA_TYPES': {'creditScore': float, 'income': float, 'loanAmount': float, 'monthDuration': float,
    #                    'yearlyReimbursement': float},
    #     'DROP': ["Unnamed: 0", "name"],
    #     'TARGET_LABEL': 'approval',
    #     'TYPE': 'BINARY',
    #     'POS_CLASS': False,
    #     'META_DATA': {'use_case': 'credit', 'origin': "synthetic"}
    # },
    #
    # 'miniloan_ib': {
    #     'NAME': 'miniloan_ib',
    #     'DATA_SET': '../data/synthetic/miniloan_thin.csv',
    #     'DATA_TYPES': {'creditScore': float, 'income': float, 'loanAmount': float, 'monthDuration': float,
    #                    'yearlyReimbursement': float},
    #     'DROP': ["Unnamed: 0"],
    #     'TARGET_LABEL': 'approval',
    #     'TYPE': 'BINARY',
    #     'POS_CLASS': False,
    #     'META_DATA': {'use_case': 'credit', 'origin': 'synthetic'}
    # },
    #
    # 'fraud_detection_ib': {
    #     'DATA_SET': '../data/adjusted/imbalance/fraud/fraud_detection_thin.csv',
    #     'DATA_TYPES': {'Amount': float, 'Age': float, 'Bill_Sep': float, 'Bill_Aug': float, 'Bill_Jul': float,
    #                    'Bill_Jun': float, 'Bill_May': float, 'Bill_Apr': float, 'PayAmount_Sep': float,
    #                    'PayAmount_Aug': float, 'PayAmount_Jul': float, 'PayAmount_Jun': float, 'PayAmount_May': float,
    #                    'PayAmount_Apr': float},
    #     'DROP': ["Unnamed: 0"],
    #     'TARGET_LABEL': 'Class',
    #     'TYPE': 'BINARY',
    #     'POS_CLASS': 1,
    #     'META_DATA': {'use_case': 'transaction-fraud', 'origin': 'adjusted'}
    # },
    #
    # 'fraud_detection_ib_v22': {
    #     'DATA_SET': '../data/adjusted/imbalance/fraud/fraud_data_imbalanced2.csv',
    #     'DATA_TYPES': {'Amount': float, 'Age': float, 'Bill_Sep': float, 'Bill_Aug': float, 'Bill_Jul': float,
    #                    'Bill_Jun': float, 'Bill_May': float, 'Bill_Apr': float, 'PayAmount_Sep': float,
    #                    'PayAmount_Aug': float, 'PayAmount_Jul': float, 'PayAmount_Jun': float, 'PayAmount_May': float,
    #                    'PayAmount_Apr': float},
    #     'DROP': [],
    #     'TARGET_LABEL': 'Class',
    #     'TYPE': 'BINARY',
    #     'POS_CLASS': 1,
    #     'META_DATA': {'use_case': "transaction-fraud", 'origin': 'adjusted'}
    # },
    #
    # 'fraud_detection_ib_v3': {
    #     'DATA_SET': '../data/adjusted/imbalance/fraud/fraud_data_imbalanced3.csv',
    #     'DATA_TYPES': {'Amount': float, 'Age': float, 'Bill_Sep': float, 'Bill_Aug': float, 'Bill_Jul': float,
    #                    'Bill_Jun': float, 'Bill_May': float, 'Bill_Apr': float, 'PayAmount_Sep': float,
    #                    'PayAmount_Aug': float, 'PayAmount_Jul': float, 'PayAmount_Jun': float, 'PayAmount_May': float,
    #                    'PayAmount_Apr': float},
    #     'DROP': [],
    #     'TARGET_LABEL': 'Class',
    #     'TYPE': 'BINARY',
    #     'POS_CLASS': 1,
    #     'META_DATA': {'use_case': 'transaction-fraud', 'origin': 'adjusted'}
    # },
    #
    # 'fraud_detection_ib_v4': {
    #     'NAME': 'fraud_detection_ib_v4',
    #     'DATA_SET': '../data/adjusted/imbalance/fraud/fraud_data_imbalanced4.csv',
    #     'DATA_TYPES': {'Amount': float, 'Age': float, 'Bill_Sep': float, 'Bill_Aug': float, 'Bill_Jul': float,
    #                    'Bill_Jun': float, 'Bill_May': float, 'Bill_Apr': float, 'PayAmount_Sep': float,
    #                    'PayAmount_Aug': float, 'PayAmount_Jul': float, 'PayAmount_Jun': float, 'PayAmount_May': float,
    #                    'PayAmount_Apr': float},
    #     'DROP': [],
    #     'TARGET_LABEL': 'Class',
    #     'TYPE': 'BINARY',
    #     'POS_CLASS': 1,
    #     'META_DATA': {'use_case': 'transaction-fraud', 'origin': 'adjusted'}
    # },
    #
    # 'fraud_detection_ib_v5': {
    #     'NAME': 'fraud_detection_ib_v5',
    #     'DATA_SET': '../data/adjusted/imbalance/fraud/fraud_data_imbalanced5.csv',
    #     'DATA_TYPES': {'Amount': float, 'Age': float, 'Bill_Sep': float, 'Bill_Aug': float, 'Bill_Jul': float,
    #                    'Bill_Jun': float, 'Bill_May': float, 'Bill_Apr': float, 'PayAmount_Sep': float,
    #                    'PayAmount_Aug': float, 'PayAmount_Jul': float, 'PayAmount_Jun': float, 'PayAmount_May': float,
    #                    'PayAmount_Apr': float},
    #     'DROP': [],
    #     'TARGET_LABEL': 'Class',
    #     'TYPE': 'BINARY',
    #     'POS_CLASS': 1,
    #     'META_DATA': {'use_case': "transaction-fraud", 'origin': 'adjusted'}
    # },
    #
    # 'fraud_detection_ib_v6': {
    #     'DATA_SET': '../data/adjusted/imbalance/fraud/fraud_data_imbalanced6.csv',
    #     'DATA_TYPES': {'Amount': float, 'Age': float, 'Bill_Sep': float, 'Bill_Aug': float, 'Bill_Jul': float,
    #                    'Bill_Jun': float, 'Bill_May': float, 'Bill_Apr': float, 'PayAmount_Sep': float,
    #                    'PayAmount_Aug': float, 'PayAmount_Jul': float, 'PayAmount_Jun': float, 'PayAmount_May': float,
    #                    'PayAmount_Apr': float},
    #     'DROP': [],
    #     'TARGET_LABEL': 'Class',
    #     'TYPE': 'BINARY',
    #     'POS_CLASS': 1,
    #     'META_DATA': {'use_case': "transaction-fraud", 'origin': 'adjusted'}
    # },
    #
    # 'magic': {
    #     'DATA_SET': '../data/other/magic.csv',
    #     'DATA_TYPES': {},
    #     'DROP': ["Unnamed: 0"],
    #     'TYPE': 'BINARY',
    #     'TARGET_LABEL': 'class:',
    #     'POS_CLASS': "h",
    #     'META_DATA': {'use_case': "other", 'origin': 'organic','source': 'https://archive.ics.uci.edu/ml/datasets/magic+gamma+telescope',
    #                   'changes': 'none'}
    # },
    #
    # 'miniloan_noisy': {
    #     'DATA_SET': '../data/synthetic/noise/miniloan_noisy.csv',
    #     'DATA_TYPES': {'creditScore': float, 'income': float, 'loanAmount': float, 'monthDuration': float,
    #                    'yearlyReimbursement': float},
    #     'DROP': [],
    #     'TYPE': 'BINARY',
    #     'TARGET_LABEL': 'approval',
    #     'POS_CLASS': 1,
    #     'META_DATA': {'use_case': 'credit', 'origin': "adjusted"}
    # },
    #
    # 'miniloan_noisy2': {
    #     'DATA_SET': '../data/synthetic/noise/miniloan_noisy2.csv',
    #     'DATA_TYPES': {'creditScore': float, 'income': float, 'loanAmount': float, 'monthDuration': float,
    #                    'yearlyReimbursement': float},
    #     'DROP': [],
    #     'TARGET_LABEL': 'approval',
    #     'TYPE': 'BINARY',
    #     'POS_CLASS': 1,
    #     'META_DATA': {'use_case': "credit", 'origin': "adjusted"}
    # },
    #
    # "miniloan_noisy3": {
    #     'DATA_SET': '../data/synthetic/noise/miniloan_noisy3.csv',
    #     'DATA_TYPES': {'creditScore': float, 'income': float, 'loanAmount': float, 'monthDuration': float,
    #                    'yearlyReimbursement': float},
    #     'DROP': [],
    #     'TARGET_LABEL': 'approval',
    #     'TYPE': 'BINARY',
    #     'POS_CLASS': 1,
    #     'META_DATA': {'use_case': "credit", 'origin': "adjusted"}
    # },
    #
    # "miniloan_noisy4": {
    #     'NAME': 'miniloan_noisy4',
    #     'DATA_SET': '../data/synthetic/noise/miniloan_noisy4.csv',
    #     'DATA_TYPES': {'creditScore': float, 'income': float, 'loanAmount': float, 'monthDuration': float,
    #                    'yearlyReimbursement': float},
    #     'DROP': [],
    #     'TARGET_LABEL': 'approval',
    #     'TYPE': 'BINARY',
    #     'POS_CLASS': 1,
    #     'META_DATA': {'use_case': "credit", 'origin': "adjusted"}
    # },
    #
    # "miniloan_noisy5": {
    #     'DATA_SET': '../data/synthetic/noise/miniloan_noisy5.csv',
    #     'DATA_TYPES': {'creditScore': float, 'income': float, 'loanAmount': float, 'monthDuration': float,
    #                    'yearlyReimbursement': float},
    #     'DROP': [],
    #     'TARGET_LABEL': 'approval',
    #     'TYPE': 'BINARY',
    #     'POS_CLASS': 1,
    #     'META_DATA': {'use_case': "credit", 'origin': "adjusted"}
    # },
    #
    # "miniloan_noisy6": {
    #     'DATA_SET': '../data/synthetic/noise/miniloan_noisy6.csv',
    #     'DATA_TYPES': {'creditScore': float, 'income': float, 'loanAmount': float, 'monthDuration': float,
    #                    'yearlyReimbursement': float},
    #     'DROP': [],
    #     'TARGET_LABEL': 'approval',
    #     'TYPE': 'BINARY',
    #     'POS_CLASS': 1,
    #     'META_DATA': {'use_case': "credit", 'origin': "adjusted"}
    # },
    #
    # "miniloan_noisy7": {
    #     'DATA_SET': '../data/synthetic/noise/miniloan_noisy7.csv',
    #     'DATA_TYPES': {'creditScore': float, 'income': float, 'loanAmount': float, 'monthDuration': float,
    #                    'yearlyReimbursement': float},
    #     'DROP': [],
    #     'TARGET_LABEL': 'approval',
    #     'TYPE': 'BINARY',
    #     'POS_CLASS': 1,
    #     'META_DATA': {'use_case': "credit", 'origin': "adjusted"}
    # },
    #
    # "miniloan_noisy8": {
    #     'DATA_SET': '../data/synthetic/noise/miniloan_noisy8.csv',
    #     'DATA_TYPES': {'creditScore': float, 'income': float, 'loanAmount': float, 'monthDuration': float,
    #                    'yearlyReimbursement': float},
    #     'DROP': [],
    #     'TARGET_LABEL': 'approval',
    #     'TYPE': 'BINARY',
    #     'POS_CLASS': 1,
    #     'META_DATA': {'use_case': "credit", 'origin': "adjusted"}
    # },
    #
    # "miniloan_noisy9": {
    #     'DATA_SET': '../data/synthetic/noise/miniloan_noisy9.csv',
    #     'DATA_TYPES': {'creditScore': float, 'income': float, 'loanAmount': float, 'monthDuration': float,
    #                    'yearlyReimbursement': float},
    #     'DROP': [],
    #     'TARGET_LABEL': 'approval',
    #     'TYPE': 'BINARY',
    #     'POS_CLASS': 1,
    #     'META_DATA': {'use_case': "credit", 'origin': "adjusted"}
    # },
    #
    # "miniloan_noisy10": {
    #     'DATA_SET': '../data/synthetic/noise/miniloan_noisy10.csv',
    #     'DATA_TYPES': {'creditScore': float, 'income': float, 'loanAmount': float, 'monthDuration': float,
    #                    'yearlyReimbursement': float},
    #     'DROP': [],
    #     'TARGET_LABEL': 'approval',
    #     'TYPE': 'BINARY',
    #     'POS_CLASS': 1,
    #     'META_DATA': {'use_case': "credit", 'origin': "adjusted"}
    # },
    #
    # "miniloan_noisy11": {
    #     'DATA_SET': '../data/synthetic/noise/miniloan_noisy11.csv',
    #     'DATA_TYPES': {'creditScore': float, 'income': float, 'loanAmount': float, 'monthDuration': float,
    #                    'yearlyReimbursement': float},
    #     'DROP': [],
    #     'TARGET_LABEL': 'approval',
    #     'TYPE': 'BINARY',
    #     'POS_CLASS': 1,
    #     'META_DATA': {'use_case': "credit", 'origin': "adjusted"}
    # },
    #
    # "miniloan_noisy12": {
    #     'DATA_SET': '../data/synthetic/noise/miniloan_noisy12.csv',
    #     'DATA_TYPES': {'creditScore': float, 'income': float, 'loanAmount': float, 'monthDuration': float,
    #                    'yearlyReimbursement': float},
    #     'DROP': [],
    #     'TARGET_LABEL': 'approval',
    #     'TYPE': 'BINARY',
    #     'POS_CLASS': 1,
    #     'META_DATA': {'use_case': "credit", 'origin': "adjusted"}
    # },
    #
    # "miniloan_noisy13": {
    #     'DATA_SET': '../data/synthetic/noise/miniloan_noisy13.csv',
    #     'DATA_TYPES': {'creditScore': float, 'income': float, 'loanAmount': float, 'monthDuration': float,
    #                    'yearlyReimbursement': float},
    #     'DROP': [],
    #     'TARGET_LABEL': 'approval',
    #     'TYPE': 'BINARY',
    #     'POS_CLASS': 1,
    #     'META_DATA': {'use_case': "credit", 'origin': "adjusted"}
    # },
    #
    # "fraud_detection_noisy": {
    #     'DATA_SET': '../data/adjusted/noise/fraud_detection_noisy.csv',
    #     'DATA_TYPES': {'Amount': float, 'Age': float, 'Bill_Sep': float, 'Bill_Aug': float, 'Bill_Jul': float,
    #                    'Bill_Jun': float, 'Bill_May': float, 'Bill_Apr': float, 'PayAmount_Sep': float,
    #                    'PayAmount_Aug': float, 'PayAmount_Jul': float, 'PayAmount_Jun': float, 'PayAmount_May': float,
    #                    'PayAmount_Apr': float},
    #     'DROP': [],
    #     'TARGET_LABEL': 'Class',
    #     'TYPE': 'BINARY',
    #     'POS_CLASS': 1,
    #     'META_DATA': {'use_case': "fraud", 'origin': 'adjusted'}
    # },
}


def core_suite():
    return [config for config in CONFIG_DICT if
            (CONFIG_DICT[config]['META_DATA']['origin'] == 'organic' and
             CONFIG_DICT[config]['META_DATA']['use_case'] != 'other')]


def synth_imbalance_suite():
    return [config for config in CONFIG_DICT if
            config.startswith('synth_ib')]


def synth_noise_suite():
    return [config for config in CONFIG_DICT if
            config.startswith('synth_noise')]


def synth_disjunctive_complexity_suite():
    return [config for config in CONFIG_DICT if
            config.startswith('synth_disj_')]


def synth_conjunctive_complexity_suite():
    return [config for config in CONFIG_DICT if
            config.startswith('synth_conj_')]


def synth_linear_complexity_suite():
    return [config for config in CONFIG_DICT if
            config.startswith('synth_linear_')]


RULES_DIR = 'rules/'
IMBALANCE_DIR = '../data/synthetic/imbalance_suite/'
NOISE_DIR = '../data/synthetic/noise_suite/'
DISJUNCTIVE_DIR = '../data/synthetic/disjunctive_suite/'
LINEAR_DIR = '../data/synthetic/linear_suite/'
CONJUNCTIVE_DIR = '../data/synthetic/conjunctive_suite/'
SIZE_DIR = '../data/synthetic/size_suite/'
for path in [RULES_DIR, IMBALANCE_DIR, NOISE_DIR, DISJUNCTIVE_DIR, LINEAR_DIR, CONJUNCTIVE_DIR, SIZE_DIR]:
    if not exists(path):
        os.makedirs(path)


print('Generating synthetic prediction problems. Working directory {}.'.format(os.getcwd()))

# IMBALANCE SUITE
print('Configuring synthetic imbalance suite.')
N_ROWS = 100000
RULESET = [lambda x: x[0] < 0.5]
prefix = 'synth_ib_'
path = IMBALANCE_DIR + prefix
for i in range(1, 14):
    index = i-1
    key = prefix + str(index)
    file_name = path + str(index) + '.csv'
    d = {'DATA_SET': file_name,
         'DATA_TYPES': {'x1': float, 'x2': float, 'x3': float, 'y': str},
         'DROP': [],
         'TARGET_LABEL': 'y',
         'TYPE': 'BINARY',
         'POS_CLASS': '1',
         'META_DATA': {'use_case': "other", 'origin': 'synthetic'}}
    CONFIG_DICT[key] = d
    if not exists(file_name):
        print('Generating', file_name)
        df = generate_imbalanced_data(ratio=(2**(-i)), n_cols=3, n_rows=N_ROWS, ruleset=RULESET)
        df.to_csv(file_name, index=False)


# NOISE SUITE
print('Configuring synthetic noise suite.')
N_ROWS = 50000
RULESET = [lambda x: x[0] < 0.5]
prefix = 'synth_noise_'
path = NOISE_DIR + prefix
for i in range(4):
    index = i
    key = prefix + str(index)
    file_name = path + str(index) + '.csv'
    d = {'DATA_SET': file_name,
         'DATA_TYPES': {'x1': float, 'x2': float, 'x3': float, 'y': str},
         'DROP': [],
         'TARGET_LABEL': 'y',
         'TYPE': 'BINARY',
         'POS_CLASS': '1',
         'META_DATA': {'use_case': "other", 'origin': 'synthetic'}}
    CONFIG_DICT[key] = d
    if not exists(file_name):
        print('Generating', file_name)
        df = generate_imbalanced_data(ratio=0.5, n_cols=3, n_rows=N_ROWS, ruleset=RULESET)
        df_noisy = make_noisy(df=df, label_col='y', error_rate=(i * 0.1))
        df_noisy.to_csv(file_name, index=False)
for j in range(1, 9):
    index = i*100+j*25
    key = prefix + str(index)
    file_name = path + str(index) + '.csv'
    d = {'DATA_SET': file_name,
         'DATA_TYPES': {'x1': float, 'x2': float, 'x3': float, 'y': str},
         'DROP': [],
         'TARGET_LABEL': 'y',
         'TYPE': 'BINARY',
         'POS_CLASS': '1',
         'META_DATA': {'use_case': "other", 'origin': 'synthetic'}}
    CONFIG_DICT[key] = d
    if not exists(file_name):
        print('Generating', file_name)
        df = generate_imbalanced_data(ratio=0.5, n_cols=3, n_rows=N_ROWS, ruleset=RULESET)
        df_noisy = make_noisy(df=df, label_col='y', error_rate=(i * 0.1 + j * 0.025))
        df_noisy.to_csv(file_name, index=False)

# DISJUNCTIVE COMPLEXITY SUITE
print('Configuring synthetic disjunctive complexity suite.')
prefix = 'synth_disj_'
path = DISJUNCTIVE_DIR + prefix
N_ROWS = 100000
for j in range(4):
    index = 10**j
    key = prefix + str(index)
    file_name = path + str(index) + '.csv'
    d = {'DATA_SET': file_name,
         'DATA_TYPES': {'x1': float, 'x2': float, 'x3': float, 'y': str},
         'DROP': [],
         'TARGET_LABEL': 'y',
         'TYPE': 'BINARY',
         'POS_CLASS': '1',
         'META_DATA': {'use_case': "other", 'origin': 'synthetic'}}
    CONFIG_DICT[key] = d
    if not exists(file_name):
        print('Generating', file_name)
        text_file = open(RULES_DIR+prefix+str(index)+'.txt', "w")
        rule_set = generate_interval_rule_set(index, text_file)
        text_file.close()
        df = generate_imbalanced_data(ratio=0.5, n_cols=3, n_rows=N_ROWS, ruleset=rule_set)
        df.to_csv(file_name, index=False)
for j in range(2, 10):
    index = j
    key = prefix + str(index)
    file_name = path + str(index) + '.csv'
    d = {'DATA_SET': file_name,
         'DATA_TYPES': {'x1': float, 'x2': float, 'x3': float, 'y': str},
         'DROP': [],
         'TARGET_LABEL': 'y',
         'TYPE': 'BINARY',
         'POS_CLASS': '1',
         'META_DATA': {'use_case': "other", 'origin': 'synthetic'}}
    CONFIG_DICT[key] = d
    if not exists(file_name):
        print('Generating', file_name)
        text_file = open(RULES_DIR+prefix+str(index)+'.txt', "w")
        rule_set = generate_interval_rule_set(j, text_file)
        text_file.close()
        df = generate_imbalanced_data(ratio=0.5, n_cols=3, n_rows=N_ROWS, ruleset=rule_set)
        df.to_csv(file_name, index=False)

# LINEAR SUITE
print('Configuring synthetic linear complexity suite.')
prefix = 'synth_linear_'
path = LINEAR_DIR + prefix
N_ROWS = 70000
for j in range(10):
    index = j
    key = prefix + str(index)
    file_name = path + str(index) + '.csv'
    d = {'DATA_SET': file_name,
         'DATA_TYPES': {'y': str},
         'DROP': [],
         'TARGET_LABEL': 'y',
         'TYPE': 'BINARY',
         'POS_CLASS': '1',
         'META_DATA': {'use_case': "other", 'origin': 'synthetic'}}
    CONFIG_DICT[key] = d
    if not exists(file_name):
        print('Generating', file_name)
        text_file = open(RULES_DIR+prefix+str(index)+'.txt', "w")
        rule_set = generate_linear_rule_set(j, text_file)
        text_file.close()
        df = generate_imbalanced_data(ratio=0.5, n_cols=j+1, n_rows=N_ROWS, ruleset=rule_set)
        df.to_csv(file_name, index=False)

# IMBALANCED CONJUNCTIVE SUITE
print('Configuring synthetic conjunctive complexity suite - preliminary version.')
prefix = 'synth_conj1_'
path = CONJUNCTIVE_DIR + prefix
N_ROWS = 100000
for j in range(10):
    index = j
    key = prefix + str(index)
    file_name = path + str(index) + '.csv'
    d = {'DATA_SET': file_name,
         'DATA_TYPES': {'y': str},
         'DROP': [],
         'TARGET_LABEL': 'y',
         'TYPE': 'BINARY',
         'POS_CLASS': 'True',
         'META_DATA': {'use_case': "other", 'origin': 'synthetic'}}
    CONFIG_DICT[key] = d
    if not exists(file_name):
        print('Generating', file_name)
        text_file = open(RULES_DIR+prefix+str(index)+'.txt', "w")
        rule_set = generate_conjunctive_rule(j, text_file)
        text_file.close()
        df = generate_uniform_data(n_cols=j+1, n_rows=N_ROWS, ruleset=rule_set)
        # generate_imbalanced_data(ratio=0.5, n_cols=j+1, n_rows=N_ROWS, ruleset=rule_set)
        df.to_csv(file_name, index=False)

# CONJUNCTIVE SUITE
print('Configuring synthetic conjunctive complexity suite.')
prefix = 'synth_conj_'
path = CONJUNCTIVE_DIR + prefix
N_ROWS = 50000
for j in [*range(10)] + [20, 50, 100, 200, 400]:
    index = j
    key = prefix + str(index)
    file_name = path + str(index) + '.csv'
    d = {'DATA_SET': file_name,
         'DATA_TYPES': {'y': str},
         'DROP': [],
         'TARGET_LABEL': 'y',
         'TYPE': 'BINARY',
         'POS_CLASS': 'True',
         'META_DATA': {'use_case': "other", 'origin': 'synthetic'}}
    CONFIG_DICT[key] = d
    if not exists(file_name):
        print('Generating', file_name)
        text_file = open(RULES_DIR+prefix+str(index)+'.txt', "w")
        rule_set = generate_conjunctive_rule2(j, text_file)
        text_file.close()
        df = generate_uniform_data(n_cols=j+1, n_rows=N_ROWS, ruleset=rule_set)
        # generate_imbalanced_data(ratio=0.5, n_cols=j+1, n_rows=N_ROWS, ruleset=rule_set)
        df.to_csv(file_name, index=False)

# SIZE SUITE
print('Configuring synthetic size variation suite.')
prefix = 'synth_size_'
path = SIZE_DIR + prefix
RULESET = [lambda x: x[0] <= 0.5]
for j in range(14):
    index = 10 * 2**j
    key = prefix + str(index)
    file_name = path + str(index) + '.csv'
    d = {'DATA_SET': file_name,
         'DATA_TYPES': {'y': str},
         'DROP': [],
         'TARGET_LABEL': 'y',
         'TYPE': 'BINARY',
         'POS_CLASS': '1',
         'META_DATA': {'use_case': "other", 'origin': 'synthetic'}}
    CONFIG_DICT[key] = d
    if not exists(file_name):
        print('Generating', file_name)
        df = generate_imbalanced_data(ratio=0.5, n_cols=1, n_rows=index, ruleset=RULESET)
        # df = generate_uniform_data(n_cols=1, n_rows=index, ruleset=RULESET)
        df.to_csv(file_name, index=False)

# RULESET = [lambda x: x[0] < 0.2 and x[2] > 0.7,
#            lambda x: 0.5 < x[1] < 0.8,
#            lambda x: x[2] < 0.1 and x[1] < 0.1]
# RULESET2 = [lambda x: x[0]+x[1] < 1.0]  # [lambda x: x[0]/x[1] < 0.5 and 0.85 > x[2] > 0.7]
# df = generate_uniform_data(n_rows=N_ROWS, ruleset=RULESET2)
# df.to_csv('../data/synthetic/linear.csv', index=False)
