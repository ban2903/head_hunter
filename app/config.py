PATH = './model/base2.cbm'

FEATURES_ROSSTAT = ['feature_00', 'feature_10', 'feature_20', 'feature_30',
       'feature_40', 'feature_50', 'feature_60', 'feature_70', 'feature_80',
       'feature_90', 'feature_100', 'feature_110', 'feature_120',
       'feature_130', 'feature_140', 'feature_150', 'feature_160',
       'feature_170', 'feature_180', 'feature_190', 'feature_200',
       'feature_210', 'feature_220', 'feature_230', 'feature_240',
       'feature_250', 'feature_260', 'feature_270', 'feature_280',
       'feature_290', 'feature_300', 'feature_310', 'feature_320',
       'feature_330', 'feature_340', 'feature_350', 'feature_360',
       'feature_370', 'feature_380', 'feature_390', 'feature_400',
       'feature_410', 'feature_411', 'feature_412', 'feature_413',
       'feature_414', 'feature_415', 'feature_420', 'feature_430',
       'feature_440', 'feature_450'] # росстат

DOP_FEATURES = ['mean_similar', 'min_similar', 'max_similar', 'feature_290', 'feature_370']

FEATURES = [
    'billing_type',
    'schedule',
    'name',
    'area',
    'allow_messages',
    'experience',
    'accept_handicapped',
    'accept_kids',
    'employer',
    'accept_temporary',
    '15',
    'lat',
    'lng',
    'department_name',
    'has_department',
    # 'description_clear',
    'description_len',
    'uniq_skills_cnt',
    'uniq_popular_skills_cnt',
    'professional_roles_id',
    'dollar_rate',
    'is_engl',
    'is_ger',
    'is_chi',
    'cnt_lang',
] + DOP_FEATURES

CAT_FEATURES = [
    'billing_type',
    'schedule',
    'area',
    'experience',
    'employer',
    '15',
    'department_name',
    'professional_roles_id'
]

TEXT_FEATURES = [
    'name',
    # 'description_clear'
]
