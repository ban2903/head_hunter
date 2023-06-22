from config import *
from catboost import CatBoostClassifier, Pool
import requests
import pandas as pd 
import re 
import string
import numpy as np 
from typing import List, Dict
import os
import shap
import joblib
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import pickle

class SalaryPrediciton:

    TAG_RE = re.compile(r'<[^>]+>')
    table = str.maketrans(dict.fromkeys(string.punctuation)) # delete punctuation
    # stop_words = stopwords.words('russian')
    
    def __init__(self) -> None:
        self.model = CatBoostClassifier().load_model(PATH)
        self.FEATURES = self.model.feature_names_
        self.professional_roles = pd.read_csv('profession_roles.csv')
        self.skills_top300 = pd.read_csv('skills_top300.csv')['skill'].values
        self.skills_top300 = {k:i for i,k in enumerate(self.skills_top300)}
        self.dictionary_shedule = self.get_schedule_dictionary()
        self.rosstat_mapping_area = self.get_rosstat_mapping()
        self.vectorizer = pickle.load(open('./model/tfidf.pkl', 'rb'))
        self.nlp = spacy.load('ru_core_news_sm', disable=['parser', 'ner'])

    def get_desc_lemm(self, description):
        doc = self.nlp(description)
        return ' '.join([token.lemma_ for token in doc])

    def get_tfidf_vector(self, description, dictionary):
        tfidf_array = np.array(self.vectorizer.transform([description]).todense(), dtype=np.float32)
        tfidf_array = np.delete(tfidf_array, [0, 1, 2, 3, 5, 7, 8, 9], 1)
        columns = np.delete(self.vectorizer.get_feature_names_out(), [0, 1, 2, 3, 5, 7, 8, 9])
        for i in range(len(columns)):
            dictionary['description_tfidf_'+columns[i]] = tfidf_array[:, i]
        return dictionary
    
    def get_rosstat_mapping(self):
        city_dictionary = pd.read_csv('city_dictionary.csv')
        pathes = []
        root = '../data/info-stat/'
        self.mapping = []
        for directory in os.listdir(root):
            for file in os.listdir(root + directory):
                pathes.append(root + directory + '/' + file)
        for i, path in enumerate(pathes):
            # print(path)
            rosstat = pd.read_excel(path)
            columns = ['out'] + [f'feature_{i}{k}'for k in range(len(rosstat.columns[1:]))]
            self.mapping.append({f'feature_{i}{k}': path for k in range(len(rosstat.columns[1:]))})
            rosstat.columns = columns
            rosstat = rosstat[columns].sort_values(columns).drop_duplicates(['out'], keep='first')
            rosstat['out'] = rosstat['out'].str.strip() # в некоторых колонках есть пробелы на конце
            rosstat['out'] = rosstat['out'].apply(
                lambda x: ' '.join(
                    [re.sub(r'[0-9]', '', xx) for xx in x.split()]
                    )
            )
            rosstat['out'] = rosstat['out'].str.replace(':', ',')
            for c in columns[1:]:
                rosstat[c] = rosstat[c].astype(str)
                rosstat[c] = pd.to_numeric(rosstat[c], errors='coerce')
            city_dictionary = city_dictionary.merge(rosstat, on=['out'], how='left')
        return city_dictionary.drop_duplicates(subset='out')
    
    def get_schedule_dictionary(self,):
        dictionary_shedule = {}
        for x in requests.get('https://api.hh.ru/dictionaries').json()['schedule']:
            key, value = x.items()
            dictionary_shedule[value[1]] = key[1]
        return dictionary_shedule

    def predict(self, data: pd.DataFrame) -> float:
        return self.model.predict(self.pool(data))

    def get_features(self, url: str) -> pd.DataFrame:
        features = {}

        id = re.findall('[0-9]+', url)[0]
        r = requests.get(f'https://api.hh.ru/vacancies/{id}')

        if r.status_code == 404:
            print('Кажется, такой ссылки не существует')
            raise
        if r.status_code == 403:
            print('Сервис отдыхает, приходите через 30 минут')
            raise

        answer = r.json()

        # сборка нужных переменных 
        features['item_id'] = [id]
        features['real_salary_from'] = [np.nan]
        features['real_salary_to'] = [np.nan]
        if not (answer['salary'] is None):
            if not (answer['salary'].get('from', None) is None):
                features['real_salary_from'] = [answer['salary']['from']  * (1.0 if answer['salary'].get('is_gross', 1) else 0.13)]
            if not (answer['salary'].get('to', None) is None):
                features['real_salary_to'] = [answer['salary']['to']  * (1.0 if answer['salary'].get('is_gross', 1) else 0.13)]
        features['url'] = [f'https://api.hh.ru/vacancies/{id}']
        features['billing_type'] = [answer['billing_type']['name']]
        features['schedule'] = [answer['schedule']['name']]
        features['name'] = [answer['name']]
        features['area'] = [answer['area']['name']]
        features['area_id'] = [answer['area']['id']] # надо будет разобраться что с ним делать
        features['allow_messages'] = [answer['allow_messages']]
        features['experience'] = [answer['experience']['name']]
        features['accept_handicapped'] = [answer['accept_handicapped']]
        features['accept_kids'] = [answer['accept_kids']]
        features['accept_temporary'] = [answer['accept_temporary']]
        features['15'] = [self.get_category_name(int(answer['professional_roles'][0]['id']))]
        features['professional_roles_id'] = [answer['professional_roles'][0]['id']]
        features['lat'] = [np.nan]
        features['lng'] = [np.nan]
        address = answer['address']
        if not (address is None):
            features['lat'] = address['lat']
            features['lng'] = address['lng']
        features['department_name'] = ['None']
        features['has_department'] = [0]
        department = answer['department']
        if not (department is None):
            features['department_name'] = [dict(eval(str(department)))['name']]
            features['has_department'] = [1]
        features['description_clear'] = [self.remove_tags(answer['description'])]
        features['description_len'] = [features['description_clear'][0].__len__()]
        features['description_clear_lemm'] = [self.get_desc_lemm(features['description_clear'][0])]
        features['uniq_skills_cnt'] = [0] 
        features['uniq_popular_skills_cnt'] = [0]
        features['key_skills_embeddiong'] = [np.zeros(self.skills_top300.keys().__len__())]
        key_skills = answer['key_skills']
        if not (key_skills is None):
            features['uniq_skills_cnt'], features['uniq_popular_skills_cnt'], features['key_skills_embeddiong'] = self.get_skills_features(key_skills)
        languages = answer['languages']
        features['is_engl'] = [0]
        features['is_chi'] = [0]
        features['is_ger'] = [0]
        features['cnt_lang'] = [0]
        if (not (languages is None)) or (len(languages) != 0):
            for language in languages:
                if language['name'] == 'Английский':
                    features['is_engl'] = [1]
                elif language['name'] == 'Немецкий':
                    features['is_ger'] = [1]
                elif language['name'] == 'Китайский':
                    features['is_chi'] = [1]
            features['cnt_lang'] = languages.__len__()
        features['mean_similar'], features['min_similar'], features['max_similar'] = self.get_similar_salary(
            id=features['item_id'][0],
            professional_role=features['professional_roles_id'][0],
            schedule=features['schedule'][0],
        )
        rosstat = self.get_rosstat(features['area_id'][0])
        # print(rosstat)
        # print(FEATURES_ROSSTAT)
        for ros_feat in FEATURES_ROSSTAT:
            features[ros_feat] = [np.nan]
            if rosstat.shape[0] != 0: # нашли такой регион 
                # print(ros_feat)
                features[ros_feat] = [rosstat[ros_feat].iloc[0]]
        features['dollar_rate'] = self.get_dollar_rate(answer['published_at'])
        features['employer'] = [answer['employer'].get('name', 'None')]
        features = self.get_tfidf_vector(features['description_clear_lemm'][0], features)
        return pd.DataFrame(features)

    def get_rosstat(self, area_id:str):
        region_name, _ = self.get_parent_area(area_id)
        return self.rosstat_mapping_area[self.rosstat_mapping_area['in']==region_name]

    def get_dollar_rate(self, dt):
        dt = pd.to_datetime(dt).strftime('%Y-%m-%d')
        base="USD"
        out_curr="RUB"
        start_date = dt
        end_date = dt
        # start_date, end_date
        url = 'https://api.exchangerate.host/timeseries?base={0}&start_date={1}&end_date={2}&symbols={3}'.format(base,start_date,end_date,out_curr)
        response = requests.get(url)
        r = response.json()
        return r['rates'][dt]['RUB']

    @staticmethod
    def get_parent_area(id):
        id_array = [id]
        while not (id_array[-1] is None):
            r = requests.get(f'https://api.hh.ru/areas/{id_array[-1]}')
            id_array.append(r.json()['parent_id'])
        r_region = requests.get(f'https://api.hh.ru/areas/{id_array[-3]}') # т.к -1 = None, -2 = Country
        r_country = requests.get(f'https://api.hh.ru/areas/{id_array[-2]}') # т.к -1 = None, -2 = Country
        return r_region.json()['name'], r_country.json()['name']

    def set_dataset(self, dataset):
        self.dataset = dataset

    def get_skills_features(self, skills: List[dict]):
        uniq_skills_cnt = skills.__len__()
        uniq_popular_skills_cnt = 0
        skill_top_array = self.skills_top300.keys()
        key_skills_embeddiong = np.zeros(skill_top_array.__len__())
        for skill in skills:
            if skill['name'] in skill_top_array:
                uniq_popular_skills_cnt += 1
                key_skills_embeddiong[self.skills_top300[skill['name']]] = 1 # массив топ 300 скиллов, помечаем = 1 если увидели
        return [uniq_skills_cnt], [uniq_popular_skills_cnt], [key_skills_embeddiong]
    
    def remove_tags(self, text:str) -> str:
        text = self.TAG_RE.sub('', text)
        text = text.translate(self.table).lower()
        return ' '.join(text.split()) # delete extra spaces
    
    def pool(self, data: pd.DataFrame) -> Pool:
        return Pool(
            data=data[self.FEATURES],
            cat_features=CAT_FEATURES,
            text_features=TEXT_FEATURES,
        )
    
    def get_category_name(self,id:int) -> str:
        return self.professional_roles[self.professional_roles.profession_id == id]['categories_name'].values[0]
    
    def get_department_name(self, department:str)->str:
        if department is None:
            return np.nan
        department = dict(eval(str(department)))
        return department['name']
    
    def get_similar_salary(self, id:int, professional_role:int, schedule:str):
        params = {
            'per_page':'100',
            'page': 0,
            'only_with_salary': True, 
            'professional_role': professional_role,
            'schedule': self.dictionary_shedule[schedule],
        }
        url = f'https://api.hh.ru/vacancies/{id}/similar_vacancies'
        items = requests.get(url, params=params)
        items = items.json()
        zp_all = []
        for item in items['items']:
            if not (item['salary']['from'] is None):
                zp_all.append(item['salary']['from'])
        zp_all = np.array(zp_all, dtype=np.float64)
        if len(zp_all) == 0:
            return np.nan, np.nan, np.nan
        return zp_all.mean(), zp_all.min(), zp_all.max()
    
    def shap_plot(self, dataset: pd.DataFrame):
        explainer = joblib.load('explainer2.bz2')
        pool = self.pool(data=dataset)
        shap_values = explainer.shap_values(pool)
        value = dataset[self.FEATURES].iloc[0,:]
        value['name'] = 'name'
        # value['description_clear'] = 'desc'
        # return value, shap_values, explainer
        # print(value.shape, shap_values.shape, explainer.expected_value.shape)
        class_ = self.model.predict_proba(pool).argmax()
        print(self.model.classes_[class_])
        shap.plots.force(
            explainer.expected_value[class_],
            shap_values[class_][0,:],
            value,
            matplotlib=True,
            figsize=(100, 5),
            text_rotation=0.7, 
            contribution_threshold=0.05, 
            show=False
        ).savefig('scratch.pdf',format = "pdf",dpi = 700,bbox_inches = 'tight')




if __name__ == '__main__':
    model = SalaryPrediciton()
    df = model.get_features('https://hh.ru/vacancy/76085328')
    score = model.predict(df)
    # score = model.predict('https://hh.ru/vacancy/76085328')
    print(score)