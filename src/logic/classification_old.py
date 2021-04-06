import sys
from sklearn.preprocessing import LabelEncoder
import joblib
from src.logic.load_data import GetTextPreProcess
import numpy as np
from root import DIR_MODELS
np.random.seed(1000)
np.warnings.filterwarnings('ignore')


class ClassificationModel:
    def __init__(self):
        self.best_model_clf = {}
        self.best_model_vec = {}
        self.configuration_pre_process = {}
        self.list_predict = {}
        self.encoder = {}
        self.models_file = DIR_MODELS
        self.list_models = ['gender', 'knowledge', 'age_range', 'depto', 'plan']

        self.encoder[self.list_models[0]] = LabelEncoder()
        self.encoder[self.list_models[1]] = LabelEncoder()
        self.encoder[self.list_models[2]] = LabelEncoder()
        self.encoder[self.list_models[3]] = LabelEncoder()
        self.encoder[self.list_models[4]] = LabelEncoder()

        self.configuration_pre_process[self.list_models[0]] = \
            {'name_method': "user_raw_2_5", 'clean_text': False, 'tracer': True}
        self.configuration_pre_process[self.list_models[1]] = \
            {'name_method': "user_raw_1_3", 'clean_text': False, 'tracer': True}
        self.configuration_pre_process[self.list_models[2]] = \
            {'name_method': "user_clear_1_1", 'clean_text': True, 'tracer': True}
        self.configuration_pre_process[self.list_models[3]] = \
            {'name_method': "depto_clear_1_2", 'clean_text': True, 'tracer': True}
        self.configuration_pre_process[self.list_models[4]] = \
            {'name_method': "plan_clear_1_2", 'clean_text': True, 'tracer': True}

        GetTextPreProcess.configuration = self.configuration_pre_process['gender']
        self.text_process = GetTextPreProcess([])

    def build_models(self):
        # Load Models
        print('\t ++ Load Models')
        self.best_model_clf[self.list_models[0]] = \
            joblib.load('{0}{1}'.format(self.models_file,
                                        'gender_LogisticRegression_user_raw_2_5_99AC_2020-01-03-14-16.sav'))
        self.best_model_clf[self.list_models[1]] = \
            joblib.load('{0}{1}'.format(self.models_file,
                                        'knowledge_base_MultinomialNB_user_raw_1_3_98AC_2020-03-02-10-58.sav'))
        self.best_model_clf[self.list_models[2]] = \
            joblib.load('{0}{1}'.format(self.models_file,
                                        'age_range_LogisticRegression_age_range_2_3_sampling_51_2020-04-19-02-26AC.sav'))
        self.best_model_clf[self.list_models[3]] = \
            joblib.load('{0}{1}'.format(self.models_file,
                                        'citizen_depto_es_MultinomialNB_2021-01-08-15-56_74AC_2021-01-08-15-59.sav'))
        self.best_model_clf[self.list_models[4]] = \
            joblib.load('{0}{1}'.format(self.models_file,
                                        'citizen_plan_es_LogisticRegression_2021-01-08-15-59_61AC_2021-01-08-16-19.sav'))
        # Load WordVector
        print('\t ++ Load WordVector')
        self.best_model_vec[self.list_models[0]] = \
            joblib.load('{0}{1}'.format(self.models_file,
                                        'gender_LogisticRegression_user_raw_2_5_2020-01-03-14-16.vec'))
        self.best_model_vec[self.list_models[1]] = \
            joblib.load('{0}{1}'.format(self.models_file,
                                        'knowledge_base_MultinomialNB_user_raw_1_3_2020-03-02-10-58.vec'))
        self.best_model_vec[self.list_models[2]] = \
            joblib.load('{0}{1}'.format(self.models_file,
                                        'age_range_LogisticRegression_age_range_2_3_sampling_2020-04-19-02-26.vec'))
        self.best_model_vec[self.list_models[3]] = \
            joblib.load('{0}{1}'.format(self.models_file,
                                        'citizen_depto_es_MultinomialNB_2021-01-08-15-56_2021-01-08-15-59.vec'))
        self.best_model_vec[self.list_models[4]] = \
            joblib.load('{0}{1}'.format(self.models_file,
                                        'citizen_plan_es_LogisticRegression_2021-01-08-15-59_2021-01-08-16-19.vec'))
        # Load Encoder
        print('\t ++ Load Encoder')
        self.encoder[self.list_models[0]].fit_transform(
            ['M', 'F'])
        self.encoder[self.list_models[1]].fit_transform(
            ['cultura', 'deportes', 'economia', 'politica', 'tecnologia', 'vida'])
        self.encoder[self.list_models[2]].fit_transform(['13-17', '18-24', '25-34', '35-49', '50-64', '65-xx'])
        self.encoder[self.list_models[3]].fit_transform([
            'Secretaria_de_Bienes_y_Servicios', 'Secretaria_de_Comunicaciones', 'Secretaria_de_Convivencia',
            'Secretaria_de_Cultura_Ciudadana', 'Secretaria_de_Educacion', 'Secretaria_de_Gestión_Humana_y_servicio_a_la_ciudadanía',
            'Secretaria_de_Gestión_Territorial', 'Secretaria_de_Gobierno', 'Secretaria_de_Hacienda', 'Secretaria_de_Inclusión',
            'Secretaria_de_Infraestructura', 'Secretaria_de_Juventud', 'Secretaria_de_Medio_Ambiente', 'Secretaria_de_Movilidad',
             'Secretaria_de_Mujeres', 'Secretaria_de_Participación_Ciudadana', 'Secretaria_de_Salud', 'Secretaria_de_Seguridad_y_Convivencia',
             'Secretaria_de_Suministros_y_Servicios',  'Secretaria_de__Desarrollo_Económico', 'Subsecretaria_de_Defensa_y_Protección_de_lo_Público',
             'Subsecretaria_de_Gob._Local_y_Convivencia'])
        self.encoder[self.list_models[4]].fit_transform(['Linea_1_recuperemos_lo_social', 'Linea_2_transformación_educativa',
                                                         'Linea_3_Ecociudad', 'Linea_4_Valle_del_software', 'Linea_5_Gobernanza_y_gobernabilidad'])

    def predict_classification(self, model, text):
        result = {'text_raw': text, 'text_process': None, 'model': model, 'predict': None, 'label': None,
                  'strength': 'low', 'state': True}
        # try:
        if model in ['depto', 'plan']:  # self.list_models:
            vectorizer_words = self.best_model_vec[model]
            classifier_text = self.best_model_clf[model]
            print(text)
            result['text_process'] = self.pre_process(model, text)
            print(result)
            vec_words_test = vectorizer_words.transform(result['text_process']).toarray()
            predicted = classifier_text.predict(vec_words_test)
            prob_classes = classifier_text.predict_proba(vec_words_test)
            if all(x == prob_classes[0][0] for x in prob_classes[0]) and prob_classes[0][int(predicted[0])] < 0.05:
                result['predict'] = -1
                result['label'] = 'otro'
                result['strength'] = 'low'
                result['strength_value'] = prob_classes[0][int(predicted[0])]
            else:
                result['predict'] = int(predicted[0])
                result['label'] = str(self.encoder[model].inverse_transform(predicted)[0])
                result['strength'] = 'low' if prob_classes[0][predicted] < 0.3 else 'high' \
                    if prob_classes[0][predicted] > 0.7 else 'medium'
                result['strength_value'] = prob_classes[0][int(predicted[0])]
        else:
            print('\t The model was not found')
            result['state'] = False
    # except Exception as e:
    #     print('classification', e)
        return result

    def pre_process(self, key, text):
        GetTextPreProcess.configuration = self.configuration_pre_process[key]
        new_text = self.text_process.process(text.lower())
        return [new_text]


if __name__ == "__main__":
    list_name = []
    best_models = ClassificationModel()
    best_models.build_models()

    tweets = ['Las medidas tomadas han sido positivas. Teniendo en cuenta el comportamiento del virus e información de salud pública analizada, hemos tomado la decisión de mantener el Aislamiento Preventivo Obligatorio hasta el 26 de abril a las 11:59 p.m. Nuestro reto es seguir salvando vidas.',
              'No sabe el lobo que caperucita va al bosque por él..',
              'Muchísimas gracias a los campesinos cuya labor es fundamental ahora y siempre!!!',
              'Me siento mal del estomago',
              'Que mal que se roben el pais de esa manera',
              'eres malo para todo',
              'Cuando nos volvamos a ver les preparo',
              'Gracias por tu ayuda. Mi mamá y mi hermana lloraron con el vídeo. Estaba muy contenta mi mamá. Me hiciste quedar como una princesa',
              'El proyecto en el que estoy es el que permite visualizar de manera consolidada la información de esos datamart']

    for text in tweets:
        # print(best_models.predict_classification('gender', text))
        # print(best_models.predict_classification('knowledge' , text))
        # print(best_models.predict_classification('age_range', text))
        print(best_models.predict_classification('depto', text))
        print(best_models.predict_classification('plan', text))

