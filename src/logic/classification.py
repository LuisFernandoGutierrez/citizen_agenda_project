import sys
from sklearn.preprocessing import LabelEncoder
import joblib
from src.logic.classifier.load_data import GetTextPreProcess
import numpy as np
import glob
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
        self.text_process = GetTextPreProcess()

    def build_models(self, models=[]):
        files_models = glob.glob(self.models_file+'/*')
        print('\t** files_models in folder', files_models)
        list_model_build = self.list_models if models == [] else models
        for name in list_model_build:
            filter_models = list(filter(lambda k: name in k, files_models))
            print('\t\t model: ', name, filter_models)
            for item_model_load in filter_models:
                if '.sav' in item_model_load:
                    self.best_model_clf[name] = joblib.load(item_model_load)
                else:
                    self.best_model_vec[name] = joblib.load(item_model_load)
        # Load Encoder
        print('\t ++ Load Encoder')
        self.encoder[self.list_models[0]].fit_transform(
            ['M', 'F'])
        self.encoder[self.list_models[1]].fit_transform(
            ['cultura', 'deportes', 'economia', 'politica', 'tecnologia', 'vida'])
        self.encoder[self.list_models[2]].fit_transform(['13-17', '18-24', '25-34', '35-49', '50-64', '65-xx'])
        self.encoder[self.list_models[3]].fit_transform([
            'Secretaria_de_Bienes_y_Servicios', 'Secretaria_de_Comunicaciones', 'Secretaria_de_Convivencia',
            'Secretaria_de_Cultura_Ciudadana', 'Secretaria_de_Educacion',
            'Secretaria_de_Gestión_Humana_y_servicio_a_la_ciudadanía',
            'Secretaria_de_Gestión_Territorial', 'Secretaria_de_Gobierno', 'Secretaria_de_Hacienda',
            'Secretaria_de_Inclusión',
            'Secretaria_de_Infraestructura', 'Secretaria_de_Juventud', 'Secretaria_de_Medio_Ambiente',
            'Secretaria_de_Movilidad',
            'Secretaria_de_Mujeres', 'Secretaria_de_Participación_Ciudadana', 'Secretaria_de_Salud',
            'Secretaria_de_Seguridad_y_Convivencia',
            'Secretaria_de_Suministros_y_Servicios', 'Secretaria_de__Desarrollo_Económico',
            'Subsecretaria_de_Defensa_y_Protección_de_lo_Público',
            'Subsecretaria_de_Gob._Local_y_Convivencia'])
        self.encoder[self.list_models[4]].fit_transform(
            ['Linea_1_recuperemos_lo_social', 'Linea_2_transformación_educativa',
             'Linea_3_Ecociudad', 'Linea_4_Valle_del_software', 'Linea_5_Gobernanza_y_gobernabilidad'])

    def predict_classification(self, model, text):
        result = {'text_raw': text, 'text_process': None, 'model': model, 'predict': None, 'label': None,
                  'strength': 'low', 'state': True}
        try:
            if model in self.list_models:
                # print('*******', model)
                vectorizer_words = self.best_model_vec[model]
                classifier_text = self.best_model_clf[model]
                # print(text)
                result['text_process'] = self.pre_process(model, text)
                # print(result)
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
        except Exception as e:
            print('classification', e)
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
              'El proyecto en el que estoy es el que permite visualizar de manera consolidada la información de esos datamart',
              'la tecnologia y el internet es importante en la sociedad y el software',
              'el plan de salud y las enfermedades asustan paz justicia y victimas',
              'Servicio públicos, energías alternativas y aprovechamiento de residuos solidos']

    for index, text in enumerate(tweets):
        print('-' * 50)
        print('text: ', text)
        for item_model in ['gender', 'knowledge', 'age_range', 'depto', 'plan']:
            result = best_models.predict_classification(item_model, text)
            print(index, item_model, result['predict'], result['label'], result['strength'], result['strength_value'])


