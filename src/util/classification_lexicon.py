import pandas as pd
from root import DIR_LEXICON
import nltk
import nltk.data
import nltk.metrics
from nltk.tokenize import word_tokenize
import unicodedata
import re
from nltk.stem import SnowballStemmer
from sklearn.preprocessing import LabelEncoder
import spacy


class ClassificationLexiconBased:
    def __init__(self, file_path=None, lang='es'):
        self.lang = 'spanish' if lang == 'es' else 'english'
        self._file_path = DIR_LEXICON + file_path
        self._encoder = LabelEncoder()
        self._lexicon = {}
        self._stopwords = nltk.corpus.stopwords.words(self.lang)
        self._stopwords_all = {}
        self.class_lexicon = {}
        # Stopwords
        for w in self._stopwords:
            w = w.lower()
            self._stopwords_all[w] = 1
            self._stopwords_all[self.proper_encoding(w)] = 1
        # Stemmer
        self._stemmer = SnowballStemmer(self.lang)
        # Lemmatization
        self.nlp = spacy.load('es_core_news_sm', disable=['parser', 'ner']) \
            if lang == 'es' else spacy.load('en_core_news_sm', disable=['parser', 'ner'])
        # Builds Models
        self.build()

    def build(self):
        df_lexicon = pd.read_csv(self._file_path, sep=';', encoding='utf-8')
        words_list = list(df_lexicon.values)
        list_value = []
        for item in words_list:
            word = item[0].lower()
            word_lemma = self.lemmatization(word)
            if item[1] not in self.class_lexicon:
                self.class_lexicon[item[1]] = 0
            self.class_lexicon[item[1]] += 1
            if len(word_lemma) > 0 and word != word_lemma:
                # print('word {} word_lemma {}'.format(word, word_lemma))
                self._lexicon[word_lemma] = {'word': word_lemma, 'label': item[1], 'value': item[2]}
                word_lemma = self.proper_encoding(word_lemma)
                self._lexicon[word_lemma] = {'word': word_lemma, 'label': item[1], 'value': item[2]}
                word_lemma = self._stemmer.stem(word_lemma)
                self._lexicon[word_lemma] = {'word': word_lemma, 'label': item[1], 'value': item[2]}

            self._lexicon[word] = {'word': word, 'label': item[1], 'value': item[2]}
            word = self.proper_encoding(word)
            self._lexicon[word] = {'word': word, 'label': item[1], 'value': item[2]}
            word = self._stemmer.stem(word)
            self._lexicon[word] = {'word': word, 'label': item[1], 'value': item[2]}
            list_value.append(item[1])
        self._encoder.fit_transform(list_value)
        print('\t** load lexicon {} with {} tokens {} real'.format(self._file_path, len(words_list), len(self._lexicon)))

    def calculate_value(self, eval_tokens):
        result_value_absolute = {}
        result_value_relative = {}
        result = {'best_relative': (' ', 0), 'best_absolute': (' ', 0), 'class_relative': [], 'class_absolute': [],
                  'words': [], 'id_best_relative': -1}
        try:
            for token in eval_tokens:
                result['words'].append(token['word'])
                if token['label'] not in result_value_absolute:
                    result_value_absolute[token['label']] = 0
                    result_value_relative[token['label']] = 0
                result_value_absolute[token['label']] += token['value']
                result_value_relative[token['label']] += 1/len(eval_tokens)

            list_result_relative = sorted(result_value_relative.items(), key=lambda x: x[1], reverse=True)
            list_result_absolute = sorted(result_value_absolute.items(), key=lambda x: x[1], reverse=True)
            # result['best_relative'] = list_result_relative[0] if len(list_result_relative) > 0 else ''
            result['best_relative'] = list_result_relative[0] if len(list_result_relative) > 0 else (' ', 0)
            result['best_absolute'] = list_result_absolute[0] if len(list_result_absolute) > 0 else (' ', 0)
            result['class_relative'] = result_value_relative
            result['class_absolute'] = result_value_absolute

            if result['best_relative'] != (' ', 0):
                result['id_best_relative'] = int(self._encoder.transform([list_result_relative[0][0]])[0])

        except Exception as e:
            print('calculate_value', e)

        return result

    def predict(self, text):
        word_result = None
        try:
            words_text = word_tokenize(str(text).lower())
            words_text = self.delete_stopword(words_text)
            words_lexicon = []
            for word in words_text:
                word = word
                word_found = False
                if word in self._lexicon:
                    words_lexicon.append(self._lexicon[word])
                    word_found = True
                if not word_found:
                    word = self.proper_encoding(word)
                    if word in self._lexicon:
                        words_lexicon.append(self._lexicon[word])
                        word_found = True
                    if not word_found:
                        word = self._stemmer.stem(word)
                        if word in self._lexicon:
                            words_lexicon.append(self._lexicon[word])
            word_result = self.calculate_value(words_lexicon)
        except Exception as e:
            print("Error predict : ", e)
        return word_result

    def delete_special_characters(self, lin):
        lin = re.sub('\/|\\|\\.|\,|\;|\:|\n|\?|\'|\t', ' ', lin)  # quita los puntos
        lin = re.sub("\s+\w\s+", " ", lin)  # quita los caractores solos
        lin = re.sub("\.", "", lin)
        lin = re.sub(" ", "", lin)
        return lin.lower()

    def delete_stopword(self, tokens):
        return_tokens_valid = []
        for word in tokens:
            if word not in self._stopwords_all and (word != "") and (len(word) > 2):
                return_tokens_valid.append(word)
        return return_tokens_valid

    def proper_encoding(self, text):
        text = unicodedata.normalize('NFD', text)
        text = text.encode('ascii', 'ignore')
        text = text.decode("utf-8")
        return text

    def lemmatization(self, text, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        doc = self.nlp(text)
        texts_out = " ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if
                              token.pos_ in allowed_postags])
        return texts_out


if __name__ == "__main__":
    # file_process = 'lexicon_emotions_en_utf8.csv'
    # cls_emo = ClassificationLexiconBased(file_process)
    # print(cls_emo.class_lexicon)
    # file_process = 'lexicon_polarity_en_utf8.csv'
    # cls_pol = ClassificationLexiconBased(file_process)
    # print(cls_pol.class_lexicon)
    # print(cls_emo.predict('Hi its pretty good.'))
    # print(cls_pol.predict('Hi its pretty good.'))
    file_process = 'lexicon_emotions_es_utf8.csv'
    cls_emo = ClassificationLexiconBased(file_process)
    file_process = 'lexicon_polarity_es_utf8.csv'
    cls_pol = ClassificationLexiconBased(file_process)
    # file_process = 'lexicon_general_interest_es_utf8.csv'
    # cls_int = ClassificationLexiconBased(file_process)
    print(cls_emo.predict("Me gusta la nueva ley de ciencia innovacion y tecnologia, Pero algo anda mal  ? "))
    print(cls_pol.predict("Me gusta la nueva ley de ciencia innovacion y tecnologia, Pero algo anda mal  ? "))
    # print(cls_int.predict("Me gusta la nueva ley de ciencia innovacion y tecnologia, Pero algo anda mal  ? "))