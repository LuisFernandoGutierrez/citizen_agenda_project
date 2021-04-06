import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DIR_DATA = "{0}{1}data{1}".format(ROOT, os.sep)
DIR_CONFIG = DIR_CONF = "{0}{1}conf{1}".format(ROOT, os.sep)
DIR_INPUT = "{0}{1}input{1}".format(DIR_DATA, os.sep)
DIR_OUTPUT = "{0}{1}output{1}".format(DIR_DATA, os.sep)
DIR_LEXICON = "{0}{1}lexicon{1}".format(DIR_DATA, os.sep)
DIR_MODELS = "{0}{1}models{1}".format(DIR_DATA, os.sep)
