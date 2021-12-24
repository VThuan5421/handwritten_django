from django.apps import AppConfig
from keras.models import load_model
from django.conf import settings
import os

class DigitAppConfig(AppConfig):
    #default_auto_field = 'django.db.models.BigAutoField'
    name = 'digit_app'
    # loading model
    model_path = os.path.join(settings.MODELS_ROOT, 'digitmodel.h5')
    digitmodel = load_model(model_path)

    model_path1 = os.path.join(settings.MODELS_ROOT, 'alphabetmodel.h5')
    alphabetmodel = load_model(model_path1)
    
