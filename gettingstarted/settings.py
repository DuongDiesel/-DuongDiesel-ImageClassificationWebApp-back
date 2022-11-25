"""
Django settings for gettingstarted project.

Generated by 'django-admin startproject' using Django 2.0.

For more information on this file, see
https://docs.djangoproject.com/en/2.0/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/2.0/ref/settings/
"""

import os
import django_heroku


# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/2.0/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.environ.get("SECRET_KEY", 'abce')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = False

ALLOWED_HOSTS = ["whatimage-django-back.herokuapp.com"]


# Application definition

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "hello",

    "images",

    "rest_framework",
    "corsheaders"
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    
    
]

CORS_ORIGIN_ALLOW_ALL = False

CORS_ORIGIN_WHITELIST = [
    "https://serene-dusk-18231.herokuapp.com",
    "https://serene-dusk-18231.herokuapp.com/",
    "https://ieee-post-image-frontend-vercel.vercel.app/",
    "https://ieee-post-image-frontend-vercel.vercel.app"
]



ROOT_URLCONF = "gettingstarted.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ]
        },
    }
]

WSGI_APPLICATION = "gettingstarted.wsgi.application"


# Database
# https://docs.djangoproject.com/en/2.0/ref/settings/#databases

DATABASES = {
    #"default": {
    #    "ENGINE" : "django.db.backends.sqlite3",
    #    "NAME": os.path.join(BASE_DIR, "db.sqlite3")
    #}
    "default": {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': os.environ.get("NAME", 'dev default value'),
        'USER': os.environ.get("USER", 'dev default value'),
        'PASSWORD': os.environ.get("PASSWORD", 'dev default value'),
        'HOST': os.environ.get("HOST", 'dev default value'),
        'PORT':  os.environ.get("PORT", 'dev default value')
    }
}

# Password validation
# https://docs.djangoproject.com/en/2.0/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"
    },
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]


# Internationalization
# https://docs.djangoproject.com/en/2.0/topics/i18n/

LANGUAGE_CODE = "en-us"

TIME_ZONE = "UTC"

USE_I18N = True

USE_L10N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/2.0/howto/static-files/

STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
STATIC_URL = "/static/"

MEDIA_ROOT = os.path.join(BASE_DIR, "media_root")
MEDIA_URL = "/media/"

django_heroku.settings(locals())


# Download Model
import urllib.request
# Define the remote file to retrieve
#remote_url = 'https://github.com/DuongDiesel/DuongDiesel_Store_models/raw/main/malaria_augmented_model.h5'
remote_url = 'https://github.com/DuongDiesel/DuongDiesel_Store_models/raw/main/v4_v4_weighted_121_Best_model.pth'
# Define the local filename to save data
#local_file = 'malaria_augmented_model.h5'
local_file = 'v4_v4_weighted_121_Best_model.pth'
# Download remote and save locally
urllib.request.urlretrieve(remote_url, local_file)