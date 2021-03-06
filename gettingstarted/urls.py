from django.urls import path, include

from django.contrib import admin

admin.autodiscover()

import hello.views

from django.conf.urls.static import static
from django.conf import settings

from django.urls import path, include

# To add a new path, first import the app:
# import blog
#
# Then add the new path:
# path('blog/', blog.urls, name="blog")
#
# Learn more here: https://docs.djangoproject.com/en/2.1/topics/http/urls/

urlpatterns = [
    path("", hello.views.index, name="index"),
    path("admin/", admin.site.urls),
    path('api-auth/', include('rest_framework.urls')),
    path('api/', include('images.api.urls', namespace='api-images'))
]


urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)