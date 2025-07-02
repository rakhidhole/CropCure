from . import views
from django.urls import path
from django.contrib.auth import views as auth_views
from django.contrib.auth.views import LogoutView

urlpatterns=[
    path("",views.index,name="index"),
    path('diseases/', views.diseases, name='diseases'),
    path('analyze-image/', views.analyze_image, name='analyze_image'),
    path('signup/', views.signup_view, name='signup'),
    path('login/', auth_views.LoginView.as_view(template_name='login.html'), name='login'),
    path('logout/', LogoutView.as_view(next_page='/'), name='logout'),
    path('history/', views.history_view, name='history'),

]
