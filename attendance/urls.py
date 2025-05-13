# urls.py for your attendance app

from django.urls import path
from django.contrib import admin
from . import views
from .admin import admin_site

urlpatterns = [
    path('', views.index, name='index'),
    path('register/', views.register_student, name='register_student'),
    path('attendance/', views.take_attendance, name='take_attendance'),
    path('reports/', views.attendance_reports, name='attendance_reports'),
    path('download-report/', views.download_report_pdf, name='download_report_pdf'),
    path('check-student-id/', views.check_student_id_exists, name='check_student_id_exists'),
    path('check-email/', views.check_email_exists, name='check_email_exists'),
    path('login/', views.user_login, name='login'),
    path('logout/', views.user_logout, name='logout'),
    path('admin/', admin_site.urls),
    path('attendance-admin/', admin_site.urls, name='admin'),
]