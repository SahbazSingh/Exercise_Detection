from django.urls import path,include
from gymAIapp import views

urlpatterns = [
    path("",views.Home,name="Home"),
    path("signup",views.signup,name="signup"),
    path('login',views.handlelogin,name="handlelogin"),
    path('logout',views.handleLogout,name="handleLogout"),
    path('contact',views.contact,name="contact"),
    path('about',views.about,name="about"),
    path('bicep',views.bicep,name="bicep"),
    path('process_bicep_curl', views.process_bicep_curl_video, name='process_bicep_curl'),
    path('analysis_results/<int:result_id>/', views.analysis_results_view, name='analysis_results_view'),
    path('squat',views.squat,name="squat"),
    path('process_squat_video', views.process_squat_video, name='process_squat_video'),
    path('squat_analysis_results/<int:result_id>/', views.squat_analysis_results_view, name='squat_analysis_results_view'),
    path('shoulder_lateral',views.shoulder_lateral,name="shoulder_lateral"),
    path('process_shoulder_raise_video', views.process_shoulder_raise_video, name='process_shoulder_raise_video'),
    path('shoulder_lateral_results/<int:result_id>/', views.analysis_results_view_shoulder, name='analysis_results_view_shoulder'),
    path('back_pull_down',views.back_pull_down,name="back_pull_down"),
    path('process_back_pull_down_video', views.process_back_pull_down_video, name='process_back_pull_down_video'),
    path('back_pull_down_results/<int:result_id>/', views.analysis_results_view_back_pull_down, name='analysis_results_view_back_pull_down'),
    path('chest_cable_pull',views.chest_cable_pull,name="chest_cable_pull"),
    path('process_chest_cable_pull_video', views.process_chest_cable_pull_video, name='process_chest_cable_pull_video'),
    path('chest_cable_pull_results/<int:result_id>/', views.analysis_results_view_chest, name='analysis_results_view_chest'),
]
    
