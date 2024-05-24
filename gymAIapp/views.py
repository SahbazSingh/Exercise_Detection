from django.shortcuts import render,redirect
from django.contrib import messages
from django.contrib.auth.models import User
from django.contrib.auth import authenticate,login,logout
from gymAIapp.models import Contact
#from django.http import HttpResponse
from .models_ml.main import exercise_detection
from .models_ml.main import load_machine_learning_models
from django.core.files.storage import FileSystemStorage
from .models import BicepCurlResult
from .models import SquatResult
from .models import ShoulderLateralRaiseResult
from .models import BackPullDownResult
from .models import ChestCablePullResult
from django.utils import timezone
from django.conf import settings
import json
#import logging

#logger = logging.getLogger(__name__)

def Home(request):
    exercises = [
        {'name': 'Arms', 'url': '/bicep', 'icon': 'bi-bicep-icon-class'},
        {'name': 'Shoulders', 'url': '/shoulder_lateral', 'icon': 'bi-shoulder-icon-class'},
        {'name': 'Chest', 'url': '/chest_cable_pull', 'icon': 'bi-chest-icon-class'},
        {'name': 'Back', 'url': '/back_pull_down', 'icon': 'bi-back-icon-class'},
        {'name': 'Legs', 'url': '/squat', 'icon': 'bi-legs-icon-class'}
    ]
    return render(request, "index.html", {'exercises': exercises})

def about(request):
    return render(request,"about.html")


def bicep(request):
    return render(request,"bicep.html")

def squat(request):
    return render(request,"squat.html")

def shoulder_lateral(request):
    return render(request,"shoulder_lateral.html")

def back_pull_down(request):
    return render(request,"back_pull_down.html")

def chest_cable_pull(request):
    return render(request,"chest_cable_pull.html")

def process_bicep_curl_video(request):
    if request.method == 'POST':
        video_file = request.FILES.get('videoFile')
        if not video_file:
            messages.error(request, "No video file provided.")
            return redirect('bicep')

        fs = FileSystemStorage()
        filename = fs.save(video_file.name, video_file)
        video_path = fs.path(filename)

        try:
            load_machine_learning_models()
            # Call the exercise_detection function, expecting a dictionary in response
            analysis_results = exercise_detection(video_file_path=video_path, video_name_to_save="processed_" + filename, exercise_type="bicep_curl")

            if analysis_results:
                # Directly access the 'error_frames_info' from the dictionary
                error_frames_info = analysis_results.get('error_frames_info', [])
                form_errors = json.dumps(error_frames_info)

                # Create a BicepCurlResult instance with the information obtained
                result = BicepCurlResult(
                    analysis_time=timezone.now(),
                    total_curls=analysis_results.get('total_curls', 0),  # Access 'total_curls' directly from the dictionary
                    form_errors=form_errors,
                    feedback=analysis_results.get('feedback', ''),
                    processed_video_filename="processed_" + filename
                )
                result.save()
                print(f"Saved result ID: {result.id}")

                # Redirect to the analysis results view with the ID of the newly saved result
                return redirect('analysis_results_view', result_id=result.id)
            else:
                messages.error(request, "Analysis failed or returned no data.")
                return redirect('bicep')
        except Exception as e:
            messages.error(request, f"Processing failed: {str(e)}")
            return redirect('bicep')
    else:
        return redirect('bicep')



def analysis_results_view(request, result_id):
    try:
        result = BicepCurlResult.objects.get(pk=result_id)
        errors = json.loads(result.form_errors)  # Assuming this returns a JSON string of errors

        # Add frame_filename to each error dictionary
        for error in errors:
            error['frame_filename'] = settings.MEDIA_URL + error.get('frame', 'default.jpg')

        context = {
            'result': result,
            'errors': errors,
            'video_url': settings.MEDIA_URL + result.processed_video_filename,
        }
        return render(request, 'analysis_result.html', context)
    except BicepCurlResult.DoesNotExist:
        messages.error(request, "Result not found.")
        return redirect('Home')
    
def process_squat_video(request):
    if request.method == 'POST':
        video_file = request.FILES.get('videoFile')
        if not video_file:
            messages.error(request, "No video file provided.")
            return redirect('squat')
        
        fs = FileSystemStorage()
        filename = fs.save(video_file.name, video_file)
        video_path = fs.path(filename)  # Using .path to get the absolute filesystem path

        try:
            analysis_results = exercise_detection(video_file_path=video_path, video_name_to_save="processed_" + filename, exercise_type="squat")
            form_errors = json.dumps(analysis_results.get('error_frames_info', []))

            result = SquatResult(
                analysis_time=timezone.now(),
                total_squats=analysis_results.get('total_squats', 0),
                form_errors=form_errors,
                feedback=analysis_results.get('feedback', ''),
                processed_video_filename="processed_" + filename
            )
            result.save()

            return redirect('squat_analysis_results_view', result_id=result.id)
        except Exception as e:
            messages.error(request, f"Processing failed: {e}")
            return redirect('squat')
    else:
        return redirect('squat')


def squat_analysis_results_view(request, result_id):
    try:
        result = SquatResult.objects.get(pk=result_id)
        # Deserialize JSON string back into Python list of dictionaries
        errors = json.loads(result.form_errors)

        for error in errors:
            error['frame_filename'] = settings.MEDIA_URL + error.get('frame', 'default.jpg')

        context = {
            'result': result,
            'errors': errors,
            'video_url': settings.MEDIA_URL + result.processed_video_filename,
        }
        return render(request, 'squat_analysis_result.html', context)
    except SquatResult.DoesNotExist:
        messages.error(request, "Result not found.")
        return redirect('Home') 

def process_shoulder_raise_video(request):
    if request.method == 'POST':
        video_file = request.FILES.get('videoFile')
        if not video_file:
            messages.error(request, "No video file provided.")
            return redirect('shoulder_lateral')  

        fs = FileSystemStorage()
        filename = fs.save(video_file.name, video_file)
        video_path = fs.path(filename)

        try:
            load_machine_learning_models()

            analysis_results = exercise_detection(video_file_path=video_path, video_name_to_save="processed_" + filename, exercise_type="shoulder_lateral_raise")

            if analysis_results:
                error_frames_info = analysis_results.get('error_frames_info', [])
                form_errors = json.dumps(error_frames_info)

                result = ShoulderLateralRaiseResult(
                    analysis_time=timezone.now(),
                    total_reps=analysis_results.get('total_reps', 0),
                    form_errors=form_errors,
                    feedback=analysis_results.get('feedback', ''),
                    processed_video_filename="processed_" + filename
                )
                result.save()
                print(f"Saved result ID: {result.id}")

                return redirect('analysis_results_view_shoulder', result_id=result.id)
            else:
                messages.error(request, "Analysis failed or returned no data.")
                return redirect('shoulder_lateral')
        except Exception as e:
            messages.error(request, f"Processing failed: {str(e)}")
            return redirect('shoulder_lateral')
    else:
        return redirect('shoulder_lateral')


def analysis_results_view_shoulder(request, result_id):
    try:
        result = ShoulderLateralRaiseResult.objects.get(pk=result_id)
        errors = json.loads(result.form_errors)

        for error in errors:
            error['frame_filename'] = settings.MEDIA_URL + error.get('frame', 'default.jpg')

        context = {
            'result': result,
            'errors': errors,
            'video_url': settings.MEDIA_URL + result.processed_video_filename,
        }
        return render(request, 'shoulder_lateral_results.html', context)
    except ShoulderLateralRaiseResult.DoesNotExist:
        messages.error(request, "Result not found.")
        return redirect('Home')
    
def process_back_pull_down_video(request):
    if request.method == 'POST':
        video_file = request.FILES.get('videoFile')
        if not video_file:
            messages.error(request, "No video file provided.")
            return redirect('back_pull_down')  

        fs = FileSystemStorage()
        filename = fs.save(video_file.name, video_file)
        video_path = fs.path(filename)

        try:
            # Load your ML models 
            analysis_results = exercise_detection(video_file_path=video_path, video_name_to_save="processed_" + filename, exercise_type="back_pull_down")

            if analysis_results:
                error_frames_info = analysis_results.get('error_frames_info', [])
                form_errors = json.dumps(error_frames_info)

                result = BackPullDownResult(
                    analysis_time=timezone.now(),
                    total_reps=analysis_results.get('total_reps', 0),
                    form_errors=form_errors,
                    feedback=analysis_results.get('feedback', ''),
                    processed_video_filename="processed_" + filename
                )
                result.save()

                return redirect('analysis_results_view_back_pull_down', result_id=result.id)
            else:
                messages.error(request, "Analysis failed or returned no data.")
                return redirect('back_pull_down')
        except Exception as e:
            messages.error(request, f"Processing failed: {str(e)}")
            return redirect('back_pull_down')
    else:
        return redirect('back_pull_down')

def analysis_results_view_back_pull_down(request, result_id):
    try:
        result = BackPullDownResult.objects.get(pk=result_id)
        errors = json.loads(result.form_errors)

        for error in errors:
            error['frame_filename'] = settings.MEDIA_URL + error.get('frame', 'default.jpg')

        context = {
            'result': result,
            'errors': errors,
            'video_url': settings.MEDIA_URL + result.processed_video_filename,
        }
        return render(request, 'back_pull_down_results.html', context)
    except BackPullDownResult.DoesNotExist:
        messages.error(request, "Result not found.")
        return redirect('Home')  
    
def process_chest_cable_pull_video(request):
    if request.method == 'POST':
        video_file = request.FILES.get('videoFile')
        if not video_file:
            messages.error(request, "No video file provided.")
            return redirect('chest_cable_pull')  

        fs = FileSystemStorage()
        filename = fs.save(video_file.name, video_file)
        video_path = fs.path(filename)

        try:
            # Load machine learning models and other necessary functions
            load_machine_learning_models()
            
            analysis_results = exercise_detection(video_file_path=video_path, video_name_to_save="processed_" + filename, exercise_type="chest_cable_pull")

            if analysis_results:
                error_frames_info = analysis_results.get('error_frames_info', [])
                form_errors = json.dumps(error_frames_info)

                result = ChestCablePullResult(
                    analysis_time=timezone.now(),
                    total_reps=analysis_results.get('total_reps', 0),
                    form_errors=form_errors,
                    feedback=analysis_results.get('feedback', ''),
                    processed_video_filename="processed_" + filename
                )
                result.save()
                print(f"Saved result ID: {result.id}")

                return redirect('analysis_results_view_chest', result_id=result.id)
            else:
                messages.error(request, "Analysis failed or returned no data.")
                return redirect('chest_cable_pull')
        except Exception as e:
            messages.error(request, f"Processing failed: {str(e)}")
            return redirect('chest_cable_pull')
    else:
        return redirect('chest_cable_pull')


def analysis_results_view_chest(request, result_id):
    try:
        result = ChestCablePullResult.objects.get(pk=result_id)
        errors = json.loads(result.form_errors)

        for error in errors:
            error['frame_filename'] = settings.MEDIA_URL + error.get('frame', 'default.jpg')

        context = {
            'result': result,
            'errors': errors,
            'video_url': settings.MEDIA_URL + result.processed_video_filename,
        }
        return render(request, 'chest_cable_pull_results.html', context)
    except ChestCablePullResult.DoesNotExist:
        messages.error(request, "Result not found.")
        return redirect('Home')  


def signup(request):
    if request.method=="POST":
        username=request.POST.get('usernumber')
        email=request.POST.get('email')
        pass1=request.POST.get('pass1')
        pass2=request.POST.get('pass2')
      
        if len(username)>11 or len(username)<11:
            messages.info(request,"Phone Number Must be 11 Digits")
            return redirect('/signup')

        if pass1!=pass2:
            messages.info(request,"Password is not Matching")
            return redirect('/signup')
       
        try:
            if User.objects.get(username=username):
                messages.warning(request,"Phone Number is Taken")
                return redirect('/signup')
           
        except Exception as identifier:
            pass
        
        
        try:
            if User.objects.get(email=email):
                messages.warning(request,"Email is Taken")
                return redirect('/signup')
           
        except Exception as identifier:
            pass
        
        
        
        myuser=User.objects.create_user(username,email,pass1)
        myuser.save()
        messages.success(request,"User is Created Please Login")
        return redirect('/login')
            
    return render(request,"signup.html")

def handlelogin(request):
    if request.method=="POST":        
        username=request.POST.get('usernumber')
        pass1=request.POST.get('pass1')
        myuser=authenticate(username=username,password=pass1)
        if myuser is not None:
            login(request,myuser)
            messages.success(request,"Login Successful")
            return redirect('/')
        else:
            messages.error(request,"Invalid Credentials")
            return redirect('/login')
            
        
    return render(request,"handlelogin.html")

def handleLogout(request):
    logout(request)
    messages.success(request,"Logout Success")    
    return redirect('/login')

def contact(request):
    if request.method=="POST":
        name=request.POST.get('fullname')
        email=request.POST.get('email')
        number=request.POST.get('num')
        desc=request.POST.get('desc')
        myquery=Contact(name=name,email=email,phonenumber=number,description=desc)
        myquery.save()       
        messages.info(request,"Thanks for Contacting us we will get back you soon")
        return redirect('/contact')
        
    return render(request,"contact.html")










