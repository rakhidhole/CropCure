'''from django.shortcuts import render
from django.http import JsonResponse
from .disease_detection2 import detect_disease
import os
from django.conf import settings
import random



# Create your views here.
def index(request):
    context={}
    return render(request,"cropCure/index.html",context)


def analyze_image(request):
    if 'image' in request.FILES:
        uploaded_image = request.FILES['image']
        print("Image received")  

        
        image_filename = uploaded_image.name

        image_path = os.path.join('C:/Users/rakhi/OneDrive/Desktop/dieasesDetector/dieasesDetector/cropCure/static/inputted_images')
        os.makedirs(image_path, exist_ok=True)

        with open(os.path.join(image_path, image_filename), 'wb') as f:
            for chunk in uploaded_image.chunks():
                f.write(chunk)
        image_newpath = 'cropCure/static/inputted_images/'+image_filename

        # disease_result = detect_disease(image_newpath)
        res = detect_disease(image_newpath)
        print(res)
        # new_res = {
        #     'disease_result': res["disease"], 
        #     'symptoms_result' : res["symptoms"], 
        #     'treatment_result' : res["treatment"],
        # }

        return JsonResponse({'disease_result': res["disease"], 'symptoms_result' : res["symptoms"], 'treatment_result' : res["treatment"]})
        # return JsonResponse({'disease_result': disease_result})
    else:
        return JsonResponse({'error': 'No image uploaded'})
       
'''
import os
import random
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from .disease_detection2 import detect_disease
from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from .models import DetectionHistory
from django.core.files.base import ContentFile
import uuid

def signup_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    else:
        form = UserCreationForm()
    return render(request, 'signup.html', {'form': form})

@login_required
def index(request):
    return render(request, "cropCure/index.html", {})
def diseases(request):
    crop = request.GET.get('crop')  # this will be used in JavaScript
    return render(request, 'cropCure/diseases.html', {'crop': crop})

'''
def analyze_image(request):
    if 'image' in request.FILES:
        uploaded_image = request.FILES['image']
        image_filename = uploaded_image.name

        # Create input image directory if not exists
        input_folder = os.path.join(settings.BASE_DIR, 'cropCure', 'static', 'inputted_images')
        os.makedirs(input_folder, exist_ok=True)

        # Save uploaded image
        image_path = os.path.join(input_folder, image_filename)
        with open(image_path, 'wb') as f:
            for chunk in uploaded_image.chunks():
                f.write(chunk)

        # Predict disease
        result = detect_disease(image_path)
        disease_name = result.get("disease")

        # Ensure disease_name is not None or empty
        if not disease_name:
            return JsonResponse({'error': 'No disease predicted.'})

        # Convert disease name to match folder naming (e.g., "Apple Black rot" -> "Apple___Black_rot")
        formatted_disease_folder = disease_name.strip().replace(" ", "___")

        # Construct full path to the disease image folder
        disease_folder = os.path.join(
            settings.BASE_DIR, 'cropCure', 'static', 'plant_disease_subset', formatted_disease_folder
        )

        image_urls = []

        # Debug print
        print("Looking in folder:", disease_folder)

        if os.path.exists(disease_folder):
            all_images = [f for f in os.listdir(disease_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            if all_images:
                selected_images = random.sample(all_images, min(4, len(all_images)))
                for img in selected_images:
                    # Build static URL for browser access
                    image_url = f"/static/plant_disease_subset/{formatted_disease_folder}/{img}"
                    image_urls.append(image_url)
            else:
                print("No images found in:", disease_folder)
        else:
            print("Disease folder does not exist:", disease_folder)

        return JsonResponse({
            'disease_result': disease_name,
            'symptoms_result': result.get("symptoms"),
            'treatment_result': result.get("treatment"),
            'image_urls': image_urls
        })

    else:
        return JsonResponse({'error': 'No image uploaded'})

        '''
@login_required
def analyze_image(request):
    if 'image' in request.FILES:
        uploaded_image = request.FILES['image']
        image_filename = f"{uuid.uuid4()}_{uploaded_image.name}"

        # Save image to input folder for model prediction
        input_folder = os.path.join(settings.BASE_DIR, 'cropCure', 'static', 'inputted_images')
        os.makedirs(input_folder, exist_ok=True)
        image_path = os.path.join(input_folder, image_filename)

        with open(image_path, 'wb') as f:
            for chunk in uploaded_image.chunks():
                f.write(chunk)

        result = detect_disease(image_path)
        disease_name = result.get("disease")
        confidence = result.get("confidence")
        symptoms = result.get("symptoms")
        treatment = result.get("treatment")

        # Save image in model's ImageField-compatible way
        uploaded_image.seek(0)
        detection = DetectionHistory(
            user=request.user,
            uploaded_image=ContentFile(uploaded_image.read(), image_filename),
            disease_name=disease_name,
            confidence=confidence,
            symptoms=symptoms,
            treatment=treatment
        )
        detection.save()

        # Prepare sample images
        formatted_disease_folder = disease_name.strip().replace(" ", "___")
        disease_folder = os.path.join(settings.BASE_DIR, 'cropCure', 'static', 'plant_disease_subset', formatted_disease_folder)

        image_urls = []
        if os.path.exists(disease_folder):
            all_images = [f for f in os.listdir(disease_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if all_images:
                selected_images = random.sample(all_images, min(4, len(all_images)))
                for img in selected_images:
                    image_urls.append(f"/static/plant_disease_subset/{formatted_disease_folder}/{img}")

        return JsonResponse({
            'disease_result': disease_name,
            'confidence_result': confidence,
            'symptoms_result': symptoms,
            'treatment_result': treatment,
            'image_urls': image_urls
        })

    return JsonResponse({'error': 'No image uploaded'})    

@login_required
def history_view(request):
    history = DetectionHistory.objects.filter(user=request.user).order_by('-detected_at')
    return render(request, 'cropCure/history.html', {'history': history})
