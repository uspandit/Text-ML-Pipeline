import os
import sys
import json
from pathlib import Path
from django.shortcuts import render
from django.http import HttpResponse, Http404
from django.core.files.storage import FileSystemStorage
from .forms import UploadFileForm
from pipeline import Pipeline  # Ensure pipeline is imported correctly

# Ensure the base directory is included in the sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

def upload_file(request):
    if request.method == 'POST':
        # Handle file upload logic here
        return HttpResponse('File uploaded successfully')
    return render(request, 'upload.html')

def download_model(request, model_path):
    file_path = os.path.join(BASE_DIR, 'models', model_path)
    if os.path.exists(file_path):
        with open(file_path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type="application/octet-stream")
            response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file_path)
            return response
    else:
        raise Http404("Model not found")

def results(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = request.FILES['csv_file']
            ngram_range = form.cleaned_data['ngram_range']
            analyzer = form.cleaned_data['analyzer']

            # Handle the uploaded CSV file and form data as needed
            fs = FileSystemStorage()
            filename = fs.save(os.path.join("./data", csv_file.name), csv_file)
            file_path = fs.path(filename)

            # Prepare configuration based on user input
            config = {
                "dataLoader": "CSVLoader",
                "dataPath": file_path,
                "preprocessing": ["fillnan", "lowercase"],
                "preprocessingPath": "./preprocessed/",
                "features": ["tfidf"],
                "featureKwargs": [{"ngram_range": [int(ngram_range), int(ngram_range) + 1], "analyzer": analyzer}],
                "featurePath": "./features/",
                "model": "TensorFlow",
                "modelKwargs": {"model_path": "./models/model.h5"},
                "modelPath": "./models/",
                "metrics": ["precision", "recall"],
                "metricsKwargs": [{"average": "macro"}, {"average": "macro"}],
                "metricPath": "./metrics/",
                "outputPath": "./results/",
                "experimentName": "result.txt"
            }

            config_path = os.path.join(BASE_DIR, 'config.json')
            with open(config_path, 'w') as config_file:
                json.dump(config, config_file)

            # Execute the pipeline
            pipeline = Pipeline(config_path)
            pipeline.execute()

            # Load and display results
            results_path = os.path.join(BASE_DIR, 'results', 'result.txt')
            with open(results_path, 'r') as result_file:
                lines = result_file.readlines()

            config_data = json.loads(lines[0])
            metrics_data = [line.strip().split(",") for line in lines[1:]]

            metrics_dict = {}
            for metric in metrics_data:
                metrics_dict[metric[0]] = float(metric[1])

            model_path = "model.h5"  # Replace with actual model file name or logic

            return render(request, 'results.html', {'config_data': config_data, 'metrics_dict': metrics_dict, 'model_path': model_path})
    
    # If GET request, display the form
    form = UploadFileForm()
    return render(request, 'upload.html', {'form': form})

def index(request):
    form = UploadFileForm()
    return render(request, 'index.html', {'form': form})
