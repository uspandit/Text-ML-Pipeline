from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.urls import reverse
from .forms import UploadFileForm
from .models import UploadedFile
from .pipeline import Pipeline  # Ensure correct import path for your Pipeline class
import os
import pandas as pd
import tensorflow as tf  # Assuming you're using TensorFlow for loading .h5 models
from scipy.sparse import csr_matrix

def handle_uploaded_file(f):
    file_path = os.path.join('uploaded_files', f.name)
    with open(file_path, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    return file_path

def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = UploadedFile(file=request.FILES['file'])
            uploaded_file.save()

            file_path = handle_uploaded_file(request.FILES['file'])
            df = pd.read_csv(file_path)
            comments = df['text'].tolist()

            # Print sample comments for debugging
            print(f"Total comments extracted: {len(comments)}")
            print(f"Sample comments: {comments[:5]}")

            # Initialize your pipeline with config.json
            pipeline = Pipeline('config.json')  # Adjust path as necessary

            # Preprocess comments
            clean_comments = pipeline.preprocessData(comments)

            # Convert any non-string items in clean_comments to strings for consistency
            clean_comments = [comment if isinstance(comment, str) else str(comment) for comment in clean_comments]

            # Print preprocessed comments for debugging
            print(f"Total clean comments: {len(clean_comments)}")
            print(f"Sample clean comments: {clean_comments[:5]}")

            if not clean_comments:
                error_message = "No valid comments found after preprocessing."
                return render(request, 'upload.html', {'form': form, 'error_message': error_message})

            # Extract features
            features = pipeline.extractFeatures(clean_comments)
            features = csr_matrix(features)

            # Create a TensorFlow sparse tensor
            sparse_tensor = tf.sparse.SparseTensor(
                indices=tf.constant(features.indices, dtype=tf.int64),
                values=tf.constant(features.data, dtype=tf.float32),
                dense_shape=tf.constant(features.shape, dtype=tf.int64)
            )

            # Reorder the sparse tensor if necessary (optional)
            reordered_sparse_tensor = tf.sparse.reorder(sparse_tensor)

            # Convert sparse tensor to dense tensor (if needed)
            dense_tensor = tf.sparse.to_dense(reordered_sparse_tensor)


            # Load the trained model
            model_path = 'models/model.h5'  # Adjust this path to your actual model path
            model = tf.keras.models.load_model(model_path)

            # Make predictions
            predictions = model.predict(dense_tensor)
            predictions = (predictions > 0.5).astype(int)

            # Combine original comments with predictions
            results = list(zip(comments, predictions))

            # Display results without saving to database
            return render(request, 'results.html', {'results': results})

    else:
        form = UploadFileForm()

    return render(request, 'upload.html', {'form': form})
