from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import os
import cv2
import image_operations.binary_operations as bo
from image_operations.thinning import zhang_suen
from image_operations.thickening import zhang_suen_thicken

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
img2 = cv2.imread('images/text.png', cv2.IMREAD_GRAYSCALE)

def process_image(file_path):
    img = cv2.imread(file_path)
    processed_img = bo.threshold(img, 127)  # Your image processing function
    return processed_img

def save_processed_image(processed_img, filename):
    processed_filename = 'processed_' + filename
    processed_file_path = os.path.join('static/uploads/', processed_filename)
    cv2.imwrite(processed_file_path, processed_img)
    return processed_filename

def apply_operations(img, operation, img2=img2, threshold_value=127, kernel_size=3, iterations=10):
    binary_img = cv2.threshold(img, threshold_value, 1, cv2.THRESH_BINARY)[1]
    operations = {
        'threshold': lambda: bo.threshold(img, threshold_value),
        'addition': lambda: bo.addition(img, img2),
        'subtraction': lambda: bo.subtraction(img, img2),
        'erosion': lambda: bo.erosion(img, kernel_size),
        'dilation': lambda: bo.dilation(img, kernel_size),
        'opening': lambda: bo.opening(img, kernel_size),
        'closing': lambda: bo.closing(img, kernel_size),
        'thinning': lambda: zhang_suen(binary_img, iterations),
        'thickening': lambda: zhang_suen_thicken(binary_img, iterations),
        'lantuejoul_skeletonization': lambda: bo.lantuejoul_skeletonization(binary_img),
        'homotopic_skeletonization': lambda: bo.homotopic_skeletonization(binary_img),
    }
    return operations.get(operation, lambda: img)()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'reset' in request.form:
            # Handle reset action, redirect back to the image_operations page
            return redirect(url_for('image_operations', filename=filename))
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('image_operations', filename=filename))
    return render_template('uploads.html')

@app.route('/image_operations/<filename>', methods=['GET', 'POST'])
def image_operations(filename):
    if request.method == 'POST':
        selected_operation = request.form['operation']
        threshold_value = int(request.form.get('threshold_value', 127))
        kernel_size = int(request.form.get('kernel_size', 3))
        iterations = int(request.form.get('iterations', 10))
        
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        processed_img = apply_operations(img, selected_operation, threshold_value=threshold_value,
                                         kernel_size=kernel_size, iterations=iterations)
        
        processed_filename = save_processed_image(processed_img, filename)
        return redirect(url_for('image_operations', filename=processed_filename))

    return render_template('image_operations.html', filename=filename)

if __name__ == "__main__":
    app.run(debug=True)
