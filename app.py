from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_from_directory, session
import image_utils
from PIL import Image
import re, json, nltk, itertools, spacy, difflib, math
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.corpus import stopwords
from werkzeug.utils import secure_filename
nltk_resources = ["punkt", "maxent_ne_chunker_tab", "words", "averaged_perceptron_tagger", "stopwords"]
import google.generativeai as genai
from dotenv import load_dotenv
import os
import csv
from pathlib import Path
import traceback
from datetime import datetime
import cv2
import easyocr


app = Flask(__name__)

app.secret_key = os.environ.get('FLASK_SECRET_KEY') or os.urandom(24).hex()


@app.route('/')
def home():
    # Moved preview_data before return statement
    preview_data = session.get('preview_data', None)
    return render_template('index1.html', preview_data=preview_data)


for resource in nltk_resources:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource)

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

CSV_PATH = Path("data/finaldata.csv")  # Better path handling

UPLOAD_FOLDER = "uploads"
CSV_FILENAME = "finaldata.csv"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def sanitize_user_id(user_id):
    """Sanitize user ID to prevent directory traversal"""
    return re.sub(r'[^a-zA-Z0-9_-]', '', user_id)



def perform_ocr(image_path):
    """Extract text from image using OCR"""
    try:
        image = Image.open(image_path)
        text_content = image_utils.scan_image_for_text(image)
        combined_text = "\n".join([
            text_content["unmodified"],
            text_content["auto_rotate"],
            text_content["grayscaled"],
            text_content["monochromed"],
            text_content["mean_threshold"],
            text_content["gaussian_threshold"],
            text_content["deskewed_1"],
            text_content["deskewed_2"],
            text_content["deskewed_3"]
        ])
        return combined_text
    except Exception as e:
        return f"OCR Error: {str(e)}"

def get_formatted_text_info(text):
    """Extract structured data using Gemini API"""
    prompt = f"""
    Extract the following details from the text and return JSON:
    - document_type (Aadhaar, PAN, Driving License, Credit Card, Passport)
    - country
    - document_id
    - email
    - phone_no
    - address
    - dob
    - gender
    - expiry_date

    Text: {text}

    Return JSON with only these keys in lowercase. If not found, use empty string.
    """
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        response = model.generate_content(prompt)
        cleaned_response = re.sub(r'```json|```', '', response.text).strip()
        return json.loads(cleaned_response)
    except Exception as e:
        return {"error": str(e)}
    
def save_to_csv(user_id, processed_data):
    """Save extracted data to CSV file with proper error handling"""
    fieldnames = ['user_id', 'document_type', 'country', 'document_id',
                  'email', 'phone_no', 'address', 'dob', 'gender', 'expiry_date']
    
    try:
        # Create parent directories if they don't exist
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare row data with fallback values
        row_data = {
            'user_id': user_id,
            'document_type': processed_data.get('document_type', 'N/A'),
            'country': processed_data.get('country', 'N/A'),
            'document_id': processed_data.get('document_id', 'N/A'),
            'email': processed_data.get('email', 'N/A'),
            'phone_no': processed_data.get('phone_no', 'N/A'),
            'address': processed_data.get('address', 'N/A'),
            'dob': processed_data.get('dob', 'N/A'),
            'gender': processed_data.get('gender', 'N/A'),
            'expiry_date': processed_data.get('expiry_date', 'N/A')
        }

        # Write to CSV
        with open(CSV_PATH, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not CSV_PATH.exists():
                writer.writeheader()
            writer.writerow(row_data)
            
        return True
        
    except Exception as e:
        print(f"CSV Save Error: {str(e)}")
        print(f"Failed data: {row_data}")  # Debugging output
        return False
    
def mask_image(image_path, result_json):
    image = cv2.imread(image_path)
    sensitive_info = [
        str(result_json.get("document_id", "")),
        str(result_json.get("phone_no", "")),
        str(result_json.get("address", "")),
        str(result_json.get("dob", "")),
        str(result_json.get("expiry_date", ""))
    ]
    sensitive_info = [str(info) for info in sensitive_info if info is not None]

    chunks = []
    for info in sensitive_info:
        # Add the whole string
        chunks.append(info)
        # Also add individual parts split by spaces
        parts = info.split()
        for part in parts:
            if len(part) >= 3:  # Only consider chunks of reasonable length
                chunks.append(part)

    reader = easyocr.Reader(['en'])
    ocr_results = reader.readtext(image)

    for (bbox, text, prob) in ocr_results:
        # Check if the OCR text matches or is contained in any sensitive info
        detected_text = text.strip()
        
        # Skip very short text (likely to cause false positives)
        if len(detected_text) < 3:
            continue
            
        for chunk in chunks:
            # Case-insensitive comparison
            if (detected_text.lower() in chunk.lower() or 
                chunk.lower() in detected_text.lower()):
                
                # Get bounding box coordinates
                (top_left, top_right, bottom_right, bottom_left) = bbox
                x_min = int(top_left[0])
                y_min = int(top_left[1])
                x_max = int(bottom_right[0])
                y_max = int(bottom_right[1])
                
                # Apply a blur mask over the detected text
                roi = image[y_min:y_max, x_min:x_max]

                # Apply Gaussian Blur to the ROI
                blurred_roi = cv2.GaussianBlur(roi, (15, 15), 30)

                # Replace the original ROI with the blurred ROI
                image[y_min:y_max, x_min:x_max] = blurred_roi
                break
    
    masked_path = os.path.join(os.path.dirname(image_path), "masked_" + os.path.basename(image_path))
    cv2.imwrite(masked_path, image)
    return masked_path

@app.route('/proceed', methods=['POST', 'GET'])
def handle_proceed():
    try:
        # Add any final processing logic here
        session.pop('preview_data', None)
        flash('Document processing completed', 'success')
        return redirect(url_for('home'))
        
    except Exception as e:
        flash(f'Proceed failed: {str(e)}', 'error')
        return redirect(url_for('home'))

@app.route('/uploads/<user_id>/<filename>')
def serve_file(user_id, filename):
    return send_from_directory(
        os.path.join(app.config['UPLOAD_FOLDER'], user_id),
        filename
    )

@app.route('/documents', methods=['POST', 'GET'])
def view_documents():
    # Get user_id from session or request args
    user_id = session.get('current_user_id')
    if not user_id:
        flash('Please upload a document first to access this page', 'error')
        return redirect(url_for('home'))

    sanitized_id = sanitize_user_id(user_id)
    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], sanitized_id)
    
    # Get list of documents
    documents = []
    if os.path.exists(user_folder):
        documents = [f for f in os.listdir(user_folder) if os.path.isfile(os.path.join(user_folder, f))]
    
    # Pagination logic
    page = request.args.get('page', 1, type=int)
    per_page = 10
    total_pages = (len(documents) + per_page - 1) // per_page
    
    return render_template('documents.html',
                         documents=documents[(page-1)*per_page : page*per_page],
                         current_page=page,
                         total_pages=total_pages,
                         user_id=user_id)

def get_upload_time(user_id, filename):
    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], sanitize_user_id(user_id))
    file_path = os.path.join(user_folder, filename)
    return datetime.fromtimestamp(os.path.getctime(file_path))

@app.route('/download/<user_id>/<filename>')
def download_document(user_id, filename):
    sanitized_id = sanitize_user_id(user_id)
    directory = os.path.join(app.config['UPLOAD_FOLDER'], sanitized_id)
    return send_from_directory(directory, filename, as_attachment=True)

# Add this after get_upload_time function
@app.context_processor
def inject_utilities():
    """Make utility functions available in templates"""
    return {
        'get_upload_time': get_upload_time
    }

# Update existing template filter with error handling
@app.template_filter('formatdatetime')
def format_datetime(value, fmt):
    """Custom datetime formatting filter"""
    if not value:
        return "Unknown"
    try:
        return value.strftime(fmt)
    except AttributeError:
        return "Invalid date"
    
def get_upload_time(user_id, filename):
    """Get file creation time with error handling"""
    try:
        user_folder = os.path.join(app.config['UPLOAD_FOLDER'], sanitize_user_id(user_id))
        file_path = os.path.join(user_folder, filename)
        return datetime.fromtimestamp(os.path.getctime(file_path))
    except Exception as e:
        app.logger.error(f"Error getting upload time: {str(e)}")
        return None

@app.route('/mask', methods=['POST', 'GET'])
def handle_mask():
    try:
        user_id = request.form['user_id']
        filename = request.form['filename']
        
        sanitized_id = sanitize_user_id(user_id)
        user_folder = os.path.join(app.config['UPLOAD_FOLDER'], sanitized_id)
        original_path = os.path.join(user_folder, filename)
        
        # Retrieve processed data from session
        preview_data = session.get('preview_data', {})
        processed_data = preview_data.get('processed_data', {})
        
        # Generate masked version
        masked_path = mask_image(original_path, processed_data)
        masked_filename = os.path.basename(masked_path)
        
        # Update session with masked version
        session['preview_data']['preview_url'] = f'/uploads/{sanitized_id}/{masked_filename}'
        session.modified = True
        
        flash('PII masked successfully', 'success')
        return redirect(url_for('home'))
        
    except Exception as e:
        flash(f'Masking failed: {str(e)}', 'error')
        return redirect(url_for('home'))
    
@app.route('/cancel', methods=['POST', 'GET'])
def cancel_preview():
    session.pop('preview_data', None)
    flash('Preview canceled', 'info')
    return redirect(url_for('home'))

# Add this after get_upload_time function
@app.template_filter('formatdatetime')
def format_datetime(value, format):
    return value.strftime(format)

@app.route('/upload', methods=['POST','GET'])
def handle_upload():
    """Handle file upload and processing"""
    try:

        session.pop('preview_data', None)
        # Validate user ID
        user_id = request.form.get('user_id')
        if not user_id:
            flash('User ID is required', 'error')
            return redirect(url_for('home'))

        # Sanitize inputs
        sanitized_user_id = sanitize_user_id(user_id)
        session['current_user_id'] = sanitized_user_id
        filename = secure_filename(request.files['file1'].filename)  # Now using secure_filename

        # Validate file
        if 'file1' not in request.files:
            flash('No file uploaded', 'error')
            return redirect(url_for('home'))

        file = request.files['file1']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('home'))

        if not allowed_file(file.filename):
            flash('Invalid file type', 'error')
            return redirect(url_for('home'))

        # Create user directory
        user_folder = os.path.join(app.config['UPLOAD_FOLDER'], sanitized_user_id)
        os.makedirs(user_folder, exist_ok=True)  # Creates directory if not exists

        # Save file
        filename = secure_filename(file.filename)
        file_path = os.path.join(user_folder, filename)
        file.save(file_path)

        # Process document
        extracted_text = perform_ocr(file_path)
        if not extracted_text:
            flash('Failed to extract text from document', 'error')
            return redirect(url_for('home'))

        # Get structured data
        processed_data = get_formatted_text_info(extracted_text)
        if 'error' in processed_data:
            flash(f'Data processing error: {processed_data["error"]}', 'error')
            return redirect(url_for('home'))
        
        if (sum(1 for value in processed_data.values() if value not in [None, "", "N/A"]) > 0):
            flash(f'Data Contains Sensitive and Private Information', 'error')

        # Save to CSV
        if save_to_csv(sanitized_user_id, processed_data):
            flash('Document processed and data saved successfully!', 'success')
        else:
            flash('Failed to save document data', 'error')

        # Store processing results for preview
        session['preview_data'] = {
            'user_id': sanitized_user_id,
            'filename': filename,
            'processed_data': processed_data,
            'preview_url': f'/uploads/{sanitized_user_id}/{filename}'
        }

        return redirect(url_for('home'))

    except Exception as e:
        flash(f'Unexpected error: {str(e)}', 'error')
        return redirect(url_for('home'))

if __name__ == '__main__':
    app.config['SESSION_PERMANENT'] = True
    app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour expiration
    app.run(debug=True)

