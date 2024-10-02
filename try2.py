import os
import re
import streamlit as st
import PyPDF2
import numpy as np
from google.cloud import vision
from fpdf import FPDF
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set the Google Cloud Vision API credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'vision_key.json'

# Function to detect text in an image using Google Cloud Vision API
def detect_text(image_path):
    """Detects text in the file using Google Cloud Vision API."""
    client = vision.ImageAnnotatorClient()
    
    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    
    response = client.document_text_detection(image=image)
    texts = response.text_annotations
    ocr_text = ""

    if texts:
        # Extract the full text from the first annotation
        ocr_text = texts[0].description

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )

    return ocr_text

# Function to format the detected text
def format_text(ocr_text):
    """Formats the detected text by breaking it into lines based on full stops and newlines."""
    formatted_lines = re.split(r'(?<=[.])\s+', ocr_text.strip())
    formatted_lines = [line.replace('\n', ' ').strip() for line in formatted_lines if line.strip()]
    return formatted_lines

# Function to convert formatted text to PDF
def convert_text_to_pdf(text_lines, output_pdf_path, font_path):
    """Converts the detected text to a PDF file with Unicode support."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Add a Unicode-compliant font (DejaVuSans)
    pdf.add_font("DejaVu", "", font_path, uni=True)
    pdf.set_font("DejaVu", size=12)

    # Add each formatted line to the PDF
    for line in text_lines:
        pdf.multi_cell(0, 10, line)

    pdf.output(output_pdf_path)
    print(f"PDF saved as {output_pdf_path}")

# Function to read content from the PDF
def read_pdf_content(file_path):
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text.strip()

# Function to calculate the cosine similarity score
def calculate_similarity(text1, text2):
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    cos_sim = cosine_similarity(vectors)
    return cos_sim[0][1] * 100  # Return as percentage

# Streamlit application
st.title("Document Processing Application")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select Page", ["Image to PDF Conversion", "PDF Comparison"])

# Image to PDF Conversion Page
if page == "Image to PDF Conversion":
    st.header("Image to PDF Conversion")
    image_upload = st.file_uploader("Upload Image for Text Detection", type=["jpg", "jpeg", "png"])

    if image_upload is not None:
        image_path = "uploaded_image." + image_upload.name.split('.')[-1]
        with open(image_path, "wb") as f:
            f.write(image_upload.read())
        
        # Detect text from the image
        raw_text = detect_text(image_path)
        
        # Format the detected text into proper lines
        formatted_text = format_text(raw_text)
        
        # Path to save the output PDF
        output_pdf_path = "output_text.pdf"
        
        # Path to the Unicode font
        font_path = r"C:\Users\Niyati\Desktop\PBL_Major\DejaVuSans.ttf"  # Adjust this path to where your font file is located
        
        # Convert the formatted text to PDF
        convert_text_to_pdf(formatted_text, output_pdf_path, font_path)
        
        st.success("Text detected and PDF created!")

# PDF Comparison Page
elif page == "PDF Comparison":
    st.header("PDF Comparison")
    st.write("Upload the generated PDF and the model answer PDF to compare their contents.")

    # File uploader for the generated PDF
    uploaded_pdf = st.file_uploader("Upload Output PDF", type=["pdf"], key="output_pdf")

    # File uploader for the model answer PDF
    model_answer_pdf = st.file_uploader("Upload Model Answer PDF", type=["pdf"], key="model_pdf")

    # Input for maximum marks
    max_marks = st.number_input("Enter Maximum Marks", min_value=1, step=1)

    if uploaded_pdf and model_answer_pdf and max_marks > 0:
        # Save the uploaded PDFs
        output_pdf_path = "uploaded_output.pdf"
        model_pdf_path = "uploaded_model.pdf"

        # Write the uploaded PDFs to the disk
        with open(output_pdf_path, "wb") as f:
            f.write(uploaded_pdf.read())

        with open(model_pdf_path, "wb") as f:
            f.write(model_answer_pdf.read())

        # Read the content of the uploaded PDFs
        pdf_content = read_pdf_content(output_pdf_path)
        model_answer_content = read_pdf_content(model_pdf_path)

        st.subheader("Uploaded Output PDF Content")
        st.write(pdf_content)

        st.subheader("Uploaded Model Answer PDF Content")
        st.write(model_answer_content)

        # Calculate similarity score
        if st.button("Evaluate Similarity"):
            similarity_score = calculate_similarity(pdf_content, model_answer_content)
            marks_obtained = (similarity_score / 100) * max_marks  # Calculate marks based on similarity percentage
            st.success(f"Similarity Score: {similarity_score:.2f}%")
            st.success(f"Marks Obtained: {marks_obtained:.2f}/{max_marks}")
