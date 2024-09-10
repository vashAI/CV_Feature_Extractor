import streamlit as st
import pdfplumber
from openai import OpenAI
import json

# Function to read and extract text from a PDF file
def read_pdf(file):
    document_text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            document_text += page.extract_text()
    return document_text

# Provide your OpenAI API Key
personal_openai_key = 'YOUR OPENAI KEY'  
client = OpenAI(api_key=personal_openai_key)

# Function to query OpenAI with the extracted text and a specific question
def ask_openai(document_text, question, with_streaming=True):
    response = client.chat.completions.create(
        #model = "gpt-4o-mini",
        model = "gpt-3.5-turbo-0125",
        #model = "gpt-4-turbo",
        #model = "gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in extracting information from CV."},
            {"role": "user", "content": f"Document: {document_text}\n\nQuestion: {question}. Return the answer in JSON format."}
        ],
        n = 1,
        stop = None,
        temperature = 0.7,
        max_tokens = 2000,
        stream = with_streaming
    )
    
    if with_streaming:
        collected_messages = []
        placeholder = st.empty()
        for chunk in response:
            collected_message = chunk.choices[0].delta.content
            if collected_message:
                collected_messages.append(collected_message)
                current_text = ''.join(collected_messages)
                placeholder.markdown(current_text)
        answer = ''.join(collected_messages)
    else:
        answer = response.choices[0].message.content
        st.markdown(answer)
    
    return answer

# Streamlit app
st.title("CV Feature Extraction")

# Upload the PDF file
uploaded_file = st.file_uploader("Upload your CV (PDF format)", type=["pdf"])

if uploaded_file is not None:
    # Displaying the uploaded file's name
    st.write(f"Uploaded file: {uploaded_file.name}")
    
    # Reading the PDF
    document_text = read_pdf(uploaded_file)
    
    # Display the extracted text (optional)
    st.subheader("Extracted Text")
    st.text_area("Text from the CV", document_text, height=200)
    
    
    # Define the question for OpenAI
    question = st.text_input("Enter the question for CV feature extraction", 
                             value="""Extract the following relevant information from the CV and convert it into JSON-format to process it later
- Candidate's skills
- Experience 
- Languages 
- Everything about coding
- Certifications
- Educational background""")
    
    if st.button("Extract Features"):
        if personal_openai_key:
            # Query OpenAI and get the response
            with st.spinner("Extracting information..."):
                response = ask_openai(document_text, question, with_streaming=True)
        else:
            st.error("Please enter your OpenAI API Key.")
