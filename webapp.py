import streamlit as st
from omr import mark_file

st.set_page_config(
    page_title="Exam Grader",
    layout="centered",
    initial_sidebar_state="auto",
)

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []


def handle_pdf_processing(answer_key, pdfs_to_grade):
    marked_files = []

    progress_bar = st.progress(0, "Starting grading...")
    status_text = st.empty()

    for i, uploaded_file in enumerate(pdfs_to_grade, start=1):
        status_text.text(f"Grading {uploaded_file.name}...")
        _, _, marked_pdf_file = mark_file(uploaded_file.read(), answer_key.read())
        marked_file_name = f"Graded_{uploaded_file.name}"
        marked_files.append({"name": marked_file_name, "data": marked_pdf_file})

        # Update progress bar
        progress = i / len(pdfs_to_grade)
        progress_bar.progress(progress, f"Graded {i} of {len(pdfs_to_grade)} exams.")

    status_text.text("Grading complete!")
    status_text.empty()
    progress_bar.empty()

    return marked_files


# --- Main Application UI ---
st.title("Automatic Exam Grader")

st.markdown("""
This application automatically grades exams based on a provided answer key. 
Upload the answer key and the exams to be graded, and the application will score them and provide the results.
""")

with st.container(border=True):
    st.subheader("Step 1: Upload Answer Key and Exams")

    answer_key = st.file_uploader(
        "Upload the answer key",
        type="pdf",
        accept_multiple_files=False,
        help="Please upload the answer key for the exam.",
    )

    pdfs_to_grade = st.file_uploader(
        "Upload the exams to be graded",
        type="pdf",
        accept_multiple_files=True,
        help="You can upload one or more exams to be graded here.",
    )

with st.container(border=True):
    st.subheader("Step 2: Grade Exams")

    is_ready_to_process = answer_key is not None and len(pdfs_to_grade) > 0

    if st.button("Grade Exams", type="primary", disabled=not is_ready_to_process):
        with st.spinner("Preparing to grade your exams..."):
            processed_results = handle_pdf_processing(answer_key, pdfs_to_grade)
            st.session_state.processed_files = processed_results
            st.success("Exams have been graded successfully!")

    if not is_ready_to_process:
        st.warning("Please upload both the answer key and the exams to enable grading.")

# --- Download Section ---
if st.session_state.processed_files:
    with st.container(border=True):
        st.subheader("Step 3: Download Your Graded Exams")
        st.info("Click the buttons below to download each graded exam.")

        for file_info in st.session_state.processed_files:
            st.download_button(
                label=f"Download {file_info['name']}",
                data=file_info["data"],
                file_name=file_info["name"],
                mime="application/pdf",
                key=file_info["name"],
            )
