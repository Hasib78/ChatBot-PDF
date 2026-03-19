import streamlit as st
from utils.api import upload_pdfs_api, clear_db_api


def render_uploader():
    st.sidebar.header("Upload PDFs")
    uploaded_files = st.sidebar.file_uploader(
        "Upload multiple PDFs", type="pdf", accept_multiple_files=True
    )

    # UPLOAD BUTTON
    if st.sidebar.button("Upload to DB"):
        if uploaded_files:
            with st.sidebar.spinner("Processing & Vectorizing..."):
                response = upload_pdfs_api(uploaded_files)
                if response.status_code == 200:
                    st.sidebar.success("Uploaded successfully!")
                else:
                    st.sidebar.error(f"Error: {response.text}")
        else:
            st.sidebar.warning("Please select at least one PDF.")

    # VISUAL SEPARATOR
    st.sidebar.divider()

    st.sidebar.header("Manage Database")

    # type="primary" makes the button stand out (often red or blue depending on theme)
    if st.sidebar.button("🗑️ Clear Database", type="primary"):
        with st.sidebar.spinner("Deleting files and vectors..."):
            response = clear_db_api()

            if response.status_code == 200:
                # Wipe the chat history on the screen
                st.session_state.messages = []
                st.sidebar.success("Database wiped clean! Ready for new PDFs.")
            else:
                st.sidebar.error("Failed to clear database.")
