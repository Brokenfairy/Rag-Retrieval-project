import os
import sys
from pathlib import Path

import streamlit as st

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.state_manager import FileStatus, StateManager
from src.pdf_downloads import discover_downloadable_pdfs


st.set_page_config(page_title="Invoice Intelligence System", layout="wide")


@st.cache_resource
def get_pipeline():
    from agents.pipeline_orchestrator import InvoicePipeline

    return InvoicePipeline()


@st.cache_resource
def get_analyst_agent():
    from agents.analyst import AnalystAgent

    return AnalystAgent()


def _render_pdf_downloads_sidebar(state_manager: StateManager) -> None:
    st.header("Download PDFs")
    downloadable = discover_downloadable_pdfs(state_manager.get_all_files() or {})

    if not downloadable:
        st.caption("No converted PDFs available yet.")
        return

    for original_name, pdf_path in downloadable:
        try:
            pdf_bytes = pdf_path.read_bytes()
        except OSError:
            continue

        st.download_button(
            label=f"Download {pdf_path.name}",
            data=pdf_bytes,
            file_name=pdf_path.name,
            mime="application/pdf",
            key=f"download-{original_name}-{pdf_path.name}",
            use_container_width=True,
        )


def _render_status_sidebar(state_manager: StateManager) -> None:
    with st.sidebar:
        st.header("Processing Status")
        if st.button("Refresh Status"):
            st.rerun()

        files = state_manager.get_all_files() or {}
        ordered_files = sorted(
            files.items(),
            key=lambda item: item[1].get("last_updated", ""),
            reverse=True,
        )

        if not ordered_files:
            st.info("No files processed yet.")
            return

        for filename, file_data in ordered_files:
            status = file_data.get("status")
            st.subheader(filename)

            if status == FileStatus.PENDING.value:
                st.warning(f"Status: {status}")
                st.progress(25)
            elif status == FileStatus.CONVERTED.value:
                st.info(f"Status: {status}")
                st.progress(50)
            elif status == FileStatus.PARSED.value:
                st.info(f"Status: {status}")
                st.progress(75)
            elif status == FileStatus.INDEXED.value:
                st.success(f"Status: {status} - Ready")
                st.progress(100)
            elif status == FileStatus.FAILED.value:
                st.error(f"Failed: {file_data.get('details', {}).get('error', 'Unknown error')}")
            else:
                st.info(f"Status: {status}")

            st.divider()

        _render_pdf_downloads_sidebar(state_manager)


def _handle_uploads(state_manager: StateManager) -> None:
    st.header("Upload Invoices")
    uploaded_files = st.file_uploader(
        "Upload invoice files (.msg or .pdf)",
        type=["msg", "pdf"],
        accept_multiple_files=True,
    )

    if st.button("Process Uploaded Files"):
        if not uploaded_files:
            st.warning("Please upload at least one file first.")
            return

        pipeline = get_pipeline()
        data_dir = Path("data")
        data_dir.mkdir(parents=True, exist_ok=True)

        for uploaded_file in uploaded_files:
            safe_name = Path(uploaded_file.name).name
            destination = data_dir / safe_name
            destination.write_bytes(uploaded_file.getbuffer())

            state_manager.add_file(safe_name)
            pipeline.process_in_background(safe_name)
            st.toast(f"Started processing {safe_name}")

        st.rerun()


def _handle_chat() -> None:
    st.header("Chat with your Invoices")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask a question about your invoices...")
    if not prompt:
        return

    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Analyzing invoices..."):
            try:
                response = get_analyst_agent().get_response(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {str(e)}")


def main():
    st.title("Invoice Intelligence System")
    state_manager = StateManager()
    _render_status_sidebar(state_manager)
    _handle_uploads(state_manager)
    _handle_chat()


if __name__ == "__main__":
    main()
