import streamlit as st
from pathlib import Path
import main  # 👈 this imports your functions from main.py

st.set_page_config(page_title="Chat With PDFs 🧠", page_icon="📂", layout="wide")

st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #141E30, #243B55);
        color: white;
        font-family: 'Poppins', sans-serif;
    }
    .main { background-color: rgba(0,0,0,0.3); padding: 25px; border-radius: 15px; }
    h1 { text-align: center; font-size: 2.5em; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>📄 Chat With Your PDFs</h1>", unsafe_allow_html=True)
st.write("Select or enter your **PDF folder path**, then chat with your documents below 👇")

# --- Folder Input ---
folder_path = st.text_input("📁 Enter folder path containing PDFs:", placeholder="e.g. D:/Projects/PDFs")

if folder_path:
    path = Path(folder_path)
    if path.exists() and path.is_dir():
        pdf_files = list(path.glob("*.pdf"))
        if pdf_files:
            st.success(f"✅ Found {len(pdf_files)} PDF(s). Ready to chat!")
        else:
            st.warning("No PDF files found in this folder.")
    else:
        st.error("Invalid folder path. Please check again.")

# --- Chat Section ---
st.markdown("---")
st.subheader("💬 Chat Section")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask your question about the PDFs:")

if st.button("Send"):
    if not folder_path:
        st.warning("Please enter a valid folder path first.")
    elif not query.strip():
        st.warning("Enter a question to ask.")
    else:
        # Run your backend logic here:
        with st.spinner("Thinking... 🧠"):
            response = main.query_pdf_folder(folder_path, query)  # 👈 this should be your function in main.py

        st.session_state.chat_history.append(("user", query))
        st.session_state.chat_history.append(("bot", response))

# --- Display Chat History ---
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"🧑‍💻 **You:** {msg}")
    else:
        st.markdown(f"🤖 **AI:** {msg}")
