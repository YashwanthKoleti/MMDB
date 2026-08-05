import streamlit as st
import requests
import json
from io import BytesIO
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Multi Media Database",
    page_icon="🗄️",
    layout="centered"
)

# Constants
API_URL_BASE = "http://127.0.0.1:8000/v1/ingestion"

st.title("🗄️ Multi Media Database")
st.markdown("Upload your media here to run OCR, transcriptions, embeddings, and store them in LanceDB.")

tab1, tab2, tab3, tab4 = st.tabs(["Image Ingestion", "Audio Ingestion", "Video Ingestion", "Document Ingestion"])

with tab1:
    st.header("Image Upload")
    image_files = st.file_uploader("Choose image files (supports multiple)", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
    
    if st.button("Process Images", key="img_btn"):
        if image_files:
            for img_file in image_files:
                with st.spinner(f"Processing image {img_file.name} (running OCR and extracting embeddings)..."):
                    try:
                        files = {"file": (img_file.name, img_file.getvalue(), img_file.type)}
                        response = requests.post(f"{API_URL_BASE}/image", files=files)
                        
                        if response.status_code in [200, 202]:
                            st.success(f"Image {img_file.name} successfully uploaded and queued!")
                            st.json(response.json())
                        else:
                            st.error(f"Error {response.status_code} for {img_file.name}: {response.text}")
                    except Exception as e:
                        st.error(f"Request failed for {img_file.name}: {str(e)}")
        else:
            st.warning("Please upload at least one image first.")

with tab2:
    st.header("Audio Upload")
    audio_files = st.file_uploader("Choose audio files (supports multiple)", type=["wav", "mp3", "ogg", "flac"], accept_multiple_files=True)
    
    if st.button("Process Audio", key="audio_btn"):
        if audio_files:
            for aud_file in audio_files:
                with st.spinner(f"Processing audio {aud_file.name} (running transcription and extracting chunks)..."):
                    try:
                        files = {"file": (aud_file.name, aud_file.getvalue(), aud_file.type)}
                        response = requests.post(f"{API_URL_BASE}/audio", files=files)
                        
                        if response.status_code in [200, 202]:
                            st.success(f"Audio {aud_file.name} successfully uploaded and queued!")
                            st.json(response.json())
                        else:
                            st.error(f"Error {response.status_code} for {aud_file.name}: {response.text}")
                    except Exception as e:
                        st.error(f"Request failed for {aud_file.name}: {str(e)}")
        else:
            st.warning("Please upload at least one audio file first.")

with tab3:
    st.header("Video Upload")
    video_files = st.file_uploader("Choose video files (supports multiple)", type=["mp4", "avi", "mov", "mkv", "webm"], accept_multiple_files=True)
    
    if st.button("Process Video", key="video_btn"):
        if video_files:
            for vid_file in video_files:
                with st.spinner(f"Processing video {vid_file.name} (running transcription and extracting embeddings)..."):
                    try:
                        files = {"file": (vid_file.name, vid_file.getvalue(), vid_file.type)}
                        response = requests.post(f"{API_URL_BASE}/video", files=files)
                        
                        if response.status_code in [200, 202]:
                            st.success(f"Video {vid_file.name} successfully uploaded and queued!")
                            st.json(response.json())
                        else:
                            st.error(f"Error {response.status_code} for {vid_file.name}: {response.text}")
                    except Exception as e:
                        st.error(f"Request failed for {vid_file.name}: {str(e)}")
        else:
            st.warning("Please upload at least one video file first.")

with tab4:
    st.header("Document Ingestion")
    
    # PDF / TXT upload
    st.subheader("File Upload (PDF/TXT)")
    doc_files = st.file_uploader("Choose PDF or TXT files (supports multiple)", type=["pdf", "txt"], accept_multiple_files=True)
    
    if st.button("Process Documents", key="doc_btn"):
        if doc_files:
            for doc_file in doc_files:
                with st.spinner(f"Processing document {doc_file.name} (parsing text, rendering pages, extracting visual features, OCR, and embedding)..."):
                    try:
                        files = {"file": (doc_file.name, doc_file.getvalue(), doc_file.type)}
                        response = requests.post(f"{API_URL_BASE}/document", files=files)
                        
                        if response.status_code in [200, 202]:
                            st.success(f"Document {doc_file.name} successfully uploaded and queued!")
                            st.json(response.json())
                        else:
                            st.error(f"Error {response.status_code} for {doc_file.name}: {response.text}")
                    except Exception as e:
                        st.error(f"Request failed for {doc_file.name}: {str(e)}")
        else:
            st.warning("Please upload at least one document file first.")

    st.markdown("---")
    
    # Clipboard / Text Paste
    st.subheader("Clipboard / Paste Text Directly")
    pasted_title = st.text_input("Enter a name/title for this text (optional)", value="pasted_text")
    pasted_text = st.text_area("Paste text content here...", height=200)
    
    if st.button("Process Pasted Text", key="paste_btn"):
        if pasted_text.strip():
            with st.spinner("Processing pasted text (chunking and embedding)..."):
                try:
                    # Package pasted text as a virtual txt file
                    virtual_file = BytesIO(pasted_text.encode("utf-8"))
                    filename = pasted_title if pasted_title.endswith(".txt") else f"{pasted_title}.txt"
                    files = {"file": (filename, virtual_file, "text/plain")}
                    
                    response = requests.post(f"{API_URL_BASE}/document", files=files)
                    
                    if response.status_code in [200, 202]:
                        st.success("Pasted text successfully processed and queued!")
                        st.json(response.json())
                    else:
                        st.error(f"Error {response.status_code}: {response.text}")
                except Exception as e:
                    st.error(f"Request failed: {str(e)}")
        else:
            st.warning("Please paste some text first.")

    st.markdown("---")

    # Global Clipboard File/Media Paste Zone
    st.subheader("📋 Global Clipboard File / Media Ingestion")
    st.markdown("Select the box below and press **Ctrl+V** or **Cmd+V** to ingest images, audio clips, videos, or raw text directly from your clipboard.")

    components.html("""
    <div id="paste-zone" style="border: 2px dashed #0080FF; padding: 30px; text-align: center; background: linear-gradient(135deg, #1e293b, #0f172a); color: #e2e8f0; border-radius: 12px; cursor: pointer; transition: all 0.3s ease; outline: none; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
        <div style="font-size: 32px; margin-bottom: 10px;">📋</div>
        <h3 style="margin: 0 0 8px 0; font-size: 18px; color: #38bdf8;">Click to Activate Paste Zone</h3>
        <p style="margin: 0 0 15px 0; font-size: 14px; color: #94a3b8;">Once active, paste any copied file or text contents (Ctrl+V / Cmd+V)</p>
        <div id="status" style="font-size: 14px; font-weight: 500; color: #38bdf8;">Click inside this container to begin...</div>
    </div>

    <script>
    const pasteZone = document.getElementById('paste-zone');
    const statusDiv = document.getElementById('status');

    pasteZone.setAttribute('tabindex', '0');

    pasteZone.addEventListener('focus', () => {
        pasteZone.style.border = '2px solid #38bdf8';
        pasteZone.style.boxShadow = '0 0 12px rgba(56, 189, 248, 0.4)';
        statusDiv.innerText = "Active: Ready to paste content!";
        statusDiv.style.color = '#38bdf8';
    });

    pasteZone.addEventListener('blur', () => {
        pasteZone.style.border = '2px dashed #0080FF';
        pasteZone.style.boxShadow = 'none';
        statusDiv.innerText = "Inactive. Click to reactivate.";
        statusDiv.style.color = '#94a3b8';
    });

    pasteZone.addEventListener('paste', async (event) => {
        const items = (event.clipboardData || event.originalEvent.clipboardData).items;
        statusDiv.innerText = "Analyzing clipboard content...";
        statusDiv.style.color = '#38bdf8';
        
        let found = false;
        for (let i = 0; i < items.length; i++) {
            const item = items[i];
            
            if (item.kind === 'file') {
                found = true;
                const file = item.getAsFile();
                statusDiv.innerText = `Detected file: ${file.name || 'clipboard_file'} (${file.type})`;
                
                let endpoint = '';
                if (file.type.startsWith('image/')) {
                    endpoint = '/v1/ingestion/image';
                } else if (file.type.startsWith('audio/')) {
                    endpoint = '/v1/ingestion/audio';
                } else if (file.type.startsWith('video/')) {
                    endpoint = '/v1/ingestion/video';
                } else if (file.type.startsWith('application/pdf') || file.type.startsWith('text/')) {
                    endpoint = '/v1/ingestion/document';
                } else {
                    statusDiv.innerHTML = `<span style="color: #ef4444;">Unsupported file type: ${file.type}</span>`;
                    continue;
                }
                
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    statusDiv.innerText = `Queuing upload of ${file.name || 'file'}...`;
                    const response = await fetch('http://127.0.0.1:8000' + endpoint, {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        statusDiv.innerHTML = `<span style="color: #4ade80;">✔ Uploaded successfully!<br>Task ID: ${result.task_id.substring(0,8)}... (${result.status})</span>`;
                    } else {
                        const err = await response.text();
                        statusDiv.innerHTML = `<span style="color: #ef4444;">❌ Error ${response.status}: ${err.substring(0,60)}</span>`;
                    }
                } catch (e) {
                    statusDiv.innerHTML = `<span style="color: #ef4444;">❌ Request failed: ${e.message}</span>`;
                }
            } else if (item.kind === 'string' && item.type === 'text/plain') {
                found = true;
                item.getAsString(async (text) => {
                    statusDiv.innerText = "Ingesting text clip...";
                    
                    const blob = new Blob([text], { type: 'text/plain' });
                    const file = new File([blob], 'clipboard_paste.txt', { type: 'text/plain' });
                    
                    const formData = new FormData();
                    formData.append('file', file);
                    
                    try {
                        const response = await fetch('http://127.0.0.1:8000/v1/ingestion/document', {
                            method: 'POST',
                            body: formData
                        });
                        
                        if (response.ok) {
                            const result = await response.json();
                            statusDiv.innerHTML = `<span style="color: #4ade80;">✔ Text successfully queued!<br>Task ID: ${result.task_id.substring(0,8)}... (${result.status})</span>`;
                        } else {
                            const err = await response.text();
                            statusDiv.innerHTML = `<span style="color: #ef4444;">❌ Error ${response.status}: ${err.substring(0,60)}</span>`;
                        }
                    } catch (e) {
                        statusDiv.innerHTML = `<span style="color: #ef4444;">❌ Request failed: ${e.message}</span>`;
                    }
                });
            }
        }
        if (!found) {
            statusDiv.innerHTML = `<span style="color: #f59e0b;">No pasteable file or text in clipboard.</span>`;
        }
    });
    </script>
    """, height=220)



