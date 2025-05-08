import streamlit as st
import tempfile
import os

# Title
st.set_page_config(page_title="Truy suất sự kiện dựa trên Video", layout="centered")
st.title("🧠 Hệ thống truy xuất sự kiện nội dung video dành cho tiếng Việt")

st.markdown("""
Hệ thống sẽ trích xuất nội dung từ video bằng giọng nói tiếng Việt (ví dụ: bài giảng, bản tin thời sự...)  
Sau đó sẽ tóm tắt thành văn bản ngắn gọn.
""")

# Upload video
uploaded_file = st.file_uploader("🎥 Tải lên video (MP4)", type=["mp4", "mkv", "mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_video_path = tmp_file.name

    st.video(temp_video_path)
    st.info("Đang xử lý video... vui lòng chờ ⏳")

    summarizer = VideoSummarizer()

    try:
        #summary_result = summarizer.summarize_video(temp_video_path)
        st.success("✅ Tóm tắt thành công!")
        st.subheader("📄 Nội dung tóm tắt:")
        st.write(summary_result)
    except Exception as e:
        st.error(f"❌ Lỗi xử lý: {str(e)}")

    # Cleanup
    os.remove(temp_video_path)
