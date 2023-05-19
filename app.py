import os
import time
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from extract import extract_docx

st.set_page_config(page_icon="📖", layout="wide")

with st.container():
    components.html(
        '''
    <body style="background: rgb(116,152,173);">
        <section class="d-flex flex-column justify-content-around" id="hero" style="width: 100%;height: 60vh;background: #7498ad;">
            <div class="container" style="text-align: center;">
                <h1 style="font-size: 65px;">PLAGIARISM DETECTION</h1>
            </div>
        </section>

    </body>

    </html>
        ''',
        height=300,
    )
    image = Image.open('image/bgPD.png')
    col1, col2, col3 = st.columns(3)
    with col2:
        st.image(image, use_column_width='always')
    components.html('''
    <h1 style="font-size: 35px;text-align: center;font-family: Adamina, serif;"><br>A program allows checking for plagiarism of the input text against texts in the database.</h1></div>
    ''')

st.divider()

output_directory = 'data'
file = st.file_uploader('Choose a file: ')

if file is not None:
    # Lấy tên file
    filename = file.name

    # Hiển thị tiến trình tải file
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Đường dẫn đến file docx ban đầu
    file_path = os.path.join(output_directory, filename)

    # Lưu file docx vào thư mục mới
    with open(file_path, 'wb') as output_file:
        output_file.write(file.getvalue())

    # Mô phỏng quá trình tải file
    for i in range(101):
        time.sleep(0.005)  # Để mô phỏng quá trình tải lâu hơn thực tế
        progress_bar.progress(i)
        status_text.text(f'Extracting: {i}%')

    st.write('Successful!')

    col1, col2 = st.columns(2)
    with col1:
        a = extract_docx(file_path)
        os.remove(file_path)
        text_to_display = '\n'.join(a)
        st.text_area('Context: ', value=text_to_display)

else:
    st.write('Please upload the data to check for plagiarism (docx).')
