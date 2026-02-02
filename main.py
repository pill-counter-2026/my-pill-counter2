import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.title("💊 AI薬カウンター")
model = YOLO('best.pt')

# スマホのカメラとファイル選択の両方に対応
img_file = st.file_uploader("写真を撮るか画像を選んでください", type=['jpg', 'png', 'jpeg'])

if img_file:
    img = Image.open(img_file)
    st.image(img, caption="読み込み完了", use_container_width=True)

    if st.button("数を数える"):
        results = model.predict(source=img, conf=0.25)
        res_img = results[0].plot()
        st.image(res_img, caption="判定結果", use_container_width=True)
        st.success(f"カウント結果: {len(results[0].boxes)} 個")