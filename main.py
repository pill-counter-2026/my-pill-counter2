import streamlit as st
from ultralytics import YOLO
from PIL import Image

# タイトル
st.title("💊 AI薬カウンター")

# 1. 学習済みモデルの読み込み
model = YOLO('best.pt')

# 2. ユーザーに種類を選んでもらう（これによって重複判定の厳しさを変えます）
mode = st.radio(
    "数えたい薬の種類を選んでください",
    ["錠剤・カプセル全体", "カプセル専用（重複防止）", "半錠（割った薬）"]
)

# 3. 写真のアップロード
img_file = st.file_uploader("写真を撮るか画像を選んでください", type=['jpg', 'png', 'jpeg'])

if img_file:
    img = Image.open(img_file)
    st.image(img, caption="読み込み完了", use_container_width=True)
    
    # 4. モードに合わせて「重なりの許容度(iou)」を自動設定
    if mode == "カプセル専用（重複防止）":
        my_iou = 0.2  # 重なりに凄く厳しくする（1つのカプセルに2枠出るのを防ぐ）
    elif mode == "半錠（割った薬）":
        my_iou = 0.5  # 小さい薬が多いので、少し重なりを許容する
    else:
        my_iou = 0.3  # 標準的な設定
    
    if st.button("数を数える"):
        # 設定を反映して判定実行
        results = model.predict(source=img, conf=0.25, iou=my_iou)
        
        # 判定後の画像を作成
        res_img = results[0].plot()
        st.image(res_img, caption="判定結果", use_container_width=True)
        
        # カウント結果を表示
        count = len(results[0].boxes)
        st.success(f"カウント結果: {count} 個")