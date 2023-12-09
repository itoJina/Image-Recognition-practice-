from flask import Flask, request, render_template, send_from_directory
import cv2
import numpy as np
import os
import uuid

app = Flask(__name__)

def process_images(file1, file2):
    # ファイルが正しくアップロードされていることを確認
    if file1 and file2:
        # 画像を読み込み、グレースケール化
        img = cv2.imdecode(np.fromstring(file1.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        template = cv2.imdecode(np.fromstring(file2.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # 類似度を算出
        res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)

        # 類似度の最大値とその位置を取得
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # テンプレート画像の幅と高さを取得
        h, w = template.shape[:2]

        # 最大類似度の位置に赤枠をつける
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)

        # 画像全体を保存
        filename = f"{uuid.uuid4().hex}.jpeg"
        cv2.imwrite(f'static/{filename}', img)
        filenames = [filename]

        return filenames
    else:
        return []

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # 学習画像と検索対象画像のアップロード
        file1 = request.files['file1']
        file2 = request.files['file2']

        filenames = process_images(file1, file2)

        # テンプレートにファイル名のリストを渡す
        return render_template('result.html', filenames=filenames)

    return render_template('upload.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(directory='static', filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
