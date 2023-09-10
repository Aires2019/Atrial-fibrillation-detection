import numpy as np
from flask import Flask, request, flash, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
from repredict import repredict
import json

app = Flask(__name__)

# 允许上传的文件格式
ALLOWED_EXTENSIONS = set(['csv', 'xslx', 'txt'])


# 检查格式
def allowed_file(filename):
    return '.' in filename and filename.split('.', 1)[1] in ALLOWED_EXTENSIONS


# upload path
UPLOAD_FOLDER = 'uploads/'


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    # 获取上传文件
    if request.method == 'POST':
        file = request.files['file']
        print(dir(file))
        # 检查对象是否合法
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = 'upload/' + filename
            print(path)
            if filename != file.filename:
                flash("Only support ASCII name")
                return render_template('home.html')
            # 保存文件
            try:
                file.save(os.path.join(UPLOAD_FOLDER, filename))
            except FileNotFoundError:
                os.mkdir(UPLOAD_FOLDER)
                file.save(os.path.join(UPLOAD_FOLDER, filename))
            return redirect(url_for('update', filename=filename))
        else:
            return 'Upload Failed'
    else:
        return render_template('home.html')


def render_file_as_page(filename):
    # 将上传的文件复制到static文件夹中
    data = np.loadtxt(os.path.join(UPLOAD_FOLDER, filename))
    target_name = 'static/' + filename
    np.savetxt(target_name, data)
    # 预测
    preds = repredict(filename)
    result = {"prediction": preds[0], "probability": preds[1], "fileName": filename, "data": preds[2]}
    return result


@app.route('/upload/<path:filename>', methods=['POST', 'GET'])
def update(filename):
    # 输入url加载图片，并返回预测值
    result = render_file_as_page(filename)
    # print(result["data"])
    return render_template('show.html', fname=filename, result=result, test=result["data"].tolist())


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
