import os
import shutil
import zipfile


def zip_dir(src_dir, zip_path):
    """
    将指定目录压缩为 zip

    :param src_dir: 要压缩的目录
    :param zip_path: 生成的 zip 文件路径
    """
    src_dir = os.path.abspath(src_dir)
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                file_path = os.path.join(root, file)

                # 压缩包内路径
                arcname = os.path.relpath(file_path, src_dir)

                z.write(file_path, arcname)


def clear_dir(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)

        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.remove(file_path)

        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)