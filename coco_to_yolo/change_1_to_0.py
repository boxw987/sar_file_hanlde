import os

def modify_txt_files(folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            # 修改每一行的第一个数字1为0
            modified_lines = []
            for line in lines:
                if line.strip() and line.strip()[0] == '1':
                    modified_lines.append('0' + line[1:])
                else:
                    modified_lines.append(line)
            
            # 将修改后的内容写回文件
            with open(file_path, 'w', encoding='utf-8') as file:
                file.writelines(modified_lines)

# 指定文件夹路径
folder_path = './sar_data/HRSID_jpg/yolo_file/Dataset/labels/val'
modify_txt_files(folder_path)
