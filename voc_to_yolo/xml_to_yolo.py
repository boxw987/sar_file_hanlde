# -*- coding: UTF-8 -*-
import os
import glob
import xml.etree.ElementTree as ET

xml_file = r'D:\study\yolov5-master_with_data\yolov5-master\sar_data\MSAR\Annotations'

# 支持中文的类别列表
l = ['飞机', '油罐', '桥梁', '船只','W']

def convert(box, dw, dh):
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]

    x = x / dw
    y = y / dh
    w = w / dw
    h = h / dh

    return x, y, w, h

def f(name_id):
    xml_path = os.path.join(xml_file, f'{name_id}.xml')
    txt_path = os.path.join(r'D:\study\yolov5-master_with_data\yolov5-master\sar_data\MSAR\yolo_style\labels', f'{name_id}.txt')
    
    with open(xml_path, 'r', encoding='utf-8') as xml_o, open(txt_path, 'w', encoding='utf-8') as txt_o:
        pares = ET.parse(xml_o)
        root = pares.getroot()
        objects = root.findall('object')
        size = root.find('size')
        dw = int(size.find('width').text)
        dh = int(size.find('height').text)

        for obj in objects:
            c = l.index(obj.find('name').text)
            bnd = obj.find('bndbox')

            b = (float(bnd.find('xmin').text), float(bnd.find('ymin').text),
                 float(bnd.find('xmax').text), float(bnd.find('ymax').text))

            x, y, w, h = convert(b, dw, dh)

            write_t = "{} {:.5f} {:.5f} {:.5f} {:.5f}\n".format(c, x, y, w, h)
            txt_o.write(write_t)

name = glob.glob(os.path.join(xml_file, "*.xml"))
for i in name:
    name_id = os.path.basename(i)[:-4]
    f(name_id)
