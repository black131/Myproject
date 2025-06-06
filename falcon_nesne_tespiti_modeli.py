# -*- coding: utf-8 -*-
"""falcon_nesne_tespiti_modeli.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1mhsywhSCIIrxVgG53uXemkMp0LdIQRR8
"""

!pip install ultralytics

from google.colab import drive
drive.mount('/content/drive')

!ls /content/drive/MyDrive/Gorsel_etiketleme/

from ultralytics import YOLO

model=YOLO("yolo11s.pt")

conf_dosyasi_konumu="/content/drive/MyDrive/Gorsel_etiketleme/falcon_colab/conf.yaml"

import os
os.path.exists('/content/drive/MyDrive/Gorsel_etiketleme/falcon_colab/conf.yaml')

sonuclar = model.train(
    data=conf_dosyasi_konumu,
    epochs=50,
    batch=16,
    imgsz=640
)

import locale
def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

!zip -r falcon_sonuclar.zip /content/runs/

