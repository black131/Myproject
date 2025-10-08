"""
eo sensor:kamera, trafik kurallari
otonom aracin en temel gorevi cevreyi tanimak: isaretler(trafik levhalari)

plan program : veri bulma, veriyi yukleme, training, testing
Versiyon 8 kullanilacak
"""
from ultralytics import YOLO
#modeli sec yolov8 nano modeli
model=YOLO("yolov8n.pt")
model.train(
     data="traffic_sign_detection/data.yaml", #yaml dosyasında veri yolları ve sınıf isimleri tanımlı
     epochs=2, #egitim dongu sayisi
     imgsz=640,#gorsel boyutu
     batch=16, #16 dan buyuk yapmayın. Minibatch boyutu donanıma baglıdır.
     name="traffic_sign_model", #cıktı klasoru adı
     lr0=0.01, #baslangıcta ki ogrenme oranı
     optimizer="SGD",#optimizer alternatif olarak ADAM kullanilabilir.
     weight_decay=0.0005, #agirlik cezasi overfitting i engellemek icin
     momentum=0.935, #SGD momentum
     patience=50,# early stopping icin sabir suresi
     workers=2, #data loader worker sayisi
     device="cpu", #cpu veya cuda
     save=True, #kac epoch varsa kaydet.
     save_period=1, #kac epochda bir kayit yapilacagi
     val=True, # her epoch sonucunda val gerceklestir.
     verbose=True,          
)

