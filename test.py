from ultralytics import YOLO
import cv2
#modeli yukle
model=YOLO("runs/traffic_sign_model/weights/yolo11n.pt")

#test edilecek gorselin yuklenmesi
image_path="test1.jpg"
image= cv2.imread(image_path)
#image tahmini
results=model(image_path)[0]
print(results)

#kutu cizimi
for box in results.boxes:
     #koordinatlar
     x1,y1,x2,y2=map(int,box.xyxy[0]) #kose koordinatlar
     cls_id=int(box.cls[0])
     confidence=float(box.conf[0])
     label=f"{model.names[cls_id]} conf: {confidence:.2f}" #detection label
     
     #kutu cizimi
     cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
     cv2.putText(image,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
cv2.imshow("Prediction",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("prediction_results.jpg",image)