import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time

# ==========================================
# ส่วนตั้งค่า (CONFIG)
# ==========================================
MODEL_PATH = "model.tflite"
LABEL_PATH = "labels.txt"
MIN_CONFIDENCE = 0.55     # ความมั่นใจ 55% ขึ้นไปค่อยแจ้งเตือน
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
INPUT_MEAN = 127.5
INPUT_STD = 127.5

# เลือกเฉพาะ ID ของยานพาหนะจาก COCO Dataset
# (อ้างอิง: 2=car, 3=motorcycle, 5=bus, 7=truck ในบาง version อาจขยับ +/- 1)
# ถ้า Detect ผิดให้ลองแก้เลขตรงนี้ครับ
TARGET_LABELS = ['car', 'motorcycle', 'bus', 'truck'] 

# ==========================================

def load_labels(filename):
    with open(filename, 'r') as f:
        # อ่านไฟล์และตัดช่องว่างออก
        return [line.strip() for line in f.readlines()]

def main():
    # 1. โหลด Model TFLite
    print(f"Loading model: {MODEL_PATH}...")
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    # รับค่าข้อมูล Input/Output ของ Model
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    floating_model = (input_details[0]['dtype'] == np.float32)

    labels = load_labels(LABEL_PATH)

    # 2. ตั้งค่ากล้อง USB
    print("Starting Camera...")
    cap = cv2.VideoCapture(0)
    
    # [สำคัญ] บังคับใช้ MJPEG เพื่อลดภาระ CPU ของ Pi Zero
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    # ตัวแปรสำหรับคำนวณ FPS
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    print("System Ready! Press 'q' to quit.")

    while True:
        t1 = cv2.getTickCount()

        # อ่านภาพจากกล้อง
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        # ย่อภาพให้เท่ากับขนาดที่ Model ต้องการ (300x300)
        frame_resized = cv2.resize(frame, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # ถ้า Model เป็น Float ให้ Normalize (แต่รุ่นที่ให้โหลดไปเป็น Quantized จะข้ามส่วนนี้)
        if floating_model:
            input_data = (np.float32(input_data) - INPUT_MEAN) / INPUT_STD

        # 3. ส่งข้อมูลเข้า Model (Inference)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # 4. รับผลลัพธ์
        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # ตำแหน่งสี่เหลี่ยม
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # ประเภทวัตถุ (ID)
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # ความมั่นใจ (0.0 - 1.0)

        # 5. วาดกรอบสี่เหลี่ยม
        for i in range(len(scores)):
            if ((scores[i] > MIN_CONFIDENCE) and (scores[i] <= 1.0)):
                
                # ดึงชื่อวัตถุจาก Label list
                class_id = int(classes[i])
                if class_id < len(labels):
                    object_name = labels[class_id]

                    # ตรวจสอบว่าเป็นยานพาหนะที่เราสนใจหรือไม่
                    if object_name in TARGET_LABELS:
                        
                        # คำนวณพิกัดจริงบนหน้าจอ
                        ymin = int(max(1, (boxes[i][0] * CAMERA_HEIGHT)))
                        xmin = int(max(1, (boxes[i][1] * CAMERA_WIDTH)))
                        ymax = int(min(CAMERA_HEIGHT, (boxes[i][2] * CAMERA_HEIGHT)))
                        xmax = int(min(CAMERA_WIDTH, (boxes[i][3] * CAMERA_WIDTH)))

                        # วาดกรอบและใส่ชื่อ
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
                        
                        label_text = f"{object_name}: {int(scores[i]*100)}%"
                        labelSize, baseLine = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        
                        # วาดพื้นหลังตัวหนังสือให้อ่านง่าย
                        cv2.rectangle(frame, (xmin, ymin - labelSize[1]), (xmin + labelSize[0], ymin + baseLine), (255, 255, 255), cv2.FILLED)
                        cv2.putText(frame, label_text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                        
                        # (Optional) ตรงนี้ใส่โค้ดให้แจ้งเตือนได้ เช่น print("Found Car!")

        # คำนวณ FPS
        t2 = cv2.getTickCount()
        time1 = (t2 - t1) / freq
        frame_rate_calc = 1 / time1

        # แสดง FPS มุมซ้ายบน
        cv2.putText(frame, "FPS: {0:.2f}".format(frame_rate_calc), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

        # แสดงภาพ (ถ้าเปิดผ่าน SSH หรือไม่มีจอ ให้ Comment บรรทัดข้างล่างนี้ออก)
        cv2.imshow('Vehicle Detector', frame)

        # กด q เพื่อออก
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
