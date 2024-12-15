import cv2
import torch
from ultralytics import YOLO

# Cargar el modelo YOLOv8 entrenado
model = YOLO("C:/Users/jarot/PycharmProjects/MachineLearningCert3/pythonProject/runs/detect/train/weights/best.pt")

# Intentar abrir la cámara usando CAP_DSHOW para Windows
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Verificar si la cámara está abierta correctamente
if not cap.isOpened():
    print("Error: No se puede acceder a la cámara")
    exit()

while True:
    # Leer un cuadro de la cámara
    ret, frame = cap.read()

    if not ret:
        print("Error: No se puede recibir la imagen de la cámara")
        break

    # Realizar la detección en el cuadro actual
    results = model(frame)

    # Dibujar los resultados de detección en el cuadro
    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = box.conf[0]  # Confianza de la detección
            if conf > 0.7:  # Filtrar detecciones por confianza
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas de la caja
                label = result.names[int(box.cls[0])]  # Nombre de la clase detectada

                # Dibujar la caja y la etiqueta en el cuadro
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostrar el cuadro con las detecciones
    cv2.imshow("Detección en tiempo real", frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
