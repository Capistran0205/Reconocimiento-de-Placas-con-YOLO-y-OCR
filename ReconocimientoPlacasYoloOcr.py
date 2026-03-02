import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import re
from collections import defaultdict, deque

# Importando el modelo previamente entrenado de Yolo
model = YOLO("license_plate_best.pt") # Ajuste fino de los pesos 
reader = easyocr.Reader(['es', 'en'], gpu=True)

# Implementando la Expresión Regular para el Reconocimiento de Placas
# Patrón para Letra Letra Letra - Numero Numero Numero - Letra
# Formato XXX-000-X
plate_pattern_1 = re.compile(r'^[A-Z]{3}-\d{3}-[A-Z]{1}$')
# Formato X-000-XXX
plate_pattern_2 = re.compile(r'^[A-Z]{1}-\d{3}-[A-Z]{3}$')
plate_universe_pattern = re.compile(r'^([A-Z]{3}-\d{3}-[A-Z]|[A-Z]-\d{3}-[A-Z]{3})$')

# Ahora hay un detalle a considerar con el OCR
# Puede existir la confusión entre los siguientes casos: 
# ¿Es un 0 o una O? ¿Es un 1 o una I? ¿Es un 5 o una S? ¿Es un 8 o una B?
# La siguiente función busca solucionar eso
def correct_plate_format(ocr_text):
    mapping_num_to_alpha = {"0": "O", "1": "I", "5": "S", "8": "B"}
    mapping_alpha_to_num = {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8"}

    ocr_text = ocr_text.upper().strip().replace("-", "")

    if len(ocr_text) != 7:
        return ""

    # Detectar el formato según la posición 1
    # Formato 1 (XXX000X): posición 1 debe ser letra
    # Formato 2 (X000XXX): posición 1 debe ser dígito
    ch1 = ocr_text[1]
    if ch1.isalpha() or ch1 in mapping_num_to_alpha:
        formato = 1  # XXX-000-X → letras en 0,1,2,6 | números en 3,4,5
        letter_positions = {0, 1, 2, 6}
        digit_positions  = {3, 4, 5}
    elif ch1.isdigit() or ch1 in mapping_alpha_to_num:
        formato = 2  # X-000-XXX → letras en 0,4,5,6 | números en 1,2,3
        letter_positions = {0, 4, 5, 6}
        digit_positions  = {1, 2, 3}
    else:
        return ""  # No encaja en ningún formato

    corrected = []
    for i, ch in enumerate(ocr_text):
        if i in letter_positions:
            if ch.isdigit() and ch in mapping_num_to_alpha:
                corrected.append(mapping_num_to_alpha[ch])  # Número → Letra
            elif ch.isalpha():
                corrected.append(ch)
            else:
                return ""  # Carácter inválido
        elif i in digit_positions:
            if ch.isalpha() and ch in mapping_alpha_to_num:
                corrected.append(mapping_alpha_to_num[ch])  # Letra → Número
            elif ch.isdigit():
                corrected.append(ch)
            else:
                return ""  # Carácter inválido

    # Reconstruir con guiones según el formato detectado
    result = "".join(corrected)
    if formato == 1:
        return f"{result[:3]}-{result[3:6]}-{result[6]}"  # XXX-000-X
    else:
        return f"{result[0]}-{result[1:4]}-{result[4:]}"  # X-000-XXX

 
 # Función para preprocesamiento de la placa dentro de la región antes del OCR
def recognize_plate(plate_crop):
    if plate_crop.size == 0:
        return ""

     # Preprocesamiento para OCR
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plate_resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    try:
        ocr_result = reader.readtext(
            plate_resized,
            detail=0,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        )
        if len(ocr_result) > 0:
            candidate = correct_plate_format(ocr_result[0]) # Corrección de los carácteres
            # ✅ Ahora valida contra ambos patrones
            if candidate and (plate_pattern_1.match(candidate) or plate_pattern_2.match(candidate)):
                return candidate
    except:
        pass

    return ""

plate_history = defaultdict(lambda: deque(maxlen=10)) # Las últimas 10 predicciones
plate_final = {}

def get_box_id(x1, y1, x2, y2):
    # Uso de las coordenadas
    return f"{int(x1/10)}_{int(y1/10)}_{int(x2/10)}_{int(y2/10)}"

def get_stable_plate(box_id, new_text):
    if new_text:
        plate_history[box_id].append(new_text)
        # Los frames más comúnes o mayoritarios
        most_common = max(set(plate_history[box_id]), key = plate_history[box_id].count)
        plate_final[box_id] = most_common
    return plate_final.get(box_id, "")

# El video de inferencia
input_video = "Carretera_video.mp4"
output_video = "Carretera_video_placasv3.mp4"

cap = cv2.VideoCapture(input_video)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))

CONF_THRESH = 0.3

# Flujo de operación Frame a Frame
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, verbose = False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = float(box.conf.cpu().numpy().item())
            if conf < CONF_THRESH:
                continue

            x1,y1,x2,y2 = map(int, box.xyxy.cpu().numpy()[0])
            plate_crop = frame[y1:y2, x1:x2]

            # Ocr con corrección
            text = recognize_plate(plate_crop)

            # Usar history para estabilizar el texto
            box_id = get_box_id(x1,x2,y1,y2)
            stable_text = get_stable_plate(box_id, text)

            # Dibujo del rectangulo para la placa
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Aplicar un zoom sobre la placa detectada
            if plate_crop.size > 0:
                overlay_h, overlay_w = 150, 400
                plate_resized = cv2.resize(plate_crop, (overlay_w, overlay_h))

                oy1 = max(0, y1 - overlay_h - 40)
                ox1 = x1
                oy2, ox2 = oy1 + overlay_h, ox1 + overlay_w

                if oy2 <= frame.shape[0] and ox2 <= frame.shape[1]:
                    frame[oy1:oy2, ox1:ox2] = plate_resized

                    # Mostrar el texto OCR estabilizado sobre 
                    if stable_text:
                        cv2.putText(frame, stable_text, (ox1, oy1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 6) # Linea negra
                        cv2.putText(frame, stable_text, (ox1, oy1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3) # Texto blanco
    out.write(frame)
    cv2.imshow("Annoted Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cierre completo de las ventanas
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video con detección de placas guardado en ", output_video)