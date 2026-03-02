# Reconocimiento de Placas con YOLO y EasyOCR

Este proyecto detecta y reconoce placas vehiculares en videos usando YOLOv8 y EasyOCR.

## Características principales
- Detección de placas con un modelo YOLOv8 entrenado (`license_plate_best.pt`).
- Reconocimiento de texto en placas usando EasyOCR, con corrección automática de errores comunes (confusión entre letras y números).
- Validación de formatos de placas típicos actuales de Veracruz (ejemplo: `XXX-000-X` y `X-000-XXX`).
- Estabilización de resultados mediante historial de predicciones.
- Visualización de placas y texto reconocido en el video de salida.

## Archivos principales
- `ReconocimientoPlacasYoloOcr.py`: Script principal.
- `license_plate_best.pt`: Modelo YOLOv8 entrenado.
- `yolov8n.pt`: Modelo base YOLOv8.
- Carpeta `Recursos/`: Imágenes de ejemplo.

## Requisitos
- Python 3.8+
- OpenCV
- ultralytics (YOLO)
- easyocr
- numpy

Instalación:
```
pip install opencv-python ultralytics easyocr numpy
```

## Ejecución
Coloca el video de entrada en la carpeta del proyecto y actualiza el nombre en el script (`input_video`). Ejecuta:
```
python ReconocimientoPlacasYoloOcr.py
```
El video anotado se guardará como `Carretera_video_placasv3.mp4`.

## Notas
- El script corrige errores comunes del OCR (por ejemplo, '0' y 'O').
- Optimizado para formatos de placas comunes en México y Latinoamérica.
- Se recomienda usar GPU para mayor velocidad con EasyOCR.
- Posterior integración con una aplicación móvil para probar el modelo desde un celular