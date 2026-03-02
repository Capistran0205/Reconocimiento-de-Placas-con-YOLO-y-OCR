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


## Dependencias
Si deseas instalar exactamente las mismas versiones usadas en este proyecto, utiliza el archivo `requirements.txt`:

```
certifi==2026.2.25
charset-normalizer==3.4.4
contourpy==1.3.3
cycler==0.12.1
easyocr==1.7.2
filelock==3.25.0
fonttools==4.61.1
fsspec==2026.2.0
idna==3.11
ImageIO==2.37.2
Jinja2==3.1.6
kiwisolver==1.4.9
lazy_loader==0.4
MarkupSafe==3.0.3
matplotlib==3.10.8
mpmath==1.3.0
networkx==3.6.1
ninja==1.13.0
numpy==2.4.2
opencv-python==4.13.0.92
packaging==26.0
pillow==12.1.1
polars==1.38.1
polars-runtime-32==1.38.1
psutil==7.2.2
pyclipper==1.4.0
pyparsing==3.3.2
python-bidi==0.6.7
python-dateutil==2.9.0.post0
PyYAML==6.0.3
requests==2.32.5
scikit-image==0.26.0
scipy==1.17.1
setuptools==82.0.0
shapely==2.1.2
six==1.17.0
sympy==1.14.0
tifffile==2026.2.24
torch==2.10.0
torchvision==0.25.0
typing_extensions==4.15.0
ultralytics==8.4.19
ultralytics-thop==2.0.18
urllib3==2.6.3
```

Instalación:
```
pip install -r requirements.txt
```
- Posterior integración con una aplicación móvil para probar el modelo desde un celular