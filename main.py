from ultralytics import YOLO
import cv2

# --------------------------------------------------
# 1. CHARGER LE MODELE YOLOv8
# --------------------------------------------------
model = YOLO("yolov8x.pt")  # on peux mettre n, s, m, l selon le modèle


# --------------------------------------------------
# 2. CHOISIR LE MODE D'ENTRÉE
# --------------------------------------------------
# Exemple :
#   input_path = "test.jpg"
#   input_path = "video.mp4"

input_path = "test.jpg"   # <--- modifier ici l'image ou la vidéo à tester


# --------------------------------------------------
# 3. DETECTION SUR UNE IMAGE
# --------------------------------------------------
def detect_image(path):
    img = cv2.imread(path)
    if img is None:
        print("Erreur : image introuvable.")
        return
    
    results = model(img)

    annotated = results[0].plot()  # image avec annotations YOLO

    cv2.imshow("Détection YOLOv8", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# --------------------------------------------------
# 4. DETECTION SUR UNE VIDEO
# --------------------------------------------------
def detect_video(path):
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la vidéo.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated = results[0].plot()

        cv2.imshow("Détection YOLOv8", annotated)

        # Quitter avec Q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# --------------------------------------------------
# 5. EXECUTION AUTOMATIQUE
# --------------------------------------------------
if input_path.lower().endswith((".jpg", ".png", ".jpeg")):
    detect_image(input_path)

elif input_path.lower().endswith((".mp4", ".avi", ".mov")):
    detect_video(input_path)

else:
    print("Format non reconnu : utilise une image ou une vidéo.")
