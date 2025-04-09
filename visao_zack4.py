import cv2
import mediapipe as mp

# Inicializações
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Configurações dos modelos
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)

# Configuração da webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Erro ao acessar a câmera!")
    exit()

# Variáveis de controle
modo = 'face'
cor_modo = (0, 255, 0)

# Especificações de desenho
estilo_linhas_face = mp_drawing.DrawingSpec(color=cor_modo, thickness=1, circle_radius=1)
estilo_pontos_face = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1)

def detectar_face(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Desenha todas as conexões do mesh facial
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=estilo_pontos_face,
                connection_drawing_spec=estilo_linhas_face)
    return len(results.multi_face_landmarks) if results.multi_face_landmarks else 0

def detectar_maos(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=cor_modo, thickness=2),
                mp_drawing.DrawingSpec(color=(255,255,255), thickness=1))
    return len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0

while True:
    success, frame = cap.read()
    if not success:
        print("Erro na captura do frame")
        break

    frame = cv2.flip(frame, 1)
    contador = 0

    try:
        if modo == 'face':
            contador = detectar_face(frame)
        elif modo == 'mao':
            contador = detectar_maos(frame)

        # Interface do usuário
        cv2.putText(frame, f"Modo: {modo} - Detectados: {contador}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor_modo, 2)
        cv2.putText(frame, "'m' - Mudar Modo | 'q' - Sair", (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        cv2.imshow('Detecção Avançada', frame)

    except Exception as e:
        print(f"Erro: {str(e)}")
        break

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('m'):
        modo = 'mao' if modo == 'face' else 'face'
        cor_modo = (0, 0, 255) if modo == 'mao' else (0, 255, 0)

cap.release()
cv2.destroyAllWindows()
hands.close()
face_mesh.close()
