import cv2
import dlib
import numpy as np
import time

# Inisialisasi detektor wajah, mata, dan prediktor landmark
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Ambil indeks landmark untuk mata kiri dan kanan
left_eye_indices = list(range(36, 42))
right_eye_indices = list(range(42, 48))

# Hitung rasio aspek mata
def aspect_ratio(eye):
    a = np.linalg.norm(eye[1] - eye[5])
    b = np.linalg.norm(eye[2] - eye[4])
    c = np.linalg.norm(eye[0] - eye[3])
    return (a + b) / (2.0 * c)

# Deteksi pergerakan bola mata
def eye_movement_ratio(eye_landmarks):
    left_eye = np.array([eye_landmarks[i] for i in left_eye_indices])
    right_eye = np.array([eye_landmarks[i] for i in right_eye_indices])

    # Hitung rasio aspek mata kiri dan kanan
    left_eye_aspect_ratio = aspect_ratio(left_eye)
    right_eye_aspect_ratio = aspect_ratio(right_eye)

    return left_eye_aspect_ratio, right_eye_aspect_ratio

# Fungsi untuk menggambar garis yang mengindikasikan arah pandang
def draw_gaze_direction(frame, left_eye_center, right_eye_center):
    gaze_x = int((left_eye_center[0] + right_eye_center[0]) / 2)
    gaze_y = int((left_eye_center[1] + right_eye_center[1]) / 2)

    # Gambar garis dari tengah mata ke titik arah pandang
    cv2.line(frame, (gaze_x - 30, gaze_y), (gaze_x + 30, gaze_y), (0, 255, 0), 2)
    cv2.line(frame, (gaze_x, gaze_y - 30), (gaze_x, gaze_y + 30), (0, 255, 0), 2)

# Fungsi utama
def main():
    # Buka kamera
    cap = cv2.VideoCapture(0)

    # Inisialisasi waktu awal
    start_time_eye = time.time()
    start_time_person = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        # Deteksi jumlah orang
        num_people = len(faces)

        for face in faces:
            landmarks = predictor(gray, face)
            eye_landmarks = [(landmark.x, landmark.y) for landmark in landmarks.parts()]
            left_eye_center = np.mean(eye_landmarks[36:42], axis=0).astype(np.int)
            right_eye_center = np.mean(eye_landmarks[42:48], axis=0).astype(np.int)

            # Cek apakah mata melihat ke arah tertentu
            left_eye_ratio, right_eye_ratio = eye_movement_ratio(eye_landmarks)  # Memperbaiki error ini
            if left_eye_ratio < 0.2 and right_eye_ratio > 0.2:
                elapsed_time_eye = time.time() - start_time_eye
                if elapsed_time_eye > 5:
                    print("Terindikasi curang! Mata melirik selama lebih dari 5 detik.")
                    start_time_eye = time.time()
            else:
                start_time_eye = time.time()  # Reset waktu jika mata menghadap ke tengah

            # Gaze tracking
            draw_gaze_direction(frame, left_eye_center, right_eye_center)

        # Cek jumlah orang dan waktu kehadiran
        if num_people > 1:
            elapsed_time_person = time.time() - start_time_person
            if elapsed_time_person > 30:
                print("Membantu peserta ujian.")
            else:
                print("Ada kehadiran orang lain.")
        else:
            start_time_person = time.time()  # Reset waktu jika jumlah orang kurang dari 2

        cv2.imshow("Deteksi Mata dan Orang", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


