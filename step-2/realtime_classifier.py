import cv2
from card_classifier import CardClassifier
from utils.card_detection import detect_card
from utils.image_processing import get_warped_card


def main():
    # Load model
    classifier = CardClassifier()
    classifier.load_model()

    cap = cv2.VideoCapture(2)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Deteksi kartu
        card_found, corners, _, _ = detect_card(frame)

        if card_found and corners is not None:
            # Get warped card image
            _, binary_warped = get_warped_card(frame, corners)

            # Predict
            card_class, confidence = classifier.predict(binary_warped)

            # Tampilkan hasil
            text = f"{card_class} ({confidence:.2%})"
            cv2.putText(frame, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Card Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
