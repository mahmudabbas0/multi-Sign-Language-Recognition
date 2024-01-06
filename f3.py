import pickle
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk, ImageFont, ImageDraw
import string
import time

font = ImageFont.truetype("a.ttf", 75, encoding="utf-8")

language_models = {
    'EN': {'model_file': './model_enO.h5', 'labels': {i: char for i, char in enumerate(string.ascii_uppercase)}},
    'AR': {'model_file': './model_ar.h5', 'labels': {0: 'ا', 1: 'ب', 2: 'ت', 3: 'ث', 4: 'ج', 5: 'ح', 6: 'خ', 7: 'د', 8: 'ذ', 9: 'ر',
    10: 'ز', 11: 'س', 12: 'ش', 13: 'ص', 14: 'ض', 15: 'ط', 16: 'ظ', 17: 'ع', 18: 'غ',
    19: 'ف', 20: 'ق', 21: 'ك', 22: 'ل', 23: 'م', 24: 'ن', 25: 'هـ', 26: 'و', 27: 'ي'
}},
    'TR': {'model_file': './model_tr.h5', 'labels': {0: 'A', 1: 'B', 2: 'C',3: 'Ç', 4: 'D', 5: 'E',6: 'F', 7: 'G', 8: 'Ğ',9: 'H', 10: 'I', 11: 'İ',12: 'J', 13: 'K', 14: 'L',15: 'M', 16: 'N', 17: 'O',18: 'Ö', 19: 'P', 20: 'R',21: 'S', 22: 'Ş', 23: 'T',24: 'U', 25: 'Ü', 26: 'V',27: 'Y', 28: 'Z', 29: ' '}},
}

current_language = 'TR'

initial_model_dict = pickle.load(open(language_models[current_language]['model_file'], 'rb'))
model = initial_model_dict['model.h5']
labels_dict = language_models[current_language]['labels']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

root = tk.Tk()
root.title("Hand Gesture Recognition")

char_var = tk.StringVar()
sentence_var = tk.StringVar()

char_entry = tk.Entry(root, textvariable=char_var, font=('Arial', 18), state='readonly', width=5)
sentence_entry = tk.Entry(root, textvariable=sentence_var, font=('Arial', 18), state='readonly', width=20)

char_label = tk.Label(root, text="Predicted Character:", font=('Arial', 16))
sentence_label = tk.Label(root, text="Formed Sentence:", font=('Arial', 16))

char_label.grid(row=0, column=0, padx=10, pady=10)
char_entry.grid(row=0, column=1, padx=10, pady=10)
sentence_label.grid(row=1, column=0, padx=10, pady=10)
sentence_entry.grid(row=1, column=1, padx=10, pady=10)

hand_present = False
current_character = ""
hand_start_time = None

def update_gui(predicted_character, formed_sentence):
    char_var.set(predicted_character)
    sentence_var.set(formed_sentence)
    root.update_idletasks()

def process_frame(frame):
    global hand_present, current_character, hand_start_time

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    predicted_character = ""
    if results.multi_hand_landmarks:
        hand_present = True

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            data_aux = []
            x_ = []
            y_ = []

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            current_character = predicted_character

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

            # Use PIL to draw the text on the image
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            draw.text((x1-100, y1 - 100), predicted_character, font=font, fill=(0, 0, 0))
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            # Check if the hand has been present for at least 5 seconds
            current_time = time.time()
            if hand_start_time is None:
                hand_start_time = current_time
            elif current_time - hand_start_time >= 3:
                sentence_var.set(sentence_var.get() + current_character)
                char_var.set(current_character)
                current_character = ""
                hand_start_time = None
    else:
        # Reset the start time when the hand is not present
        hand_start_time = None

    return frame

def update_video_feed():
    ret, frame = cap.read()
    frame = cv2.resize(frame, (800, 600))

    processed_frame = process_frame(frame)
    photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)))
    video_label.config(image=photo)
    video_label.photo = photo
    video_label.after(10, update_video_feed)

cap = cv2.VideoCapture(0)
video_label = tk.Label(root)
video_label.grid(row=2, column=0, columnspan=2)
update_video_feed()

# Function to change language and update the model
def change_language(lang):
    global model, labels_dict, current_language

    current_language = lang
    model_dict = pickle.load(open(language_models[current_language]['model_file'], 'rb'))
    model = model_dict['model.h5']
    labels_dict = language_models[current_language]['labels']

# Clear stored characters function
def clear_characters():
    sentence_var.set("")
    char_var.set("")
    global current_character, hand_start_time
    current_character = ""
    hand_start_time = None

# Function to delete the last character
def delete_last_character():
    current_sentence = sentence_var.get()
    current_char = char_var.get()

    if current_sentence:
        sentence_var.set(current_sentence[:-1])
        char_var.set(current_sentence[-1])
    elif current_char:
        char_var.set("")

    global current_character, hand_start_time
    current_character = ""
    hand_start_time = None

# Delete Last Character button
delete_button = tk.Button(root, text="Delete", command=delete_last_character)
delete_button.grid(row=3, column=4, padx=10, pady=10)

# Clear button
clear_button = tk.Button(root, text="Clear", command=clear_characters)
clear_button.grid(row=3, column=3, padx=10, pady=10)

# Language selection buttons
en_button = tk.Button(root, text="EN", command=lambda: change_language('EN'))
en_button.grid(row=3, column=1, padx=10, pady=10)

ar_button = tk.Button(root, text="AR", command=lambda: change_language('AR'))
ar_button.grid(row=3, column=0, padx=10, pady=10)

tr_button = tk.Button(root, text="TR", command=lambda: change_language('TR'))
tr_button.grid(row=3, column=2, padx=10, pady=10)

root.mainloop()
cap.release()
cv2.destroyAllWindows()
