import os
import subprocess
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import requests
from PIL import Image
import cv2
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from gtts import gTTS
from playsound import playsound
import pygame
from collections import Counter
import pytesseract

# def transcribe_audio_from_microphone():
recognizer = sr.Recognizer()
microphone = sr.Microphone()
with microphone as source:
    recognizer.adjust_for_ambient_noise(source)
    print("Say something!")
    audio = recognizer.listen(source, timeout=10.0)
try:
    transcription = recognizer.recognize_google(audio)
    print("You said: " + transcription)
    # return transcription
except sr.UnknownValueError:
    transcription = "Could not transcribe audio"
    print(transcription)
        # return transcription

def capture_image_from_webcam():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return None
    
    # Capture a frame from the webcam
    ret, frame = cap.read()
    
    # Release the webcam
    cap.release()
    
    if ret:
        # Save the captured frame as an image
        cv2.imwrite("captured_image.jpg", frame)
        return "captured_image.jpg"
    else:
        print("Error: Could not capture image from webcam")
        return None

def ocr(image_path):
    image = Image.open(image_path)

    # Perform OCR on the image
    text = pytesseract.image_to_string(image)

    # Print the extracted text
    print(text)

    # Language in which you want to convert
    language = 'en'

    # Create a gTTS (Google Text-to-Speech) object
    tts = gTTS(text=text, lang=language, slow=False)

    # Save the generated audio to a file
    tts.save("output.mp3")

    # Initialize pygame mixer
    pygame.mixer.init()

    # Load the audio file
    pygame.mixer.music.load("output.mp3")

    # Play the audio using pygame
    pygame.mixer.music.play()

    # Keep the script running to allow audio to play
    input("Press Enter to exit...")

def alert(image_path):
    image = Image.open(image_path)

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    
    detected_labels = []

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        label_name = model.config.id2label[label.item()]
        detected_labels.append(label_name)

        print(
            f"Detected {label_name} with confidence {round(score.item(), 3)} at location {box}"
        )

    # Count occurrences of each label
    label_counts = Counter(detected_labels)

    # Create the formatted detected_labels_text
    detected_labels_text1 = "There are "+", ".join(f"{count} {label}" for label, count in label_counts.items())
    print(f"Detected Labels Text: {detected_labels_text1}")

    language = 'en'

    # Create a gTTS (Google Text-to-Speech) object
    tts = gTTS(text=detected_labels_text1, lang=language, slow=False)

    # Save the generated audio to a file
    tts.save("output.mp3")

    # Initialize pygame mixer
    pygame.mixer.init()

    # Load the audio file
    pygame.mixer.music.load("output.mp3")

    # Play the audio using pygame
    pygame.mixer.music.play()

    # Keep the script running to allow audio to play
    input("Press Enter to exit...")

# Capture an image from the webcam



# Perform OCR if the transcription is "ocr"
if transcription == "ocr":
    image_path = capture_image_from_webcam()
    ocr(image_path)
# Perform object detection if the transcription is "alert"
elif transcription == "alert":
    image_path = capture_image_from_webcam()
    alert(image_path)
