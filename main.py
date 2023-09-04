#main and also voice output in all of the codes
import os
import subprocess
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import requests
from PIL import Image


def execute_command(command):
    try:
        subprocess.run(["python", command])
    except Exception as e:
        print(f"Error executing command: {str(e)}")

def speak(text):
    tts = gTTS(text=text, lang="en")
    tts.save("output.mp3")

    # Play the generated audio
    sound = AudioSegment.from_mp3("output.mp3")
    play(sound)

def main():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening for voice command...")
        audio = recognizer.listen(source)

        try:
            # Recognize speech using Google Cloud Speech Recognition
            text = recognizer.recognize_google(audio)
            print(f"Command detected: {text}")

            # Check the recognized command and execute the corresponding Python file
            if "describe" in text.lower():
                speak("Starting image captioning")
                execute_command("sas.py")
            elif "vqa" in text.lower():
                speak("Starting vqa")
                execute_command("pas.py")
            elif "ocr" in text.lower():
                speak("Starting ocr")
                execute_command("ocr.py")
            elif "alert" in text.lower():
                speak("Starting alert")
                execute_command("alert.py")                                
            else:
                speak("No recognized command.")
        except sr.UnknownValueError:
            speak("Google Cloud Speech Recognition could not understand audio.")
        except sr.RequestError as e:
            speak(f"Could not request results from Google Cloud Speech Recognition service; {e}")

if __name__ == "__main__":
    main()
