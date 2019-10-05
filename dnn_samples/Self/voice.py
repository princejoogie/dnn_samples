import speech_recognition as sr
import pyttsx

r = sr.Recognizer()
engine = pyttsx.init()

while True:
    with sr.Microphone() as source:
        print('Say Anything: ')
        audio = r.listen(source)

        try:
            text = r.recognize_google(audio)
            
            print('You Said: {}'.format(text))
            if text == "hey computer":
                engine.say('Hello, Prince')
                engine.runAndWait()
        except:
            print('Sorry, Could not Recognitze your Voice.')
