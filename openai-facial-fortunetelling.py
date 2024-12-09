import cv2
import requests
import asyncio
import time
from aiohttp import ClientSession
from aiohttp import FormData

# Function to read API credentials
def read_credentials(api_key_file, api_secret_file):
    with open(api_key_file, 'r') as file:
        api_key = file.read().strip()
    with open(api_secret_file, 'r') as file:
        api_secret = file.read().strip()
    return api_key, api_secret

async def detect_faces_async(frame, api_key, api_secret, session):
    url = "https://api-us.faceplusplus.com/facepp/v3/detect"

    # Encode the frame as an image
    _, img_encoded = cv2.imencode('.jpg', frame)

    # Prepare the request
    form = FormData()
    form.add_field("image_file", img_encoded.tobytes(), filename="frame.jpg", content_type="image/jpeg")
    form.add_field("api_key", api_key)
    form.add_field("api_secret", api_secret)

    try:
        # Send the request
        async with session.post(url, data=form) as response:
            if response.status == 200:
                result = await response.json()
                print(f"[DEBUG] Detect Faces Response: {result}")

                # Extract face_tokens
                if 'faces' in result and result['faces']:
                    face_tokens = [face['face_token'] for face in result['faces']]
                    return face_tokens
                else:
                    return None  # No faces detected
            else:
                # Log non-200 responses
                error_text = await response.text()
                print(f"[DEBUG] Detect Faces Error: {response.status} - {error_text}")
                return None

    except Exception as e:
        print(f"[DEBUG] Detect Faces Exception: {e}")
        return None
        
async def analyze_faces_async(face_tokens, api_key, api_secret, session):
    url = "https://api-us.faceplusplus.com/facepp/v3/face/analyze"

    # Prepare the request
    data = {
        "api_key": api_key,
        "api_secret": api_secret,
        "face_tokens": ",".join(face_tokens),  # Pass multiple tokens if needed
        "return_attributes": "emotion"
    }

    try:
        # Send the request
        async with session.post(url, data=data) as response:
            if response.status == 200:
                result = await response.json()
                print(f"[DEBUG] Analyze Faces Response: {result}")

                # Extract emotions for each face
                emotions = [
                    {
                        "face_token": face['face_token'],
                        "dominant_emotion": max(face['attributes']['emotion'], key=face['attributes']['emotion'].get)
                    }
                    for face in result['faces']
                ]
                return emotions
            else:
                # Log non-200 responses
                error_text = await response.text()
                print(f"[DEBUG] Analyze Faces Error: {response.status} - {error_text}")
                return None

    except Exception as e:
        print(f"[DEBUG] Analyze Faces Exception: {e}")
        return None

# Asynchronous function to send API requests
async def analyze_frame_async(frame, api_key, api_secret, session):
    url = "https://api-us.faceplusplus.com/facepp/v3/face/analyze"

    # Encode the frame as an image
    _, img_encoded = cv2.imencode('.jpg', frame)

    # Prepare the request
    files = {
        "image_file": ("frame.jpg", img_encoded.tobytes())
    }
    data = {
        "api_key": api_key,
        "api_secret": api_secret,
        "return_attributes": "emotion"
    }

    try:
        # Log the start of the request
        print("[DEBUG] Sending request to Face++ API...")

        # Send the request
        async with session.post(url, data=data) as response:
            print(f"[DEBUG] Response Status Code: {response.status}")

            if response.status == 200:
                result = await response.json()
                print(f"[DEBUG] API Response: {result}")

                if 'faces' in result and result['faces']:
                    emotions = result['faces'][0]['attributes']['emotion']
                    dominant_emotion = max(emotions, key=emotions.get)
                    return dominant_emotion
                else:
                    return "No face detected"
            else:
                # Log non-200 responses
                error_text = await response.text()
                print(f"[DEBUG] API Error: {response.status} - {error_text}")
                return f"Error: {response.status}, {error_text}"

    except Exception as e:
        print(f"[DEBUG] Request Exception: {e}")
        return "Error: API request failed"

# Function to handle video stream
async def video_emotion_analysis(api_key, api_secret):
    cap = cv2.VideoCapture(0)
    frame_count = 0
    frame_interval = 30  # Process one frame every 30 frames (1 second for 30 FPS)

    async with ClientSession() as session:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame. Exiting...")
                break

            # Resize frame for faster processing
            frame = cv2.resize(frame, (320, 240))
            frame_count += 1

            if frame_count % frame_interval == 0:
                print("[DEBUG] Processing frame...")
                face_tokens = await detect_faces_async(frame, api_key, api_secret, session)

                if face_tokens:
                    emotions = await analyze_faces_async(face_tokens, api_key, api_secret, session)
                    if emotions:
                        # Show emotions for each face
                        emotion_text = ", ".join([f"{e['dominant_emotion']}" for e in emotions])
                    else:
                        emotion_text = "No emotion data"
                else:
                    emotion_text = "No faces detected"
            else:
                emotion_text = "Analyzing..."

            # Display the detected emotion on the video feed
            cv2.putText(frame, f"Emotion: {emotion_text}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show the video feed
            cv2.imshow("Emotion Analysis", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


# Read API credentials
api_key_file = "api_key.txt"
api_secret_file = "api_secret.txt"
api_key, api_secret = read_credentials(api_key_file, api_secret_file)

# Run the asynchronous video emotion analysis
asyncio.run(video_emotion_analysis(api_key, api_secret))
