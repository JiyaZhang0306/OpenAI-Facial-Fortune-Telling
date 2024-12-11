from openai import AzureOpenAI
import os
import requests
import json
import cv2
import asyncio
from aiohttp import ClientSession, FormData

# Semaphore to limit concurrent requests
semaphore = asyncio.Semaphore(1)

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_KEY"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_version="2023-10-01-preview"
)

# Initial system message for GPT
messages = [
    {"role": "system", "content": "Describe the current emotions based on face features, and write current feelings in a second view. Give me the potential guess about the reason of emotion the people who is detcting, using 'you' to call the charater'. And give back a fortune-telling biscuit to the person based on their facial emotions."}
]

# Function to read API credentials
def read_credentials(api_key_file, api_secret_file):
    with open(api_key_file, 'r') as file:
        api_key = file.read().strip()
    with open(api_secret_file, 'r') as file:
        api_secret = file.read().strip()
    return api_key, api_secret

# Function to detect faces
async def detect_faces_async(frame, api_key, api_secret, session):
    async with semaphore:
        url = "https://api-us.faceplusplus.com/facepp/v3/detect"
        _, img_encoded = cv2.imencode('.jpg', frame)

        # Prepare the request
        form = FormData()
        form.add_field("image_file", img_encoded.tobytes(), filename="frame.jpg", content_type="image/jpeg")
        form.add_field("api_key", api_key)
        form.add_field("api_secret", api_secret)

        try:
            async with session.post(url, data=form) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"[DEBUG] Detect Faces Response: {result}")

                    if 'faces' in result and result['faces']:
                        face_tokens = [face['face_token'] for face in result['faces']]
                        print(f"[DEBUG] Retrieved Face Tokens: {face_tokens}")
                        return face_tokens
                else:
                    error_text = await response.text()
                    print(f"[DEBUG] Detect Faces Error: {response.status} - {error_text}")
        except Exception as e:
            print(f"[DEBUG] Detect Faces Exception: {e}")
    return None

# Function to analyze faces
async def analyze_faces_async(face_tokens, api_key, api_secret, session):
    async with semaphore:
        url = "https://api-us.faceplusplus.com/facepp/v3/face/analyze"

        # Prepare the request payload
        request_payload = {
            "api_key": api_key,
            "api_secret": api_secret,
            "face_tokens": ",".join(face_tokens),
            "return_attributes": "emotion"  # Valid attributes
        }

        print(f"[DEBUG] Sending Face Tokens for Analysis: {face_tokens}")

        try:
            async with session.post(url, data=request_payload) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"[DEBUG] Analyze Faces Response: {result}")
                    return result
                else:
                    error_text = await response.text()
                    print(f"[DEBUG] Analyze Faces Error: {response.status} - {error_text}")
        except Exception as e:
            print(f"[DEBUG] Analyze Faces Exception: {e}")
    return None

# Retry mechanism with exponential backoff
async def retry_on_error(coro_func, retries=3, delay=2):
    for attempt in range(retries):
        try:
            return await coro_func()
        except Exception as e:
            print(f"[DEBUG] Retry ({attempt + 1}/{retries}) after error: {e}")
            await asyncio.sleep(delay * (2 ** attempt))
    print("[DEBUG] All retries failed.")
    return None

# Video processing function
async def video_emotion_analysis(api_key, api_secret):
    cap = cv2.VideoCapture(0)
    frame_count = 0
    frame_interval = 60  # Process one frame every 60 frames (approximately 2 seconds for 30 FPS)

    async with ClientSession() as session:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame. Exiting...")
                break

            frame = cv2.resize(frame, (320, 240))
            frame_count += 1
            emotion_text = "Analyzing..."

            # Process frames only at the specified interval
            if frame_count % frame_interval == 0:
                print("[DEBUG] Processing frame...")
                face_tokens = await retry_on_error(
                    lambda: detect_faces_async(frame, api_key, api_secret, session)
                )

                if face_tokens:
                    analyze_result = await retry_on_error(
                        lambda: analyze_faces_async(face_tokens, api_key, api_secret, session)
                    )

                    if analyze_result and 'faces' in analyze_result:
                        emotions = analyze_result['faces'][0]['attributes']['emotion']
                        dominant_emotion = max(emotions, key=emotions.get)
                        emotion_text = f"Emotion: {dominant_emotion}"

                        # Pass to OpenAI GPT
                        gpt_message = {"role": "user", "content": emotion_text}
                        messages.append(gpt_message)

                        try:
                            response = client.chat.completions.create(
                                model="GPT-4",
                                messages=messages
                            )
                            gpt_response = response.choices[0].message.content
                            print("[DEBUG] GPT Response:", gpt_response)

                        except Exception as e:
                            print(f"[DEBUG] GPT API Exception: {e}")
                    else:
                        emotion_text = "No emotion data available"
                else:
                    emotion_text = "No faces detected"

                # Introduce a delay to slow down processing
                await asyncio.sleep(2)  # Add a 2-second delay between detections

            # Display emotion text on the video feed
            cv2.putText(frame, emotion_text, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Emotion Analysis", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


# Main block
if __name__ == "__main__":
    api_key_file = "api_key.txt"
    api_secret_file = "api_secret.txt"
    api_key, api_secret = read_credentials(api_key_file, api_secret_file)

    asyncio.run(video_emotion_analysis(api_key, api_secret))
