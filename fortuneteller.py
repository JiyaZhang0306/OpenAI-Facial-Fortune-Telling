from openai import AzureOpenAI
import os
import requests
import json

import cv2
import asyncio
from aiohttp import ClientSession, FormData

# Semaphore to limit concurrent requests
semaphore = asyncio.Semaphore(1)

client = AzureOpenAI(
    api_key=os.getenv("AZURE_KEY"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_version="2023-10-01-preview"
)


# Function to read API credentials
def read_credentials(api_key_file, api_secret_file):
    with open(api_key_file, 'r') as file:
        api_key = file.read().strip()
    with open(api_secret_file, 'r') as file:
        api_secret = file.read().strip()
    return api_key, api_secret

# Function to detect faces
async def detect_faces_async(frame, api_key, api_secret, session):
    async with semaphore:  # Use semaphore to limit concurrency
        url = "https://api-us.faceplusplus.com/facepp/v3/detect"
        _, img_encoded = cv2.imencode('.jpg', frame)

        # Prepare the request
        form = FormData()
        form.add_field("image_file", img_encoded.tobytes(), filename="frame.jpg", content_type="image/jpeg")
        form.add_field("api_key", api_key)
        form.add_field("api_secret", api_secret)

        try:
            async with session.post(url, data=form, ssl=False) as response:
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
    async with semaphore:  # Use semaphore to limit concurrency
        url = "https://api-us.faceplusplus.com/facepp/v3/face/analyze"

        # Prepare the request with only valid attributes
        data = {
            "api_key": api_key,
            "api_secret": api_secret,
            "face_tokens": ",".join(face_tokens),
            "return_attributes": "emotion"  # Valid attributes
        }

        print(f"[DEBUG] Sending Face Tokens for Analysis: {face_tokens}")

        try:
            async with session.post(url, data=data, ssl=False) as response:
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
            return await coro_func()  # Properly call the coroutine
        except Exception as e:
            print(f"[DEBUG] Retry ({attempt + 1}/{retries}) after error: {e}")
            await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff
    print("[DEBUG] All retries failed.")
    return None

# Video processing function
async def video_emotion_analysis(api_key, api_secret):
    cap = cv2.VideoCapture(0)
    frame_count = 0
    frame_interval = 30  # Process one frame every 30 frames

    async with ClientSession() as session:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame. Exiting...")
                break

            # Resize frame for faster processing
            frame = cv2.resize(frame, (320, 240))
            frame_count += 1

            # Default text for display
            emotion_text = "Analyzing..."

            if frame_count % frame_interval == 0:
                print("[DEBUG] Processing frame...")
                
                # Step 1: Detect faces with retry
                face_tokens = await retry_on_error(
                    lambda: detect_faces_async(frame, api_key, api_secret, session)
                )

                if face_tokens:
                    await asyncio.sleep(1.0)  # Add a delay between requests

                    # Step 2: Analyze faces with retry
                    analyze_result = await retry_on_error(
                        lambda: analyze_faces_async(face_tokens, api_key, api_secret, session)
                    )

                    if analyze_result and 'faces' in analyze_result:
                        for face in analyze_result['faces']:
                            face_rectangle = face['face_rectangle']
                            attributes = face['attributes']

                            # Extract attributes
                            emotions = attributes.get('emotion', {})
                            dominant_emotion = max(emotions, key=emotions.get) if emotions else "N/A"

                            print(f"[DEBUG] Face Analysis - Emotion: {dominant_emotion}")

                            # Annotate the frame
                            cv2.rectangle(frame,
                                          (face_rectangle['left'], face_rectangle['top']),
                                          (face_rectangle['left'] + face_rectangle['width'],
                                           face_rectangle['top'] + face_rectangle['height']),
                                          (0, 255, 0), 2)
                            cv2.putText(frame, f"{dominant_emotion}: {emotions[dominant_emotion]:.2f}",
                                        (face_rectangle['left'], face_rectangle['top'] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                            # Update emotion text and interact with GPT
                            emotion_text = (
                                f"The dominant emotion detected is '{dominant_emotion}'. "
                                f"The detailed emotions are: {emotions}. "
                                "Please provide a personalized fortune-telling biscuit based on this information."
                            )

                            messages = [
                                 {"role": "system", "content": (
                                 "You are a cool fortune teller. Based on detected emotions from a person's facial expressions, "
                                 "offer a personalized fortune-telling biscuit. Examples:\n"
                                 )},
                                 {"role": "user", "content": emotion_text}
                            ]


                            try:
                                response = openai.ChatCompletion.create(
                                    engine="gpt-4",  # Specify the engine if needed
                                    messages=messages
                                )
                                chat_response = response['choices'][0]['message']['content']
                                print("[DEBUG] GPT Response Content:", chat_response)
                                return chat_response

                                # Append GPT response to maintain conversation history
                                messages.append({"role": "assistant", "content": chat_response})
                            except Exception as e:
                                print(f"[DEBUG] GPT API Exception: {e}")

            # Show the video feed
            cv2.imshow("Emotion Analysis", frame)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    cap.release()
    cv2.destroyAllWindows()

# Main function
if __name__ == "__main__":
    # Read API credentials
    api_key_file = "api_key.txt"
    api_secret_file = "api_secret.txt"
    api_key, api_secret = read_credentials(api_key_file, api_secret_file)

    # Run the video emotion analysis
    asyncio.run(video_emotion_analysis(api_key, api_secret))