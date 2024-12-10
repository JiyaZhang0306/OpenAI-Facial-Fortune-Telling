const video = document.getElementById("video");
const canvas = document.createElement("canvas");
const emotionDisplay = document.getElementById("emotion-display");

// Access user's camera
navigator.mediaDevices
  .getUserMedia({ video: true })
  .then((stream) => {
    video.srcObject = stream;
    video.play();
  })
  .catch((err) => {
    console.error("Error accessing camera:", err);
  });

// Capture frame and send to Flask
async function captureFrame() {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  // Convert frame to Base64
  const frameData = canvas.toDataURL("image/jpeg");

  try {
    const response = await fetch("http://127.0.0.1:5000/emotion-analysis", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ frame: frameData }),
    });

    const data = await response.json();
    if (data.emotions) {
      emotionDisplay.innerText = `Dominant Emotion: ${data.dominant_emotion}`;
    } else {
      emotionDisplay.innerText = "No emotions detected.";
    }
  } catch (err) {
    console.error("Error during emotion analysis:", err);
    emotionDisplay.innerText = "Error analyzing emotions.";
  }
}

// Capture a frame every 2 seconds
setInterval(captureFrame, 2000);
