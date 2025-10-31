# import the necessary libraries
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="Y4FNUG8Esbj83F65kkBA"
)

# infer on a local image
result = CLIENT.infer("object-detection-main/ds2.jpg", model_id="ss-uniform/3")

# Load the image
image = Image.open("object-detection-main/ds2.jpg")
draw = ImageDraw.Draw(image)

# Parse and draw predictions
for prediction in result['predictions']:
    # Get the center, width, and height
    x_center = prediction['x']
    y_center = prediction['y']
    width = prediction['width']
    height = prediction['height']
    
    # Calculate the top-left and bottom-right coordinates
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    
    label = prediction['class']
    score = prediction['confidence']

    # Draw bounding box and label
    draw.rectangle(((x1, y1), (x2, y2)), outline="green", width=2)
    draw.text((x1, y1 - 10), f"{label} ({score:.2f})", fill="red")

# Display the image with predictions
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis("off")
plt.show()
