Installation Guide (Required Library: DeepFace)

This project requires DeepFace, a powerful facial recognition and analysis framework.
Follow the steps below to install all necessary dependencies before running the project.

1. Install Python (Recommended: Python 3.8–3.11)

Make sure you have a compatible Python version.
Check your Python version:

python --version

2. Create a Virtual Environment (Optional but Recommended)
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows

3. Install Required Library: DeepFace

To install DeepFace along with all dependencies (TensorFlow, Keras, OpenCV, etc.):

pip install deepface


This command installs:

DeepFace

TensorFlow (or PyTorch depending on environment)

OpenCV

NumPy

Pandas

RetinaFace / MTCNN (face detectors)

Other required utilities


4. Verify Installation

Run the following script to confirm everything works:

from deepface import DeepFace

result = DeepFace.analyze("your_image.jpg", actions=["emotion"])
print(result)


If no error appears, DeepFace is successfully installed.

You’re Ready to Start!

DeepFace is now installed and configured.
You can continue with running the project or exploring its features.
