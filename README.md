This project is a lightweight, low-latency real-time computer vision system powered by MediaPipe and OpenCV. It implements accurate facial state analysis and multi-type hand gesture recognition through webcam video stream processing, with full keypoint visualization and stability optimization.

Project Features:\n
• Real-time facial state detection: eye open/closed recognition based on the EAR algorithm
• Real-time facial state detection: mouth open/closed recognition based on the MAR algorithm
• Comprehensive hand gesture recognition: support 13 predefined gestures (digits 0–9, phone call, shush, fist, open palm)
• Dual-hand detection: simultaneously detect and recognize two hands in the camera view
• Automatic left/right hand identification and labeling
• Real-time visualization of facial mesh keypoints and finger joint states
• Custom anti-shake mechanism: valid recognition only after 0.3s stable hold to reduce jitter errors
• Automatic caching of the latest 20 recognition results for record review
• Real-time FPS monitoring and integrated visual information panel
• Built-in gesture guide panel and full Chinese UI display
• Low-latency inference, suitable for real-time interactive scenarios

Repository Contents\n
This repository contains all complete development, demonstration, and documentation resources for the project:
• Full Python source code for facial expression and hand gesture detection
• Core algorithm implementation (EAR/MAR calculation, gesture logic, anti-shake filtering)
• Static HTML official webpage: personal introduction, project overview, and design ideas
• Embedded online functional demo display on the webpage
• Original project demo video assets
• Complete project documentation and usage guidelines

Tech Stack:\n
• Python: Main programming language for algorithm and logic implementation
• MediaPipe (Face Mesh + Hands): Provides high-precision facial and hand keypoint detection
• OpenCV: Webcam capture, real-time image rendering, and UI display
• NumPy: Numerical calculation for EAR/MAR threshold judgment
• HTML: Build project introduction website and online demo page
