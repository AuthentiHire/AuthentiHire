
#  AuthentiHire 

AuthentiHire is an **AI-driven interview integrity and analytics platform** designed to ensure fairness, transparency, and authenticity in virtual interviews. It leverages **computer vision, audio processing, and deep learning** to detect suspicious behavior, deepfakes, and integrity violations during online interviews.

---

## 📌 Problem Statement
With the rapid adoption of remote hiring, organizations face challenges such as:
- Candidate impersonation
- Use of deepfake video and audio
- Lack of behavioral monitoring
- Difficulty ensuring unbiased and fair interviews

AuthentiHire addresses these challenges by providing **real-time monitoring and post-interview analytics**.

---

##  Features
- **Gaze Shift Detection** – Identifies abnormal eye movements indicating distractions or external assistance  
-  **Emotion Detection** – Analyzes facial expressions to detect emotional states during interviews  
- **Multiple Person Detection** – Detects presence of more than one person in the video frame  
- **Video Deepfake Detection** – Flags manipulated or synthetic video content  
- **Audio Deepfake Detection** – Analyzes voice signals to detect synthetic or altered audio  
-  **Interview Analytics Report** – Generates detailed analytics using a unique meeting ID  
- **Bias Reduction Support** – Helps promote fair and transparent interview evaluations  

---

## 🛠️ Tech Stack

### Backend & Frameworks
- Flask
  
### AI / Machine Learning & Computer Vision
- YOLO  
- XceptionNet  
- DeepFace  
- OpenCV  

### Audio Processing
- Librosa  

### Data & Utilities
- NumPy  
- Pandas  
- SQLite  

### Frontend
- HTML  
- CSS  
- JavaScript  

---

## System Workflow
1. Candidate joins an interview session  
2. Video and audio streams are captured  
3. AI models analyze:
   - Gaze patterns  
   - Facial emotions  
   - Presence of multiple people  
   - Deepfake indicators  
4. Audio is processed for deepfake detection  
5. Analytics are generated and stored using a **meeting ID**  
6. Recruiters receive a comprehensive integrity report  

---

## Output & Analytics
- Integrity score  
- Emotion timeline  
- Gaze deviation metrics  
- Deepfake confidence scores  
- Multi-person detection alerts  

---

##  Use Cases
- Online recruitment interviews  
- Remote technical assessments  
- Academic viva voce and evaluations  
- Secure virtual meetings  

---

## Project Status
- Prototype completed  
- Continuous improvements in model accuracy and UI  

## Repository
- https://github.com/AuthentiHire/AuthentiHire


