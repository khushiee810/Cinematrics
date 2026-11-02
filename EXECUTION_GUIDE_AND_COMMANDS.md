# 🚀 CineMetrics: Step-by-Step Execution Guide & Command Reference

This document provides the complete, copy-pasteable command reference for setting up, running, testing, and managing the **CineMetrics** platform on your machine (Windows, macOS, and Linux).

---

## 📑 Table of Contents
1. [Prerequisites & System Verification](#1-prerequisites--system-verification)
2. [Project Setup & Virtual Environment](#2-project-setup--virtual-environment)
3. [Installing Dependencies](#3-installing-dependencies)
4. [Running the Web Application (`app.py`)](#4-running-the-web-application-apppy)
5. [Running Machine Learning Evaluation (`ml_evaluation.py`)](#5-running-machine-learning-evaluation-ml_evaluationpy)
6. [Interactive Testing Walkthrough](#6-interactive-testing-walkthrough)
7. [Git & GitHub Commands Reference](#7-git--github-commands-reference)
8. [Troubleshooting & Common Fixes](#8-troubleshooting--common-fixes)
9. [Quick 30-Second Execution Summary](#9-quick-30-second-execution-summary)

---

## 1. Prerequisites & System Verification

Before executing the project, verify that Python (version 3.10 or higher) and Git are installed.

Open **PowerShell** (or Command Prompt / Terminal) and run:

```powershell
python --version
pip --version
git --version
```

> **Expected Output:**
> ```text
> Python 3.10.x (or higher, e.g. Python 3.14.x)
> pip 24.x.x
> git version 2.x.x
> ```

---

## 2. Project Setup & Virtual Environment

Navigate to the project folder on your Desktop:

### Step 2.1: Navigate to the Project Directory
```powershell
cd C:\Users\HP\Desktop\sentiment_analysis_app
```

### Step 2.2: Create a Virtual Environment
Isolating dependencies ensures no conflicts with other Python projects on your machine:

```powershell
python -m venv venv
```

### Step 2.3: Activate the Virtual Environment

- **On Windows (PowerShell):**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
  *(If you see a script execution error, run this one-time command first: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`)*

- **On Windows (Command Prompt `cmd`):**
  ```cmd
  venv\Scripts\activate.bat
  ```

- **On macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```

> **Verification:** Your terminal prompt will now show `(venv)` at the beginning of the line:
> ```text
> (venv) PS C:\Users\HP\Desktop\sentiment_analysis_app>
> ```

---

## 3. Installing Dependencies

Install all required packages (`Flask`, `nltk`, `scikit-learn`) pinned in `requirements.txt`:

```powershell
pip install -r requirements.txt
```

> **Verification:** Check that all libraries installed successfully:
> ```powershell
> pip list
> ```
> You should see `Flask`, `nltk`, and `scikit-learn` in the list.

---

## 4. Running the Web Application (`app.py`)

To launch the full interactive CineMetrics web application:

```powershell
python app.py
```

### Terminal Output:
```text
Starting Movie Analytics Hub on http://127.0.0.1:5000 ...
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
```

### Accessing the Web Interface:
Open your browser (Chrome, Edge, Firefox, Brave) and navigate to:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)** (or `http://localhost:5000`)

To stop the web server anytime, press <kbd>Ctrl</kbd> + <kbd>C</kbd> in your terminal.

---

## 5. Running Machine Learning Evaluation (`ml_evaluation.py`)

CineMetrics includes a standalone machine learning module that trains and evaluates **Supervised SVM** and **Unsupervised K-Means** strictly on **Precision**, **Recall**, and **F1-Score**.

Open a new terminal window in the project folder and run:

```powershell
cd C:\Users\HP\Desktop\sentiment_analysis_app
python ml_evaluation.py
```

### Expected Terminal Output:
```text
=================================================================
CINEMETRICS MACHINE LEARNING EVALUATION (Dataset: 200 reviews)
=================================================================
Class Distribution: Counter({'Positive': 126, 'Neutral': 44, 'Negative': 30})
-----------------------------------------------------------------

[1] SUPERVISED MACHINE LEARNING: Support Vector Machine (SVM)
Class          | Precision  | Recall     | F1-Score  
----------------------------------------------------
Positive       | 1.00       | 1.00       | 1.00      
Neutral        | 1.00       | 1.00       | 1.00      
Negative       | 1.00       | 1.00       | 1.00      
----------------------------------------------------
Overall (Macro)| 1.00       | 1.00       | 1.00      

[2] UNSUPERVISED MACHINE LEARNING: K-Means Clustering (K=3)
Class          | Precision  | Recall     | F1-Score  
----------------------------------------------------
Positive       | 0.70       | 1.00       | 0.82      
Neutral        | 1.00       | 0.43       | 0.60      
Negative       | 0.00       | 0.00       | 0.00      
----------------------------------------------------
Overall (Macro)| 0.57       | 0.48       | 0.47      
=================================================================
```

---

## 6. Interactive Testing Walkthrough

Once `python app.py` is running, test these 5 real-world scenarios in your browser:

### Test Case 1: Horror Domain Praise (Solving Vanilla NLP Misclassification)
1. Type into the review box:
   ```text
   Terrifying horror with incredible jump scares and creepy monsters!
   ```
2. Click **"Analyze Review"** (or use the *"👻 Test Horror"* preset button).
3. **Observed Result:** 
   - **Sentiment:** `Positive 😊` (Compound Score: `+0.89`)
   - **Genre Detected:** `Horror 👻` (Confidence: `94%`, Clues: `horror`, `jump scares`, `terrifying`)

---

### Test Case 2: Sarcasm & Irony Inversion
1. Type into the review box:
   ```text
   Oh great, another masterpiece if you have no brain.
   ```
2. Click **"Analyze Review"** (or use the *"🙃 Test Sarcasm"* preset button).
3. **Observed Result:** 
   - **Sentiment:** `Negative (Sarcastic) 🙃` (Compound Score: `-0.55`)
   - **Explanation:** *"Sarcasm & irony detected in phrasing ('conditional irony'). True sentiment inverted to Negative."*

---

### Test Case 3: Genuine Horror Criticism (Guard Clause Active)
1. Type into the review box:
   ```text
   Cheap jump scares and boring plot. Not scary at all, complete waste of time.
   ```
2. Click **"Analyze Review"**.
3. **Observed Result:** 
   - **Sentiment:** `Negative 😞` (Compound Score: `-0.72`)
   - **Explanation:** Critical negative guard words (*"waste of time"*, *"not scary"*, *"boring"*) block the horror boost, preserving true criticism.

---

### Test Case 4: Movie ID Intelligence Hub & Instant Archive Filtering
1. Type into the review box (or the Movie ID filter box):
   ```text
   M2024_01
   ```
2. Click **"Analyze Review"**.
3. **Observed Result:**
   - The top display immediately switches to the **Movie Intelligence Hub** showing:
     - Film Genre: `Sci-Fi 🚀`
     - Release Year: `2024`
     - Box Office Gross: `$29.8M`
     - Star Rating: `8.5 / 10`
     - Sentiment Consensus: `75% Positive, 25% Neutral`
     - Full review list for `M2024_01`.
   - The **Decade Film Archive Table** below automatically filters in real time to show only rows for `M2024_01`.
   - A glowing **"Show All Archive (200)"** reset button appears to restore all rows.

---

### Test Case 5: Testing via REST API (cURL / Postman / Python)
You can also interact with the server programmatically using `curl` or Python requests:

```powershell
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"review_text\": \"Absolute masterpiece with mind-bending visuals!\"}"
```

**JSON Response:**
```json
{
  "badge_label": "Positive",
  "color": "success",
  "compound": 0.9124,
  "emoji": "😊",
  "genre_clues": ["mind-bending", "masterpiece"],
  "genre_confidence": "92%",
  "genre_detected": "Sci-Fi",
  "genre_emoji": "🚀",
  "is_movie_id": false,
  "is_sarcastic": false,
  "sentiment": "Positive"
}
```

---

## 7. Git & GitHub Commands Reference

Your project is already initialized with Git and connected to your GitHub repository:
👉 **`https://github.com/khushiee810/Cinematrics`**

### Check Git Status
```powershell
cd C:\Users\HP\Desktop\sentiment_analysis_app
git status
```

### Commit New Changes and Push to GitHub
```powershell
git add .
git commit -m "docs: Update execution guide and test scenarios"
git push origin main
```

### Pull Latest Code from GitHub
```powershell
git pull origin main
```

---

## 8. Troubleshooting & Common Fixes

| Problem | Cause | Exact Fix Command |
|---|---|---|
| **`Execution of scripts is disabled on this system`** | Windows PowerShell security policy blocks virtual environment script. | Run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`<br>Then reactivate: `.\venv\Scripts\Activate.ps1` |
| **`Address already in use: Port 5000`** | Another application or previous Flask process is still holding port 5000. | Find PID: `netstat -ano \| findstr :5000`<br>Kill it: `taskkill /PID <PID_NUMBER> /F` |
| **`ModuleNotFoundError: No module named 'flask'`** | Virtual environment is not activated or dependencies not installed. | Run: `pip install -r requirements.txt` |
| **`remote: Repository not found`** | Wrong GitHub URL or repository not created on GitHub. | Update remote URL: `git remote set-url origin https://github.com/khushiee810/Cinematrics.git` |

---

## 9. Quick 30-Second Execution Summary

Whenever you want to start the project quickly:

```powershell
# 1. Open project folder
cd C:\Users\HP\Desktop\sentiment_analysis_app

# 2. Run Web Server
python app.py

# 3. Open in Browser:
# http://127.0.0.1:5000

# 4. In a second terminal, run ML evaluation:
python ml_evaluation.py
```

---
*CineMetrics is created and maintained by Khushi Gupta.*
