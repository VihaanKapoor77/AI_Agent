# AI_Agent
A simple AI_Agent that can help you code.

# How to Use

1st run:
```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
To make the virtual environment and install the requirements.  

Then run:
```powershell
Copy-Item .env.example .env
notepad .env
```
To create your own `.env` file.  

Then open the file and replace:
`"GOOGLE_API_KEY=your_actual_key_here"` with your actual API key.  

Save it, then finally run:
```powershell
.\.venv\Scripts\python.exe agents.py
```

