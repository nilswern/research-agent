' Start ResearchPilot without a terminal window (double-click).
' Stop it with the Quit button in the web UI.
Option Explicit

Dim fso, shell, root, python
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

root = fso.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = root

python = root & "\.venv\Scripts\python.exe"
If Not fso.FileExists(python) Then python = "python"

' Window style 0 = hidden, False = do not wait for it to finish.
shell.Run """" & python & """ main.py --web --port 8000", 0, False
