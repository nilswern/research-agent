' ResearchPilot ohne Terminal-Fenster starten (Doppelklick).
' Beenden ueber den "Beenden"-Button in der Weboberflaeche.
Option Explicit

Dim fso, shell, root, python
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

root = fso.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = root

python = root & "\.venv\Scripts\python.exe"
If Not fso.FileExists(python) Then python = "python"

' Fensterstil 0 = unsichtbar, False = nicht auf das Ende warten.
shell.Run """" & python & """ main.py --web --port 8000", 0, False
