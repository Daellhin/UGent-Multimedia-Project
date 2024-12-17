OS: Windows 11

Versie controle:
    Python:         3.11.10
    Opencv:         4.10.0
    Numpy:          2.1.1
    Scipy:          1.14.1
    Matplotlib:     3.9.2
    Audioplayer:    0.6
	Skimage:        0.24.0
	Tqdm:           4.67.0
	Moviepy:	    2.0.0

IDE: PyCharm & VS Code

--------------------------------------------------------------------------------

Bestanden/Mappen:
output-final/
    map met de finale output video's

output-temp/
    map met de tijdelijke output video's (beelden en audio apart)

main.py
    Uitvoerbaar sequentieel script voor alle video's

audio.py
    Bestand met functies voor audio verwerking
    Wordt geïmporteerd in main.py

visualisations.py
    Bestand met functies voor het visualiseren van van data (spectogram, histogram, magnitude spectrum,...)
    Wordt geïmporteerd in audio.py

utils.py
    Algemene hulpfuncties
    Wordt geïmporteerd in visualisations.py

versies.py
    Deze code werd gebruikt voor het opvragen van de huidige versies voor de versie controle

start commando:  python main.py

--------------------------------------------------------------------------------
