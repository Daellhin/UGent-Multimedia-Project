# Project multimedia

## Dependencies installeren
`Standaard dependencies:` getest onder python 3.13
```bash
# Create virtual environment
python3 -m venv .venv
# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
.venv\Scripts\activate   # On Windows
# Install dependencies
pip install -r requirements.txt
```

## Setup
> Voordat de main code kan gerund worden moeten de video files toegevoegd worden. Deze zijn niet in de repo toegevoegd door hun grote (Max file size is 100Mb op Github)

[Onedrive link naar video files](https://1drv.ms/f/c/84b5fd901f3c5eab/Ehs1BtV6Uz9Op9xOuB16UlcB3Cuzgyr_4kZ3g4k5oOIjqw?e=ruYD5C), deze moeten gedownload worden en de 3 folders(ArchiveVideos, DegradedVideos, SourceVideos) in de root folder steken (zelfde niveau als src)

## Running
`src` folder bevat de finale project code, hierbij kan het main bestand gebruikt worden om het programma te runnen. Om geen problemen met relatieve paden te hebben gebruik de `Run main.py` launch configuratie (Shortcut F5)
- Video file is verwerkt met video en audio appart, deze worden dan tijdelijk naar `output-temp` geschreven, daarna worden deze gecombineerd en in `output-final` geschreven

`Project` folder beat code geschreven bij het ontwikkelen van het finale programma. Deze zijn niet gegarandeerd om te werken of goede output te geven
