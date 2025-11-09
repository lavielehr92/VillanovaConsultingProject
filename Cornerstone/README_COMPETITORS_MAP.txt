Quick Start — Competitors Map (Google Maps API)
===============================================

Files created:
- map_competitors.py
- competitors.csv  (sample you can edit/add rows)

Steps
-----
1) Open a terminal in this folder and install deps:
   pip install requests pandas

2) Get a Google Maps API key (enable Geocoding API and Maps JavaScript API) and set it:
   - macOS/Linux:  export GOOGLE_MAPS_API_KEY=YOUR_KEY_HERE
   - Windows CMD:  set GOOGLE_MAPS_API_KEY=YOUR_KEY_HERE
   - Windows PowerShell:  $Env:GOOGLE_MAPS_API_KEY="YOUR_KEY_HERE"

3) Put competitor schools into competitors.csv.
   Columns: name,address,city,state,zip,type,website,notes

4) Run the script:
   python map_competitors.py --csv competitors.csv --out competitors_map.html

5) Open the generated HTML file in your browser:
   /mnt/data/competitors_map.html

Tips
----
- The script caches geocodes in geocode_cache.json to limit billable lookups.
- If you already have lat/lng columns in the CSV, use --skip-geocode.
- To center the initial map yourself, pass --center 39.94,-75.23 and --zoom 11.