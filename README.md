## Brief
Ok so we just use this platform to **test and later push our main code** here, for any changes i suggest adding ur work into a **personal folder with a short .md file** to explain what u did, just so we can keep track thanks!


## How to run Annotator

- 1. run `pip install -r requirements.txt` 
- 2. run `python app.py` 
- 3. follow the link shown in ur CLI
- 4. u will find an option to upload a folder with the **image pngs**, upload the one from the drive (or u can do a few out of them but not it takes a folder only) 
- 5. right click anywhere on the UI , in the tab that opens to the right click 'console' to see ur downloads being logged
- 6. to annotate click 2 end points ull see the binary mask next to it then click download
- 7. if there is no runway just click download without marking any dots
- 8. the binary masks will be stored in tiff and png in the respective local folders where ur app.py is in `generated_png` and `generated_tiffs`

####Note: Images with prefix 0 imply no airstrip, prefix 1 implies airstrip is present