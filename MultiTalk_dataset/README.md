## Overview
MultiTalk dataset is a new multilingual 2D video dataset featuring over 420 hours of talking videos across 20 languages. 
It contains 293,812 clips with a resolution of 512x512, a frame rate of 25 fps, and an average duration of 5.19 seconds per clip.
The dataset shows a balanced distribution across languages, with each language representing between 2.0% and 9.7% of the total.  

<img alt="statistic" src="../assets/statistic.png" width=560>


<details><summary><b>Detailed statistics</b></summary><p>

| Language | Total Duration(h) | #Clips | Avg. Duration(s) |                                                     Annotation                                                      |
|:---:|:---:|:---:|:---:|:-------------------------------------------------------------------------------------------------------------------:|
| Arabic | 10.32 | 9048 | 4.11 |     [arabic.json](https://github.com/postech-ami/MultiTalk/tree/main/MultiTalk_dataset/annotations/arabic.json)     |
| Catalan | 41.0 |  29232 | 5.05 |    [catalan.json](https://github.com/postech-ami/MultiTalk/tree/main/MultiTalk_dataset/annotations/catalan.json)    |
| Croatian | 41.0 |  25465 | 5.80 |   [croatian.json](https://github.com/postech-ami/MultiTalk/tree/main/MultiTalk_dataset/annotations/croatian.json)   |
| Czech | 18.9 | 11228 | 6.06 |      [czech.json](https://github.com/postech-ami/MultiTalk/tree/main/MultiTalk_dataset/annotations/czech.json)      |
| Dutch | 17.05 | 14187 | 4.33 |      [dutch.json](https://github.com/postech-ami/MultiTalk/tree/main/MultiTalk_dataset/annotations/dutch.json)      |
| English | 15.49 |  11082 | 5.03 |    [english.json](https://github.com/postech-ami/MultiTalk/tree/main/MultiTalk_dataset/annotations/english.json)    |
| French | 13.17 |  11576 | 4.10 |     [french.json](https://github.com/postech-ami/MultiTalk/tree/main/MultiTalk_dataset/annotations/french.json)     |
| German | 16.25 | 10856 | 5.39 |     [german.json](https://github.com/postech-ami/MultiTalk/tree/main/MultiTalk_dataset/annotations/german.json)     |
| Greek | 17.53 | 12698 | 4.97 |      [greek.json](https://github.com/postech-ami/MultiTalk/tree/main/MultiTalk_dataset/annotations/greek.json)      |
| Hindi | 24.41 | 16120 | 5.45 |      [hindi.json](https://github.com/postech-ami/MultiTalk/tree/main/MultiTalk_dataset/annotations/hindi.json)      |
| Italian | 13.59 | 9753 | 5.02 |    [italian.json](https://github.com/postech-ami/MultiTalk/tree/main/MultiTalk_dataset/annotations/italian.json)    |
| Japanese | 8.36 | 5990 | 5.03 |   [japanese.json](https://github.com/postech-ami/MultiTalk/tree/main/MultiTalk_dataset/annotations/japanese.json)   |
| Mandarin | 8.73 | 6096 | 5.15 |   [mandarin.json](https://github.com/postech-ami/MultiTalk/tree/main/MultiTalk_dataset/annotations/mandarin.json)   |
| Polish | 21.58 | 15181 | 5.12 |     [polish.json](https://github.com/postech-ami/MultiTalk/tree/main/MultiTalk_dataset/annotations/polish.json)     |
| Portuguese | 41.0 | 25321 | 5.83 | [portuguese.json](https://github.com/postech-ami/MultiTalk/tree/main/MultiTalk_dataset/annotations/portuguese.json) |
| Russian | 26.32 | 17811 | 5.32 |    [russian.json](https://github.com/postech-ami/MultiTalk/tree/main/MultiTalk_dataset/annotations/russian.json)    |
| Spanish | 23.65 | 18758 | 4.54 |    [spanish.json](https://github.com/postech-ami/MultiTalk/tree/main/MultiTalk_dataset/annotations/spanish.json)    |
| Thai | 10.95 | 7595 | 5.19 |       [thai.json](https://github.com/postech-ami/MultiTalk/tree/main/MultiTalk_dataset/annotations/thai.json)       |
| Turkish | 12.9 | 11165 | 4.16 |    [turkish.json](https://github.com/postech-ami/MultiTalk/tree/main/MultiTalk_dataset/annotations/turkish.json)    |
| Ukrainian | 41.0 | 24650 | 5.99 |  [ukrainian.json](https://github.com/postech-ami/MultiTalk/tree/main/MultiTalk_dataset/annotations/ukrainian.json)  |
</p></details>

## Download

### Usage
**Prepare the environment:**
```bash
pip install pytube
pip install opencv-python
```

**Run script:** 
```bash
cd MultiTalk_Dataset
``` 
You can pass the languages you want to download as arguments to the script. If you want to download all 20 languages, run the following script.  
```bash
sh dataset.sh arabic catalan croatian czech dutch english french german greek hindi italian japanese mandarin polish portuguese russian spanish thai turkish ukrainian
```

After downloading, the folder structure will be as below. Each language folder contains the .mp4 videos.  
You can change the ${ROOT} folder in the [code](https://github.com/postech-ami/MultiTalk/tree/main/MultiTalk_dataset/download_and_process.py).
```
    ${ROOT}
    ├── multitalk_dataset        # MultiTalk Dataset
    │   ├── arabic
    │   │   ├── O-VJXuHb390_0.mp4
    │   │   ├── O-VJXuHb390_1.mp4
    │   │   ├── ...
    │   │   └── ...             
    │   ├── catalan                                        
    │   ├── ...          
    │   └── ...                 
    └── raw_video              # Original videos (you can remove this directory after downloading)
        ├── arabic              
        ├── catalan                          
        ├── ...          
        └── ...
```

### JSON File Structure
```javascript
{
    "QrDZjUeiUwc_0":  // clip 1 
    {
        "youtube_id": "QrDZjUeiUwc",                                // youtube id
        "duration": {"start_sec": 302.0, "end_sec": 305.56},        // start and end times in the original video
        "bbox": {"top": 0.0, "bottom": 0.8167, "left": 0.4484, "right": 0.9453},  // bounding box
        "language": "czech",                                        // language
        "transcript": "já jsem v podstatě obnovil svůj list z minulého roku"      // transcript
    },
    "QrDZjUeiUwc_1":  // clip 2 
    {
        "youtube_id": "QrDZjUeiUwc",                                
        "duration": {"start_sec": 0.12, "end_sec": 4.12},        
        "bbox": {"top": 0.0097, "bottom": 0.55, "left": 0.3406, "right": 0.6398},  
        "language": "czech",                                       
        "transcript": "ahoj tady anička a vítejte u dalšího easycheck videa"      
    }
    "..."
    "..."

}
```
