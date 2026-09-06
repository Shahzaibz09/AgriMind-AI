# Demo Sample Images

Clearly-labeled demo examples for live hackathon demos of the AgriMind AI
Streamlit app. Loaded by the "Try a demo image" button in `app/app.py`.

Each file is a single, unmodified copy of an existing repository image
(selected because it passes the quality guard, the crop verification gate,
and is classified correctly with high confidence):

| File | Source (single copy) | Expected result |
|------|----------------------|-----------------|
| `tomato_demo.JPG` | `data/raw/tomato/Tomato___Early_blight/0034a551-...-RS_Erly.B 9432.JPG` | Tomato · Early Blight (~98.6%) |
| `potato_demo.JPG` | `data/raw/potato/Potato___Late_blight/0051e5e8-...-RS_LB 4640.JPG` | Potato · Late Blight (~100%) |
| `apple_demo.JPG` | `data/raw/apple/Apple___Apple_scab/01a66316-...-FREC_Scab 3003.JPG` | Apple · Apple Scab (~100%) |

Source images are part of the PlantVillage dataset already included in
this repository; no new data is introduced.
