# AgriMind AI — Mobile MVP (Step 1: UI Foundation)

Flutter Android app for AgriMind AI, the AI-powered crop health
intelligence tool built for the Alibaba Cloud AI Hackathon Pakistan 2026.

**Status: Step 1 — UI foundation only.** All screens, navigation, green
AgriMind branding and the trilingual language architecture (English /
اردو / Roman Urdu) are in place. No AI inference runs in the app yet:
analysis and assistant results are clearly marked placeholders until the
backend is connected in a later step.

## Screens

| Screen | Purpose |
|---|---|
| Splash | Branding + tagline, short transition to Home |
| Home | Explanation, primary action *Analyze Crop*, secondary *Ask AgriMind*, How-it-works, About |
| Crop Selection | 🍅 Tomato · 🥔 Potato · 🍎 Apple with clear selection state |
| Leaf Analysis | Take Photo / Choose from Gallery (placeholders) / Try Demo Image |
| Analysis | UI states for quality check → crop verification → disease analysis |
| Result | Crop, disease, confidence, status, Grad-CAM area, next step — placeholder-marked |
| Farmer Assistant | Question input + quick questions (symptoms / manage / prevent) |
| Language | English / اردو / Roman Urdu (Urdu switches the app to RTL) |

## Planned architecture (Step 2)

```text
Flutter  →  FastAPI  →  existing Python/PyTorch inference (src/)
```

`lib/services/analysis_service.dart` and
`lib/services/assistant_service.dart` define the interfaces the UI
depends on; the current demo implementations return clearly marked
placeholders. Swapping in the real backend requires no screen changes.

## Running

```bash
flutter pub get
flutter run          # on an Android device or emulator
flutter analyze
flutter test
flutter build apk --debug
```

The existing Streamlit web app in `../app/` and all model files are
untouched by this project.
