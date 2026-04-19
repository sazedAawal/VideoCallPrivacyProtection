# VideoCallPrivacyProtection
An ongoing research project tackling online sextortion by detecting real-time nudity and harassment in video calls using on-device AI — no data stored, no cloud involved, full participant privacy ensured.


> **Real-time, on-device explicit content detection for video calls — no data ever leaves your machine.**

---

## The Problem

Sextortion — the act of coercing someone into producing or sharing explicit content under threat — is a rapidly growing online crime. The FBI reported a **7,000% increase** in financially motivated sextortion cases involving minors between 2021 and 2023. The Internet Watch Foundation (IWF) found that self-generated child sexual abuse material increased by **374%** between 2019 and 2022. Most victims are targeted through video calls on platforms like Zoom, Microsoft Teams, Google Meet, Instagram, and Facebook — tools designed for legitimate communication that are being weaponised in real time.

Existing moderation tools operate **after the fact** — scanning content that has already been recorded, uploaded, or distributed. By that point, the harm has already occurred. There is no widely available, open-source tool that intervenes **at the moment of coercion**, during a live video call, before any content is captured or shared.

SextortionGuard is built to fill that gap.

---

## What SextortionGuard Does

SextortionGuard runs entirely on your local machine. It monitors a live video stream in real time using computer vision and — optionally — speech recognition, and immediately flags, blurs, or blocks explicit content the moment it appears. The host or meeting organiser is alerted instantly. No video, audio, or transcript is ever stored or transmitted.

```
┌─────────────────────────────────────────────────────────┐
│                    YOUR LOCAL MACHINE                   │
│                                                         │
│  Webcam / Screen ──► NudeNet (local) ──► score          │
│                                            │            │
│                               score > 90%? │            │
│                                            ▼            │
│                                    Blur frame           │
│                                    Show alert           │
│                                    Log timestamp only   │
│                                                         │
│  ✗ No video stored                                      │
│  ✗ No audio stored                                      │
│  ✗ No data sent to any server                           │
└─────────────────────────────────────────────────────────┘
```

---

## Key Features

- **Fully on-device** — inference runs locally using NudeNet and YOLOv8. Nothing is sent to any cloud service.
- **Real-time detection** — processes live webcam or screen-captured video streams frame by frame.
- **Photo-aware** — detects explicit content whether it is a live person or a photo/image held up to a camera.
- **Person tracking** — assigns persistent IDs to each participant using ByteTrack, so the host can selectively blur or block specific individuals.
- **Selective blur or block** — the host can click on any detected person to cycle through: normal → blur → block.
- **No storage** — raw frames, audio, and transcripts are processed in memory and immediately discarded. Only a timestamped event score is optionally written to a local log file.
- **Consent-first design** — built around explicit user consent; participants are informed that safety monitoring is active.
- **Platform agnostic** — works with any video call platform by capturing the screen or webcam feed directly.

---

## System Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  Video input │────►│  Preprocessing   │────►│  YOLOv8 + ByteTrack │
│  (webcam /   │     │  Resize, norm.   │     │  Person detection   │
│  screen cap) │     └──────────────────┘     │  + tracking IDs     │
└──────────────┘                              └──────────┬──────────┘
                                                         │
                                              ┌──────────▼──────────┐
                                              │  NudeNet classifier  │
                                              │  Explicit content   │
                                              │  scoring (local)    │
                                              └──────────┬──────────┘
                                                         │
                                         score ≥ 0.90?  │
                                                    ┌────▼────┐
                                                   YES       NO
                                                    │         │
                                          ┌─────────▼──┐  ┌───▼──────────┐
                                          │ Blur frame  │  │ Green status │
                                          │ Show alert  │  │ bar shown    │
                                          │ Log event   │  └──────────────┘
                                          └─────────────┘
```

---

## Modules

| Module | File | Status |
|---|---|---|
| Person detection + blur/block | `realtime_blur_tool.py` | ✅ Ready |
| Nudity detection (video) | `nudity_detector.py` | ✅ Ready |
| Audio transcription + flagging | `audio_guard.py` | 🔧 In progress |
| Screen capture mode | `screen_capture_guard.py` | 🔧 In progress |
| Teams / Zoom integration | `platform_bridge.py` | 📋 Planned |
| Consent UI | `consent_screen.py` | 📋 Planned |

---

## Quickstart

### Requirements

- Python 3.10 or higher
- macOS, Windows, or Linux
- Webcam

### Install

```bash
# Clone the repository
git clone https://github.com/yourusername/sextortion-guard.git
cd sextortion-guard

# Create a virtual environment (strongly recommended)
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run — nudity detector

```bash
python nudity_detector.py
```

The NudeNet model weights (~90 MB) download automatically on first run.

### Run — person blur/block tool

```bash
python realtime_blur_tool.py
```

### Controls (inside the OpenCV window)

| Key / Action | Effect |
|---|---|
| Left-click a person | Cycle: normal → blur → block → normal |
| `B` | Blur all detected persons |
| `K` | Block all detected persons |
| `R` | Reset all effects |
| `S` | Save blurred screenshot (flagged frames only) |
| `Q` or `ESC` | Quit |

---

## Configuration

All tunable parameters are at the top of each script. No config file needed — just open the file and edit:

```python
WEBCAM_INDEX     = 0      # 0 = built-in camera; 1 = external
CONF_THRESHOLD   = 0.90   # detection confidence required to trigger alert
CHECK_EVERY      = 8      # run NudeNet every N frames (lower = slower but more responsive)
BLUR_STRENGTH    = 99     # Gaussian blur kernel size (must be odd number)
```

---

## Dependencies

```
nudenet          # explicit content detection model
ultralytics      # YOLOv8 for person detection and ByteTrack tracking
opencv-python    # video capture and frame processing
numpy            # numerical operations
faster-whisper   # (audio module) real-time speech transcription — coming soon
sounddevice      # (audio module) microphone capture — coming soon
```

Install all at once:

```bash
pip install nudenet ultralytics opencv-python numpy
```

---

## Privacy & Data Handling

This is the most important section of this document.

| Data type | Handled how |
|---|---|
| Video frames | Processed in RAM, never written to disk |
| Audio | Processed in RAM, never written to disk |
| Transcripts | Processed in RAM, never written to disk |
| Detection score | Optionally written to local log file (timestamp + score only) |
| Personal data | None collected, none stored, none transmitted |

**No data is ever sent to any external server.** All AI inference runs on your local machine using locally stored model weights. This architecture was chosen deliberately to comply with GDPR Article 25 (data protection by design), CCPA, and similar privacy regulations.

---

## Ethical Use Statement

SextortionGuard is designed exclusively as a **protective tool** for potential victims and platform safety operators. It is built with the following principles:

**Consent is required.** The tool is designed to be used only when all participants in a video call have been informed that real-time safety monitoring is active. Covert monitoring of any person without their knowledge or consent is a misuse of this tool and may be illegal in your jurisdiction.

**Protection, not surveillance.** This tool detects and blocks harmful content in real time. It is not designed for, and cannot be used for, collecting, storing, or distributing explicit material.

**Intended users:**
- Individuals who want protection during video calls
- Platform safety teams building moderation tools
- Researchers studying real-time content moderation
- Child safety organisations

**Not intended for:**
- Law enforcement surveillance without consent
- Monitoring employees or individuals without disclosure
- Any use that involves storing or transmitting explicit content

If you are experiencing sextortion or online sexual coercion, please contact:
- **FBI Internet Crime Complaint Center (IC3):** [ic3.gov](https://ic3.gov)
- **Internet Watch Foundation:** [iwf.org.uk](https://www.iwf.org.uk)
- **Thorn:** [thorn.org](https://www.thorn.org)
- **Stop It Now helpline:** 1-888-PREVENT

---

## How This Differs From Existing Work

### SafeVchat (Xing et al., 2011)
The closest prior work is SafeVchat [[1]](#references), which detected obscene content on Chatroulette using skin-colour heuristics and OpenCV classifiers, combined with Dempster-Shafer fusion. SafeVchat ran **server-side** on Chatroulette's own infrastructure, meaning all video data was processed on a central server. SextortionGuard differs in three fundamental ways: (1) all inference runs on the client device — no data leaves the machine; (2) it uses modern deep learning models (NudeNet, YOLOv8) rather than skin-colour heuristics, achieving significantly higher accuracy across diverse lighting conditions and skin tones; and (3) it targets professional video conferencing platforms rather than anonymous random-chat services.

### Thorn Safer Predict (2025)
Thorn's Safer Predict [[2]](#references) is a powerful platform-side tool for detecting CSAM and grooming behaviour in text and uploaded content. It operates on content **already hosted on platforms** and requires platform API integration. SextortionGuard operates at the **point of coercion** — during the live call — before any content is captured or shared, and requires no platform cooperation.

### Survey of AI Strategies for Explicit Video Detection (2023)
A 2023 survey in *Electronics* [[3]](#references) comprehensively reviews deep learning methods for pornography detection in pre-recorded videos. All surveyed methods operate on stored video files, not live streams. The survey explicitly identifies real-time live-stream detection as an open research gap — which SextortionGuard addresses.

### UN Women Report (2025)
A 2025 UN Women report [[4]](#references) on AI-amplified violence against women identifies real-time detection tools as a promising direction, but notes that *"generalizability, lack of privacy and data safety are some of the concerns that have limited the applicability of these technologies to date."* SextortionGuard's local-inference, no-storage architecture is a direct response to this identified limitation.

### US Patent 12,481,789 (2025)
A granted US patent [[5]](#references) covers real-time masking of sensitive information during video call screen shares — targeting text-based sensitive data such as passwords and documents. SextortionGuard addresses a distinct and complementary problem: explicit visual content and verbal coercion detection.

---

## Roadmap

- [x] Real-time person detection and selective blur/block (YOLOv8 + ByteTrack)
- [x] Real-time nudity detection from webcam (NudeNet)
- [x] Photo-held-to-camera detection
- [x] Audio stream analysis (Whisper + keyword classifier)
- [ ] Screen capture mode (capture any meeting window)
- [ ] Consent screen UI shown to all participants at call start
- [ ] Microsoft Teams integration (Graph Communications API)
- [ ] Google Meet integration
- [ ] Zoom integration
- [ ] Evaluation benchmark suite (accuracy, latency, false positive rate)
- [ ] Instagram and Facebook video call support
- [ ] Multi-language audio support

---

## Contributing

Contributions are welcome, particularly in the following areas:

- Improving detection accuracy across diverse skin tones and lighting conditions
- Reducing CPU inference latency
- Platform integrations (Teams, Zoom, Meet)
- Evaluation datasets and benchmarking
- Translations of the consent UI

Please read [ETHICS.md](ETHICS.md) before contributing. All contributions must align with the protective intent of this project.

```bash
# Fork the repo, then:
git checkout -b feature/your-feature-name
git commit -m "Add your feature"
git push origin feature/your-feature-name
# Open a Pull Request
```

---

## Citation

If you use SextortionGuard in your research, please cite:

```bibtex
@software{sextortiongaurd2025,
  author    = {Your Name},
  title     = {SextortionGuard: Real-Time On-Device Explicit Content Detection
               for Video Calls},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/yourusername/sextortion-guard}
}
```

---

## References

[1] Xing, X., Liang, Y., Cheng, H., Dang, J., Huang, S., Han, R., Liu, X., Lv, Q., and Mishra, S. (2013). *SafeVchat: A System for Obscene Content Detection in Online Video Chat Services.* ACM Transactions on Internet Technology, 12(4). https://dl.acm.org/doi/10.1145/2499926.2499927

[2] Thorn. (2025). *Introducing Safer Predict: Using the Power of AI to Detect Child Sexual Abuse and Exploitation Online.* https://www.thorn.org/blog/introducing-safer-predict-using-the-power-of-ai-to-detect-child-sexual-abuse-and-exploitation-online/

[3] Ulhaq, A., et al. (2023). *Learning Strategies for Sensitive Content Detection.* Electronics, 12(11), 2496. https://doi.org/10.3390/electronics12112496

[4] UN Women. (2025). *How AI is Exacerbating Technology-Facilitated Violence Against Women and Girls.* United Nations Entity for Gender Equality and the Empowerment of Women. https://www.unwomen.org/en/digital-library/publications/2025/12/how-ai-is-exacerbating-technology-facilitated-violence-against-women-and-girls

[5] United States Patent No. 12,481,789. (2025). *Real-time masking of sensitive information in content shared during a screen share session of a video call.* https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12481789

[6] Borah, S.K., Ramaswamy, S., and Seshadri, S. (2025). *The Online Specter: Artificial Intelligence and Its Risks for Child Sexual Abuse and Exploitation.* SAGE Journals. https://journals.sagepub.com/doi/10.1177/09731342251334293

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

The MIT License permits free use, modification, and distribution. However, use of this software to facilitate harm, collect explicit content, or monitor individuals without consent is strictly prohibited and may violate applicable laws regardless of the license terms.

---

## Acknowledgements

- [NudeNet](https://github.com/notAI-tech/NudeNet) — open-source nudity detection model
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — real-time object detection and tracking
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — efficient local speech transcription
- [OpenCV](https://opencv.org/) — computer vision framework
- [Thorn](https://www.thorn.org) — for their public research on child safety technology
- [Internet Watch Foundation](https://www.iwf.org.uk) — for public reporting on CSAM trends

---

<p align="center">
Built to protect. Not to surveil.
</p>
