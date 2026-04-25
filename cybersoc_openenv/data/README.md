# Curated Alert Reference Data

`curated_alerts.json` is a small hand-authored SOC alert dataset aligned to the main CyberSOC attack families:

- phishing and malicious email delivery
- malware beacons and ransomware staging
- token replay and identity abuse
- lateral movement and credential theft
- data exfiltration
- analyst false positives and benign automation noise

The dataset is intended for:

- offline evaluation
- few-shot prompt examples for `inference.py`
- demos and explainability walkthroughs

It is not a model-training corpus by itself. It is a structured reference set that complements the stateful OpenEnv scenarios.
