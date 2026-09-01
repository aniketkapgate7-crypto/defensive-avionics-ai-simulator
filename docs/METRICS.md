# Evaluation Metrics

## Signal module

- Overall accuracy
- Macro-F1 to expose class imbalance
- Confusion matrix
- Accuracy and macro-F1 grouped by SNR
- Inference latency per batch

Do not report only the easiest SNR range. Clearly state all filtering rules.

## Policy module

- Mean and standard deviation of episode return
- Scenario-completion rate
- Mean episode length
- Abstract-resource efficiency
- Performance relative to a rule-based baseline
- Results across at least three fixed random seeds

## Vision module

- Precision and recall
- mAP@0.50 and mAP@0.50:0.95
- False alerts per minute on negative test videos
- Frames per second on the demonstration laptop
- Urgency-category stability across consecutive frames

## Integrated system

- End-to-end update latency
- Dropped-frame rate
- Stale-prediction rate
- Demo success rate across a fixed scenario suite

Every final figure must record the model version, configuration file, seed, and
test split used to generate it.
