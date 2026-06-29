
Skip to main content

Advertisement
Nature

    View all journals
    Search
    Log in

    Explore content
    About the journal
    Publish with us

    Sign up for alerts
    RSS feed

    nature articles article 

    Article
    Open access
    Published: 01 June 2026

Passive heart-rate monitoring during smartphone use in everyday life

    Shun Liao, Paolo Di Achille, Jiang Wu, Silviu Borac, Jonathan Wang, Xin Liu, Eric S. Teasley, Lawrence Cai, Yuzhe Yang, Yun Liu, Daniel McDuff, Hao-Wei Su, Brent Winslow, Anupam Pathak, Mark Malhotra, Shwetak Patel, James A. Taylor, Jameson K. Rogers & Ming-Zher Poh 

Nature (2026) Cite this article

    42k Accesses

    105 Altmetric

    Metrics details

Abstract

Resting heart rate (RHR) is a key biomarker of cardiovascular health and mortality1,2,3, but passively tracking it longitudinally generally requires a wearable device, limiting its availability. Here we present passive heart-rate monitoring (PHRM), a deep-learning system that uses facial video-based photoplethysmography for passive measurements of heart rate (HR) and RHR during everyday smartphone interactions. Our system was developed using 192,353 videos from 485 participants and validated on 162,546 videos from 211 participants in laboratory and free-living conditions, representing, to our knowledge, the largest validation study of its kind. PHRM outperformed state-of-the-art methods on our benchmarks. Compared with reference electrocardiograms, PHRM achieved a mean absolute percentage error (MAPE) lower than 10% for HR measurements across three skin-tone groups of light, medium and dark pigmentation, meeting industry accuracy standards; MAPE for each skin-tone group was non-inferior versus the others. Daily RHR measured by PHRM had a mean absolute error of less than five beats per minute, compared with a wearable HR tracker, and was associated with known risk factors for cardiovascular disease. These results highlight the potential of smartphones for enabling passive and equitable monitoring of heart health. To facilitate further research, we publicly release a large, annotated smartphone video dataset along with a pre-trained HR model.
Similar content being viewed by others
Prospective validation of smartphone-based heart rate and respiratory rate measurement algorithms
Article Open access 12 April 2022
HealthRing: Physiology Dataset for Health Sensing on Rings
Article Open access 10 June 2026
Monitoring long-term cardiac activity with contactless radio frequency signals
Article Open access 05 December 2024
Main

Heart rate (HR) is an important and dynamic vital sign that is influenced by numerous inputs4, and resting heart rate (RHR) is recognized as a biomarker and prognostic factor for overall mortality1,2,3. Longitudinal increases in RHR are associated with higher mortality and adverse cardiovascular events5,6,7. Measurement of RHR conventionally requires a sustained period of rest, which limits the practicality of evaluating long-term trajectories. However, the sensitivity of HR to various factors suggests that the cardiovascular system is better assessed through multiple daily measurements than through brief, standardized clinic-based measurements8,9,10. Daily average HR has been shown to be a strong independent predictor of all-cause mortality11, even more so than clinic-measured RHR, and consumer wearable devices typically derive a daily RHR by passively aggregating HR measurements during periods of rest throughout the day12. Daily RHR monitoring can provide insights into cardiovascular health and detect physiological changes linked to fitness levels or illness13,14,15. Nonetheless, the adoption of consumer wearables, while growing, remains limited, especially among those who are most likely to benefit from these health-monitoring technologies16. Given that smartphones are already ubiquitous—owned by 90% of US adults and 69% of people globally17, and used 144 times daily, on average18—they offer an attractive alternative for opportunistic HR measurements across the day during normal phone use. The blood volume pulse can be measured from a distance using a technique called video-based remote photoplethysmography (rPPG)19,20, which can measure HR21,22,23,24 and screen for irregular rhythms, such as atrial fibrillation25, through smartphone cameras. However, existing rPPG studies have small sample sizes, are limited to controlled environments and face generalizability issues in real-world conditions. Crucially, the accuracy of current rPPG methods is known to drop significantly for darker skin tones, owing to an increased concentration of melanin26. Similar concerns apply to other PPG-based devices, such as pulse oximeters, which has led to scrutiny and calls for diversity in validation studies from health governing bodies like the US Food and Drug Administration (FDA) and the UK National Health Service (NHS)27,28. Furthermore, as previous rPPG studies have mainly involved active HR measurements in situated conditions, there remains a need to address passive HR measurements during everyday phone use under unconstrained, free-living conditions.

In this study, we present a smartphone-based deep-learning system that enables passive measurements of both HR and daily RHR in the background during normal phone use (collectively referred to as passive heart-rate monitoring; PHRM). Compared with previous work, our system provides several advances. First, we validate its performance in a prospective study on a large and diverse set of videos (more than 162,000), collected in laboratory conditions as well as in free-living, real-world conditions using participants’ personal phones. Second, our system meets industry accuracy standards and achieves prespecified non-inferiority targets for people of all skin tones, demonstrating its potential for equitable HR monitoring. PHRM outperformed state-of-the-art methods on our benchmarks. Third, we show that PHRM-derived daily RHR also achieves prespecified levels of accuracy and is associated with well-established cardiovascular health metrics and risk factors. Finally, we publicly release both a pre-trained HR model and a large and diverse smartphone video dataset comprising all skin pigmentation groups with reference HR labels to facilitate further research.
Overview of the system

We designed and developed the PHRM system with two major components (Fig. 1). First, we constructed an end-to-end HR estimation module that takes as input a short (eight-second) video clip of the user’s face, performs video stabilization and preprocessing (by face cropping, resizing, interpolating and computing frame differences) and predicts HR along with a measure of confidence using an ensemble29 of computationally efficient temporal shift convolutional neural networks (TSCNNs)24. We introduced a deep-learning architecture that reframes HR estimation as a multi-class classification problem over a discretized range of biologically plausible HRs (40–180 beats per minute; bpm). This distributional output allows the model to express uncertainty. If there is a high degree of uncertainty in the model’s estimation of HR (for example, owing to extreme motion), the probability distribution flattens, whereas a regression model would be forced to output a single—probably erroneous—point estimate. Next, we designed an algorithm to derive daily RHR by aggregating the HR predictions throughout the day using the confidence of predictions and a Kalman filter. PHRM was designed to run passively in the background and automatically initiate video capture via the front-facing camera on a screen-unlock event.
Fig. 1: Overview, development and validation of the PHRM system.
Fig. 1: Overview, development and validation of the PHRM system.The alternative text for this image may have been generated using AI.
Full size image

a, In our research study with consented participants, after a screen-unlock event, PHRM passively captures, processes and analyses 8-s facial video clips using a deep neural network (DNN) to estimate HR and associated prediction confidence to determine whether the measurement is valid. To compute daily RHR, PHRM aggregates valid HR measurements from intermittent 8-s video clips throughout a single day and applies a Kalman filter to improve estimates. b, Workflow diagram of the studies used to develop and validate the PHRM system. We used data from five independent, prospective laboratory studies and a prospective free-living study.
Study populations

To develop and validate PHRM, we conducted a series of studies to acquire datasets comprising face videos and HR ground truth (Table 1). In all of our studies, we recruited for diversity across age, sex and skin-tone groups. We used the electrocardiogram (ECG) as the reference HR ground truth for both the laboratory-based and the free-living validation studies. In total, we collected 192,353 videos from 485 participants for PHRM development, and 162,546 videos from 211 participants for PHRM validation.
Table 1 Baseline characteristics of participants across studies
Full size table

First, we obtained data to train and tune PHRM from four separate studies performed in controlled laboratory settings (n = 26,423 videos from 357 participants). This data comprises a variety of lighting conditions and physiological states, including at rest, during various exercises and after exercise (Extended Data Table 1). To provide an external test set for model validation, we conducted a fifth, prospective laboratory study that enrolled 104 participants (n = 1,731 videos) and captured videos under 5 different lighting conditions and in both at-rest and post-exercise physiological states. The mean age in this external test set was 51.3 ± 14.8 years; 71 (68.3%) participants were female. We divided participants into three groups of skin pigmentation (Fitzpatrick I–III, Fitzpatrick IV–V and Fitzpatrick VI) by converting their objective individual topology angle (ITA)—as measured by a spectrocolorimeter at the cheeks and forehead—into Fitzpatrick skin types30. We specified these skin-tone groups to intentionally overrepresent participants of the darkest skin tones and ensure that models were developed that perform accurately for this group, a decision that aligned with the three skin pigmentation cohorts subsequently proposed by the FDA27. ITA values ranged from −73.48° to 88.81°, with 44 (42.3%), 25 (24.0%) and 35 (33.7%) participants in skin pigmentation group 1 (lightest), 2 (medium) and 3 (darkest), respectively.

Next, we conducted a prospective free-living study designed to passively record face videos during normal personal phone use in everyday life during an eight-day period. The detailed video recording protocol is provided in the Supplementary Information. We applied stratified sampling on the basis of age, sex, body mass index (BMI) and the Monk Skin Tone (MST) scale to split the free-living data at the participant level: data from 50% of the participants (n = 165,930 videos from 128 participants) were set aside for model development (30% for training and 20% for tuning), and data from the remaining 50% of participants (n = 160,815 videos from 107 participants) were set aside as the test split for validation. We switched to using the MST in the prospective free-living study because it was designed to be more inclusive of the spectrum of skin tones that we see in our society (the laboratory studies took place before the introduction of MST and used the Fitzpatrick scale, the de facto industry standard at that time). The mean age in the test split of the free-living study was 37.9 ± 11.4 years; 57 (53.3%) participants were female. Following the FDA’s proposal, the entire range of skin pigmentation based on the self-reported MST was represented with at least one participant for each MST value of 1–10. We divided participants into three MST cohorts, yielding 39 (36.4%), 29 (27.1%) and 39 (36.4%) participants in the MST 1–4, MST 5–7 and MST 8–10 cohorts, respectively. This distribution also fulfilled the FDA recommendations to have at least 40% of each sex, and at least 25% of participants in each of the three MST cohorts. Six individuals in the test split did not meet the minimum adherence criteria (that is, at least 3 days with more than 40 video clips per day; Supplementary Fig. 2), yielding 101 participants (n = 158,471 videos) for our final analysis of free-living performance.

Participants uploaded 230.7 ± 172.2 face videos per day. The distribution of the video upload rate per participant was strongly left-skewed (Extended Data Fig. 2); most participants uploaded a high proportion of their videos (mode = 95%, median = 84.4% and interquartile range (IQR) = 22.9%). These videos were recorded passively throughout the day during normal personal phone use after a screen-unlock event. As expected, the unconstrained nature of free-living use and passive recordings yielded videos with a diversity of environments, lighting conditions, camera angles and face coverings (Fig. 2a). These videos spanned all hours of the day, and a wide range of lux and smartphone motion levels, as measured by the smartphone ambient light sensor and accelerometer, respectively (Fig. 2c). Illuminance measurements captured by the ambient light sensor spanned the full dynamic range of daily life (Supplementary Table 7). Although most of the recordings occurred under typical indoor lighting categorized as dim (45.6%) or bright (32.4%), substantial subsets captured challenging extremes, including dark conditions (14.2%) and outdoor environments (7.8%). We randomly sampled skin patches from video-frame crops of participant’s cheeks to visualize the range of skin pigmentation under various lighting conditions across the MST range (Fig. 2b). Concurrently, smartphone sensors characterized a broad spectrum of user behaviour (Supplementary Table 8). Android activity recognition classified the majority of videos as being recorded while users were still (79.7%), followed by walking (16.0%), being in a vehicle (4.1%) and running (0.3%). To assess ecological representativeness, we benchmarked these distributions against a large (n = 10,155) independent dataset of US adults showing typical patterns of smartphone use31. Although the requirement for active screen engagement naturally shifted specific proportions, yielding lower in-vehicular usage (4.1% versus 10.5%) and higher ambulatory activity (16.0% versus 7.3%), the persistence of diverse locomotive states similar to that in the independent dataset confirms that our dataset captures a realistic cross-section of daily life filtered through natural phone usage patterns. Accelerometer-based step counters revealed finer-grained motion dynamics during these states (Supplementary Table 9); users were rarely completely stationary (non-movement: 5.0%), with most videos capturing incidental steps (52.7%) and sporadic movement (15.1%). Distinct locomotive patterns were also well-represented, ranging from purposeful stepping (9.6%) to walking at varying paces (17.1%).
Fig. 2: Representative examples of the diversity of free-living data used to validate the PHRM system.
Fig. 2: Representative examples of the diversity of free-living data used to validate the PHRM system.The alternative text for this image may have been generated using AI.
Full size image

a, Illustrative examples of the variety of environments, lighting conditions, front-facing camera angles and face obstructions for videos captured in the free-living conditions. b, Examples of facial skin patches randomly sampled from video frames of the cheeks of participants across the full range of MST values. Videos are sorted by mean brightness across columns and MST across rows. c, From left to right: histograms of the number of 8-s video clips by time of day; illuminance measured by the smartphone ambient light sensor; and the average magnitude of linear acceleration of the smartphone during the videos.
In-laboratory HR test performance

We first investigated how well smartphones measure HR in controlled conditions by comparing PHRM predictions with HR measured by the reference ECG. In the prospective laboratory study, comprising 104 participants, we successfully obtained a valid HR measurement (by gating on the confidence scores associated with the PHRM predictions; see details in Methods) in 1,360 out of 1,750 face videos (77.7%). The one participant from whom we did not obtain any valid HR measurements was seated far from the camera, resulting in a high failure rate (62.5%) of detecting the facial landmarks that are needed to perform video stabilization. Compared with the reference ECG HR, PHRM achieved a mean absolute error (MAE) of 4.09 (95% confidence interval (CI): 3.03, 5.33) and a mean absolute percentage error (MAPE) of 5.65% (95% CI: 4.25, 7.29) at the participant level in the overall study population (Extended Data Table 1). The MAPE values for all five lighting conditions and for both the at-rest and the post-exercise conditions were significantly lower than the prespecified study target of 10% (P < 0.001), according to the American National Standards Institute (ANSI) and Consumer Technology Association (CTA) ANSI/CTA-2065 standard32, indicating robustness across lighting and physiological conditions. The MAPE for the post-exercise condition (2.74%) was lower than that for the at-rest condition (6.01%), which seems counterintuitive, because the post-exercise state is expected to be more challenging, owing to motion, heavy breathing and rapid changes in HR; this result might be due to the effectiveness of the use of measurement gating to remove erroneous estimates under such noisy conditions. Indeed, the PHRM measurement success rate while participants were at rest was 78.4% (95% CI: 76.3, 82.5), which was higher than the 62.1% (95% CI: 56.6, 67.6) success rate after exercise (Extended Data Table 2).

The Bland–Altman plot showed minimal bias (−0.7) and 95% limits of agreement, adjusted for multiple measurements per participant, between −12.9 and 11.5 bpm (Extended Data Fig. 1). The participant-level MAPE by skin-tone group was 3.81% (95% CI: 2.43, 5.94) for group 1, 4.43% (95% CI: 3.12, 6.06) for group 2 and 8.93% (95% CI: 5.60, 12.60) for group 3; all were significantly lower than 10% (P < 0.025). The MAPE was highest for group 3 under incandescent lighting.

For comparison, we evaluated 15 rPPG models from 2019 to 2025, representing the current state of the art in rPPG, on this external test set. These models comprise a hue-channel-based signal processing algorithm (Savur33) and deep-learning-based architectures including PhysNet34, TS-CAN24, EfficientPhys35, PhysFormer36, PhysMamba37, RhythmMamba38 and ME-rPPG39. For each of these deep-learning architectures, we evaluated two distinct versions available: one trained on the Pulse Rate Detection (PURE) dataset40 and one trained on the Remote Learning Affect and Physiology (RLAP) dataset41. The PHRM model was the only model to achieve a MAPE lower than 10% across all skin tones, significantly outperforming these previous methods by a wide margin across all skin-tone groups (Fig. 3a).
Fig. 3: Comparison of PHRM and state-of-the-art rPPG methods on our HR test benchmarks.
Fig. 3: Comparison of PHRM and state-of-the-art rPPG methods on our HR test benchmarks.The alternative text for this image may have been generated using AI.
Full size image

a,b, MAPE values at the participant level of PHRM and 15 baselines by skin-tone group in diverse laboratory (a; n = 44, 25 and 35 participants in the light, medium and dark skin-tone groups, respectively) and free-living (b; n = 39, 29 and 39 participants in the light, medium and dark skin-tone groups, respectively) conditions. Deep-learning baseline models were trained on the PURE (ending with -P) and RLAP (ending with -R) datasets. Filled bars indicate MAPE on valid HR measurements after gating on the confidence scores associated with the PHRM predictions; empty bars indicate MAPE on all videos without any gating. Red dashed lines indicate the prespecified accuracy target of MAPE <10% according to the ANSI/CTA-2065 standard. Error bars are upper 95% CIs.
Free-living HR test performance

Next, we evaluated the accuracy of passive HR measurements during smartphone use in real-world conditions. We successfully obtained valid HR measurements for 100% of the 101 participants in the free-living test set. These videos were captured using 26 smartphone models. We observed a lower video-level HR measurement success rate of 43.1% (68,307 valid measurements out of the 158,471 face videos) than was observed in the laboratory study, as expected given that the measurements were in uncontrolled conditions without participants being told to stay still for the duration of each measurement. The video-level measurement success rate decreased by skin-tone group: it was 58% in group 1, 45% in group 2 and 25% in group 3. Compared with the reference ECG HR, PHRM achieved a video-level MAE of 3.59 (95% CI: 3.33, 3.88) and MAPE of 4.83% (95% CI: 4.39, 5.28); at the participant level, the MAE was 4.58 (95% CI: 4.14, 5.06) and the MAPE was 6.09% (95% CI: 5.35, 6.87). Both video-level and participant-level MAPEs were significantly lower than 10% (P < 0.001), indicating that PHRM provided accurate passive HR measurements (Table 2).
Table 2 Accuracy of passive HR and daily RHR measurements by PHRM in free-living conditions across skin pigmentation groups
Full size table

The participant-level MAPE by skin-tone group was 5.04% (95% CI: 4.36, 5.89) for group 1, 5.12% (95% CI: 4.51, 5.80) for group 2 and 7.84% (95% CI: 6.25, 9.71) for group 3; all were significantly lower than 10% (P < 0.001). The MAPE values for all three skin-tone groups were significantly lower than 10% at both the video and the participant level (P < 0.001 for all comparisons), showing that PHRM provided accurate passive HR measurements for all skin pigmentation groups (Fig. 4b). The performance advantage of PHRM compared with existing rPPG models was even more pronounced as the existing rPPG models yielded substantially higher errors across all skin tones (Fig. 3b). The difference in PHRM participant-level MAPE between skin pigmentation group 1 versus others, group 2 versus others and group 3 versus others was −1.64 percentage points (95% CI: −2.96, −0.37), −1.31 percentage points (95% CI: −2.51, −0.17) and 2.75 percentage points (95% CI: 1.04, 4.70), respectively. This met the prespecified non-inferiority target of differences less than five percentage points, indicating that PHRM provided equitable HR measurements for all skin pigmentation groups. The Bland–Altman plot showed minimal bias (0.64) and 95% limits of agreement, adjusted for multiple measurements per participant, between −11.3 and 10.3 bpm (Fig. 4a). Notably, videos with lower errors tended to have higher confidence scores, illustrating the effectiveness of the confidence-based gating algorithm. Overall, our results show that PHRM is robust to diverse lighting, physiological and real-world conditions, and that it provided accurate and equitable HR measurements across skin pigmentation groups.
Fig. 4: Accuracy of passive HR and RHR measurements by PHRM in free-living conditions.
Fig. 4: Accuracy of passive HR and RHR measurements by PHRM in free-living conditions.The alternative text for this image may have been generated using AI.
Full size image

a, Bland–Altman plot showing the agreement between PHRM-estimated HR values and reference ECG measurements. Colours indicate the confidence level of PHRM predictions. Dashed lines show the bias, lower and upper limits of agreement adjusted for repeated measurements with unequal numbers of replicates. b, Box plots showing the distribution of MAPE values for individual participants, grouped by skin pigmentation. n = 39, 29 and 39 participants in skin-tone groups 1, 2 and 3, respectively. The box bounds the IQR divided by the median, and whiskers extend to a maximum of 1.5 × IQR beyond the box. Red dashed line indicates the prespecified accuracy target of MAPE <10%. c, Bland–Altman plot showing the agreement between PHRM-estimated daily RHR values and the reference wearable HR tracker measurements. Colours indicate the day number since the start of RHR predictions. Dashed lines show the bias, lower and upper limits of agreement adjusted for repeated measurements with unequal numbers of replicates. d, MAE of PHRM-estimated RHR as a function of day number since the start of RHR predictions, grouped by skin pigmentation. Shaded areas indicate 95% CIs. Red dashed line indicates the prespecified accuracy target of MAE <5 bpm.
Daily RHR measurements

Because our findings indicated that smartphones could measure HR passively in free-living conditions, we tested our hypothesis that RHR could be estimated on a daily basis using these intermittent HR measurements by comparing PHRM predictions with daily RHR measured by a reference wearable HR tracker. Out of the 101 participants, 90 (89.1%) had one or more days with at least 20 valid HR measurements, which was the minimum number of measurements needed to compute a valid daily RHR (Supplementary Fig. 2). Among these participants, a valid daily RHR measurement was obtained in 504 out of 685 (73.6%) participant days. Compared with the reference wearable HR tracker, PHRM achieved a day-level and participant-level MAE of 3.62 bpm (95% CI: 3.18, 4.09) and 4.39 (95% CI: 3.67, 5.17) bpm for RHR measurements, respectively, in the overall study population. Both MAE values were significantly lower than the prespecified study target of 5 bpm (P < 0.001), indicating that PHRM provided accurate daily RHR measurements passively, during ordinary phone use. The Bland–Altman plot showed minimal bias (0.1) and adjusted 95% limits of agreement between −9.1 and 9.2 bpm (Fig. 4c).

The PHRM-derived daily RHR was highly correlated with daily RHR from the wearable HR tracker (r = 0.87, P < 0.001); correlation with conventional methods of measuring RHR using ECG in a supine position (r = 0.73, P < 0.001) and in a sitting position (r = 0.74, P < 0.001) was also high (Extended Data Fig. 3). The intra-person standard deviation (s.d.) for supine and sitting RHR was 5.08 bpm (95% CI: 4.44, 5.78) and 4.52 bpm (95% CI: 3.95, 5.11), respectively. The coefficient of variation (CV) for supine and sitting RHR was 7.85% (95% CI: 6.84, 9.01) and 6.68% (95% CI: 5.81, 7.64), respectively. We observed that the reproducibility of PHRM-derived RHR was higher than that of conventional methods, with a significantly lower intra-person s.d. of 1.20 bpm (95% CI: 1.06, 1.35) and CV of 1.77% (95% CI: 1.56, 1.99).

At the day level, MAE of daily RHR measurements by skin-tone group was 3.44 (95% CI: 2.75, 4.62) for group 1, 3.47 (95% CI: 2.77, 4.25) for group 2 and 4.06 (95% CI: 3.23, 5.13) for group 3; all were significantly lower than 5 bpm (P < 0.001). The participant-level MAE by skin-tone group was 3.72 (95% CI: 2.94, 4.57) for group 1, 3.56 (95% CI: 2.60, 4.77) for group 2 and 5.86 (95% CI: 4.18, 7.70) for group 3; MAEs were significantly lower than 5 bpm for groups 1 (P < 0.005) and 2 (P < 0.001), but did not reach significance for group 3 (P = 0.32). However, we found that the MAE across all skin-tone groups decreased over time as the Kalman filter converged. From the third day onwards, the MAE for group 3 was significantly lower than 5 bpm (Fig. 4d). Visually, we observed that the PHRM-derived RHR was able to capture similar trends to those obtained from the wearable HR tracker (Extended Data Fig. 4).
Association of daily RHR with health factors

Finally, we examined whether the daily RHR estimated from smartphone use was associated with well-established risk factors for cardiovascular disease as a way to confirm the validity of our approach. We found that participants with a higher PHRM-derived RHR, after adjusting for chronological age, sex and day of measurement, were more likely to exhibit markers of poorer health; namely, obesity and poor cardiovascular fitness (Table 3). In a generalized least squares (GLS) model of PHRM-derived RHR, both higher BMI and lower cardiovascular fitness were independent predictors (β = 1.92 ± 0.57 bpm, P < 0.001 and β = −1.90 ± 0.27 bpm, P < 0.001 per 1 s.d. increase of BMI and maximal oxygen consumption (VO2 max), respectively). Together, these results show that the PHRM system produces daily RHR estimates that are accurate and associated with markers of health status.
Table 3 GLS model of PHRM-estimated daily RHR
Full size table
Discussion

To our knowledge, this is the first demonstration that smartphones can be used to monitor both HR and daily RHR passively during normal personal phone use in the real world. It is also a large and diverse prospective validation study of a smartphone-based rPPG system. Notably, our proposed system produces accurate HR measurements that meet the ANSI and CTA standards32 for consumer HR monitors (MAPE ≤ 10%) across all skin pigmentation groups, and it outperforms state-of-the-art rPPG methods on our benchmarks. We show that smartphone-based rPPG meets a prespecified non-inferiority target for performance across skin-tone groups, which is crucial for equitable measurement. We also found that smartphones were able to produce accurate estimates of daily RHR; smartphone-derived RHR was associated with known risk factors for cardiovascular disease.

Our study advances rPPG from conceptual feasibility on stationary devices to large-scale validation using personal smartphones in free-living environments. We address major gaps in previous research, such as small sample sizes and limited skin-tone diversity, by validating our system on 162,546 videos from 211 participants, and adhering to FDA guidelines for representation across all skin tones. By deliberately ensuring a substantial representation of darker skin tones in our studies and introducing a deep-learning architecture that reframes HR estimation as a multi-class classification problem, we achieve equitable performance across all groups. For more detailed information on our comparative benchmarks and modelling techniques, please refer to the Supplementary Discussion. Furthermore, we show that these passive, opportunistic HR measurements can be aggregated into longitudinal biomarkers, such as daily RHR. Notably, we found that PHRM-based daily RHR had a lower intra-person s.d. and CV than those obtained by conventional methods, indicating that it is consistent and reproducible. Presumably, this is because many observations of HR throughout the day capture more consistent RHR values than does a single measurement in the supine or sitting position10,42. Smartphone-based RHR was significantly associated with known risk factors for cardiovascular disease that influence clinic-based RHR, including obesity43,44,45 and low VO2 max46,44,. The fact that the passively estimated RHR reproduces these known, fundamental relationships with markers of obesity and cardiovascular fitness provides confidence that the signal we are extracting is indeed the physiological RHR. The results of our system on this study population demonstrate its potential clinical significance and suggest the possibility of using it to ambiently monitor elements of health status through the indicator of RHR. This opens up the possibility of automatically collecting longitudinal RHR data across weeks, months, seasons and years, which could provide valuable health information.

Our work has some limitations. In this work, we defined measurement equity in terms of two prespecified criteria for fairness; namely, achieving an accuracy of MAPE below the 10% ANSI/CTA standard for all three skin-tone groups individually, and non-inferiority of accuracy between groups. This, however, is not a denial of any difference between skin-tone groups. For example, the video-level measurement success rate was lower in the darkest skin pigmentation group after automated confidence-based gating. This is likely to be related to the signal-to-noise (SNR) of the pulse signal captured in the videos, which has been reported to decrease with darker skin pigmentation47,48. Lower SNR has been attributed to the increased melanin content in darker skin pigmentation, which limits the amount of light that enters the deeper skin layers with pulsatile blood vessels and absorbs a portion of diffuse reflections carrying pulsatile information, whereas specular reflections are not reduced47,48. In addition, we cannot rule out other factors that might differ between the skin-tone groups, such as participant physical activity levels, phone device hardware and environment or lighting conditions. Our goal was not to produce HR measurements for every instance of phone interaction, because certain scenarios are simply not conducive to such measurements (for example, very brief phone pickups, heavy motion or very poor lighting conditions), but to opportunistically provide accurate HR measurements. Comparison of the full and gated datasets revealed that the retained valid videos preserved the broad spectrum of daily life conditions we captured in our free-living study, representing a cross-section of genuine, unconstrained user behaviour. Of note, despite the lower success rate in the darkest skin-tone group, the system still captured a sufficient number of valid measurements to compute an accurate daily RHR for those participants, demonstrating the feasibility of this approach across all skin tones. A potential broader mitigation would be to increase the number of measurement attempts when a low SNR is encountered. Relatedly, under controlled conditions, we also observed that the MAPEs were highest for the darkest skin-tone group under incandescent lighting. This could be due to the fact that incandescent lighting contains much less spectral power in the green wavelengths49, which is optimal for blood absorption and hence the SNR of the pulse signal. One approach is to investigate optimizing camera exposure settings to boost the SNR, which might improve both the success rate and the accuracy of measurements under such lighting. Globally, incandescent lighting is increasingly uncommon, because there are ongoing efforts in many countries to phase out incandescent light bulbs to promote energy efficiency50. There were a few participants with high, outlying MAPEs. From visual inspection of the videos from these participants, we observed frequent head motion and talking. Although our video stabilization method was effective in improving the overall HR accuracy, we observed that it introduced noticeable artefacts in certain videos. Considering other motion stabilization techniques might present an opportunity for further improvements.

In this research study, we did not account for constraints on battery consumption in favour of collecting more data, and thus opportunistically collected videos at any time a participant’s phone was unlocked. To minimize smartphone power consumption and improve measurement success rates, future efforts should focus on identifying optimal sensing conditions, such as sufficient ambient brightness and low motion, before activating the camera. Further refinements to the RHR algorithm could include accounting for circadian rhythms based on measurement timing and using accelerometer data to verify adequate rest periods. In addition, because smartphone use involves inherent hand and finger movements, further development is needed to accurately distinguish these interactions from overall physical activity levels. Finally, our study focused on the validity, reliability and equity of HR and daily RHR measurements in a general population; future studies could investigate the utility of this technology for clinical use cases, such as monitoring patients with atrial fibrillation or heart failure.

Some key privacy concerns need to be considered for respectful use of this technology. Smartphone users should be asked to grant explicit informed consent before enabling passive video-based HR measurement. To ensure participant privacy, videos were saved locally on devices and manually reviewed and uploaded by participants for research use. Participants were specifically instructed to exclude sensitive content or faces other than their own. Although this manual review process introduces potential human-in-the-loop variance, it served as an ethical necessity to protect privacy and ensure data quality by confirming that videos correctly matched the reference HR data. The risk of data bias is considered minimal, owing to the high volume of approved uploads per participant. The PHRM system was designed such that it could be run locally on a smartphone’s processors, enabling the videos to be processed on-device. To balance measurement accuracy with coverage and device power consumption, a minimum video duration of eight seconds was selected. Consequently, videos were not recorded for very brief phone interactions. Despite these constraints, the validation dataset effectively captured a wide range of HRs and included diverse real-world challenges such as motion, low light and awkward angles. Such a system could be implemented in a protected on-device environment isolated from unauthorized access, such as a trusted execution environment, to ensure that the video images remain secure during execution. In addition, implementation of such a system could make HR measurement contingent on successful face authentication, mitigating measurements of other individuals and incorrect HR data attribution.

In conclusion, we have developed and validated a system for passive measurement of HR and daily RHR during normal phone use that performs accurately across all skin pigmentation groups. This advancement in the state of the art of rPPG presents a promising approach to improve equitable access to the benefits of heart-health tracking, by widening its availability to everyone who has a smartphone.
Methods
Studies

Between October 2020 and March 2024, we conducted five independent, prospective laboratory studies and a prospective free-living study to obtain datasets to develop and validate PHRM. All study protocols were approved by an institutional review board (Quorum, now known as Advarra, and WCG). We obtained informed consent from all participants, and the study was performed in accordance with the principles of the Declaration of Helsinki.

In the laboratory validation studies, we objectively measured skin tone from each participant using a RM200QC spectrocolorimeter (Pantone) to image the skin of the cheeks and forehead. For the free-living study, because it was entirely remote with no in-person component, we provided participants with a visual representation of the MST to self-assess their skin tone51.
Reference measurements

To validate HR measurements of PHRM in laboratory settings, we used ECG recorded by the BIOPAC MP160 system as the reference ground truth. We used a custom LabVIEW (National Instruments) application to record three-lead ECG signals from electrodes placed on study participants’ upper chests (or upper arms) and lower abdomens.

For validating HR measurements of PHRM in real-world, free-living conditions, we used the Polar H10 ECG chest strap. The Polar H10 has been validated as providing accurate HR measurements during physical activity52,53. Participants were instructed to put the chest strap on every morning and to wear it for at least seven hours each day, except during showers or sleep.

Because aggregating multiple watch HR measurements provides more consistent RHR values, compared with spot measurements in a supine or sitting position10, we chose to use the daily RHR from the Fitbit Charge 6 (Google) as our primary reference for RHR. The daily RHR produced by Fitbit devices is computed by combining multiple HR measurements across ‘at rest’ periods throughout the day, in which the on-device accelerometer has determined that the person is at rest, and has not been moving recently. If available, sleeping HR is also used to improve the daily RHR estimate. The Fitbit daily RHR has been shown to be closest to RHR measurements that are taken when people are lying down, immediately after waking up12. In addition, participants were instructed to perform two conventional RHR measurements first thing in the morning, before eating, drinking, exercising or showering. After putting on the ECG chest strap, they lay in a supine position for 6 min. Next, they sat still for another 6 min. Supine and sitting RHR measurements were computed as the minimum HR from the ECG recordings and served as secondary references. HR has been found to stabilize in most individuals after four minutes of inactivity54.
Time synchronization

To synchronize the clocks across all of the study devices during the free-living study, participants performed a daily routine comprising a series of three jumps. We instructed participants to stand still with their hands placed in front of the chest. They held their smartphone enrolled in the study in their dominant hand; the wearable HR tracker was placed on the off-hand. To perform the series of jumps, we asked participants to start a timer, and complete the following sequence: standing still for one minute, three jumps spaced by 10 s, followed by standing still for another 10 s. We aligned the timestamps of the smartphone and the ECG chest strap by maximizing the cross-correlation between their respective accelerometer signals after resampling the signals to 60 Hz.
PHRM-HR module

The PHRM-HR module is the component of our algorithm that predicts a HR measurement. The PHRM-HR module processes an 8-s video input to predict HR. In this section, we describe the preprocessing pipeline applied to raw video, the deep neural network used for HR extraction and the confidence-gating algorithm.
Developmental metric

To assess and optimize each subcomponent of the PHRM-HR module, we used the tuning dataset and computed the root mean square error (RMSE) after excluding the bottom 20% of videos with the lowest confidence scores. We note that this exclusion was performed only during the development stage for hyperparameter optimization on the tuning dataset, and not during the testing stage. The choice of 20% was based on empirical results to reduce variance due to outliers that made it difficult to distinguish between hyperparameter choices.
Video preprocessing

To prepare video data for HR extraction, we implemented a five-step preprocessing pipeline. First, frames were stabilized using an affine transformation based on facial landmark centroids55 to improve robustness against motion artefacts. Second, we standardized varying mobile device frame rates to a consistent 15 frames per second using linear interpolation. Third, we performed face cropping using a minimal bounding box with a 20% margin to isolate the face and reduce background noise. Fourth, frames were resized to 32 × 32 pixels using anti-aliased resampling to ensure computational efficiency on mobile devices while preserving signal quality. Finally, we applied frame differencing between consecutive frames to highlight pixel changes associated with physiological signals. Detailed descriptions of each preprocessing step are provided in the Supplementary Information.
PHRM-HR network

To extract HR from the preprocessed video data, we used a deep-learning model based on a TSCNN architecture24. This backbone was selected for its efficiency in modelling temporal dependencies, facilitating on-device processing on smartphones. The network processes 8-s video segments at 15 Hz (a total of 120 frames) to produce a pseudo-PPG signal, which is converted into the frequency domain via a fast Fourier transform layer. These frequencies were bucketed into 1-bpm bins, and a softmax function was applied to generate HR probabilities, with the final HR computed as a weighted sum. The model was trained by reformulating HR regression as a classification task using focal loss56 applied to the bucketed ground-truth values. We performed an extensive hyperparameter search using a Gaussian process-based tool57, and generated final predictions by ensembling the top five models. Further details on the specific network layers, optimization schedules, data augmentation techniques, hyperparameters and ablation results are provided in the Supplementary Information.
Confidence gating

Owing to the unconstrained environment of normal phone use in real-world settings, it was necessary to apply a gating criterion to discard face videos that were too noisy for reliable HR estimation, such as those with no face present, or with face coverings or excessive movement. We found that the deep-learning model’s confidence of a HR prediction was an effective metric for this purpose, and derived an optimal threshold using the tuning dataset. Specifically, we use the negative entropy of the HR probabilities generated by the PHRM-HR module, where a higher negative entropy indicates a higher confidence. In the Supplementary Information, we compared two alternatives, including pseudo-PPG SNR and maximum HR probability, and found that negative entropy was the most effective for determining valid HR measurements.

Another key aspect of gating is determining the cut-off, which defines the threshold below which videos are filtered. To ensure that this cut-off was accurate and equitable across skin tones, we applied two rules when searching for the cut-off in the free-living tune dataset. First, the overall MAPE for each skin-tone group had to be lower than 8%. Second, the MAPE gap between any two skin-tone groups had to be lower than 3%. The search process is detailed in the Supplementary Information.
PHRM-RHR model

For the daily RHR algorithm, we maintained simplicity to enhance its generalization. First, we aggregated the valid HR measurements across a single day by computing the tenth percentile value and applying a bias correction factor (which is a constant across all participants). We performed a grid search for the optimal percentile and bias correction values using the tune dataset. Next, we applied a Kalman filter to refine the RHR prediction from noisy estimates. Using the tune dataset, we also identified the minimum number of valid HR measurements needed in a day to provide a valid RHR estimate. The core of our approach is statistical: by passively capturing a large volume of HR measurements throughout the day, we create a dense distribution of the user’s daily HR. This statistical approach is robust against sporadic measurements during physical or emotional arousal, as these high HR values are filtered out as upper-tail outliers. The motivation for the Kalman filter is grounded in optimal state estimation theory. Our system produces a series of daily RHR estimates, each with some degree of measurement error. The Kalman filter systematically combines the prediction from the previous day’s state with the new, noisy measurement to produce a refined, more accurate estimate of the current day’s true RHR. It provides a principled way to increase the stability of daily RHR estimates and track the underlying physiological trend more accurately.
Statistical analysis

For HR measurements, we established a pre-determined accuracy target of MAPE <10% in accordance with the ANSI/CTA-2065 standards for consumer HR monitors32; this is based on the ANSI/AAMI standard for HR accuracy for ECG monitors, which states that a ±10% allowable error is consistent with clinical needs58. By using MAPE, we directly assessed our system’s performance against the established benchmark for commercial consumer HR devices. Measurements were paired observations: PHRM-estimated HR and reference HR from ECG. A paired measurement was dropped if PHRM did not produce a valid HR measurement; that is, the confidence of PHRM predictions was lower than the gating cut-off. Each participant contributed multiple 8-s videos for HR measurements. We computed MAPE at the video level as the mean value for all absolute percentage error values from each paired measurement, and used bootstrapping at the participant level to obtain 95% CIs. Because each participant contributed a different number of videos, we also computed MAPE at the participant level as the mean value for the MAPEs of individual participants, which we visualized using box plots. Because the errors were not normally distributed according to a Shapiro–Wilk test, we determined whether the participant-level MAPE values were significantly lower than 10% on the basis of whether P was less than 0.05 using the Wilcoxon signed-rank test. We determined non-inferiority on the basis of whether the upper limit of the 95% CI around the difference in MAPE across participants in any of the three skin pigmentation groups, compared with that across participants in the other two skin pigmentation groups, was less than a prespecified five percentage points. Bland–Altman plots were used to visualize the agreement between the estimated values and the reference measurements; limits of agreement were adjusted for repeated measurements with unequal numbers of replicates59.

For daily RHR measurements, we adopted a prespecified accuracy target of MAE <5 bpm, which provided a stricter requirement than MAPE <10%. Similar to above, measurements were paired observations: PHRM-estimated daily RHR and reference daily RHR from the wearable HR tracker. A minimum of 20 valid HR measurements on a given day was required for PHRM to yield a valid RHR estimate for that day. Each participant contributed multiple days for RHR measurements. We computed MAE at the day level as the mean value for all absolute error values from each paired measurement and used cluster bootstrapping to obtain the 95% CIs. To account for multiple observations, we also computed MAE at the participant level as the mean value for the MAEs of individual participants, which we visualized using box plots. Because the Kalman filter in the PHRM-RHR module takes time to converge, we also computed the MAE for each day since the start of PHRM-RHR predictions to evaluate the MAE over time. We determined whether MAE values were significantly less than 5 bpm on the basis of whether P was less than 0.05 using the Wilcoxon signed-rank test. Bland–Altman plots were used to visualize the agreement between the estimated values and the reference measurements; limits of agreement were adjusted for repeated measurements with unequal numbers of replicates59.

Associations between PHRM-estimated daily RHR and known risk factors were evaluated using a GLS model accounting for the correlation structure present across the serial estimates of RHR while also adjusting for the clusters within participants60. Ground-truth measurements and associated estimates of RHR were collected daily for each participant over the course of the study. Therefore, statistical models of RHR have to (1) account for the serial correlation existing among repeated measurements; and (2) adjust for the presence of clusters within participants. Given its flexibility, we decided to follow a GLS modelling approach. First, we evaluated several possible forms for the correlation structure across days of measurement given a participant cluster and ultimately selected the ‘spherical correlation’ form, because it was most suited to our data in terms of the Akaike information criterion. We then fitted a GLS model of the PHRM-estimated RHR using BMI and VO2 max as covariates and adjusting for age, sex, age × sex and day of measurement. The analyses were performed in R v.4.41, using the rms v.6.8 and nlme v.3.1 packages.
Reporting summary

Further information on research design is available in the Nature Portfolio Reporting Summary linked to this article.
Data availability

The data for three independent, prospective laboratory studies as well as a de-identified free-living dataset (collectively, the ‘HR datasets’) are made available for research purposes only. To access the datasets, applicants must provide a verified academic email address and a study protocol approved by an institutional review board or an equivalent ethics committee. Furthermore, a data security plan must be submitted detailing how the datasets will be encrypted and restricted to authorized team members. Applicants must also provide affirmative consent to neither share the datasets with third parties nor include explicit examples of the data in any presentations or publications. Instructions for obtaining access are available on GitHub: https://github.com/Google-Health/consumer-health-research/tree/main/rppg. Please refer to the Supplementary Information for more details.
Code availability

Pseudocode implementation of the algorithms is available in the Supplementary information.
References

    Kannel, W. B., Kannel, C., Paffenbarger, R. S. Jr & Cupples, L. A. Heart rate and cardiovascular mortality: the Framingham study. Am. Heart J. 113, 1489–1494 (1987).

    Article
     
    CAS
     
    PubMed
     
    Google Scholar
     

    Raisi-Estabragh, Z. et al. Age, sex and disease-specific associations between resting heart rate and cardiovascular mortality in the UK BIOBANK. PLoS ONE 15, e0233898 (2020).

    Article
     
    CAS
     
    PubMed
     
    PubMed Central
     
    Google Scholar
     

    Alhalabi, L. et al. Relation of higher resting heart rate to risk of cardiovascular versus noncardiovascular death. Am. J. Cardiol. 119, 1003–1007 (2017).

    Article
     
    PubMed
     
    Google Scholar
     

    Ceconi, C., Guardigli, G., Rizzo, P., Francolini, G. & Ferrari, R. The heart rate story. Eur. Heart J. Suppl. 13, C4–C13 (2011).

    Article
     
    Google Scholar
     

    Nauman, J., Janszky, I., Vatten, L. J. & Wisløff, U. Temporal changes in resting heart rate and deaths from ischemic heart disease. JAMA 306, 2579–2587 (2011).

    Article
     
    CAS
     
    PubMed
     
    Google Scholar
     

    Seviiri, M. et al. Resting heart rate, temporal changes in resting heart rate, and overall and cause-specific mortality. Heart 104, 1076–1085 (2018).

    Article
     
    CAS
     
    PubMed
     
    Google Scholar
     

    Vazir, A. et al. Association of resting heart rate and temporal changes in heart rate with outcomes in participants of the Atherosclerosis Risk in Communities study. JAMA Cardiol. 3, 200–206 (2018).

    Article
     
    PubMed
     
    PubMed Central
     
    Google Scholar
     

    Johansen, C. D. et al. Resting, night-time, and 24 h heart rate as markers of cardiovascular risk in middle-aged and elderly men and women with no apparent heart disease. Eur. Heart J. 34, 1732–1739 (2013).

    Article
     
    CAS
     
    PubMed
     
    Google Scholar
     

    Hansen, T. W. et al. Prognostic superiority of daytime ambulatory over conventional blood pressure in four populations: a meta-analysis of 7,030 individuals. J. Hypertens. 25, 1554–1564 (2007).

    Article
     
    CAS
     
    PubMed
     
    Google Scholar
     

    Dunn, J. et al. Wearable sensors enable personalized predictions of clinical laboratory measurements. Nat. Med. 27, 1105–1112 (2021).

    Article
     
    CAS
     
    PubMed
     
    PubMed Central
     
    Google Scholar
     

    Korshøj, M. et al. The relation of ambulatory heart rate with all-cause mortality among middle-aged men: a prospective cohort study. PLoS ONE 10, e0121729 (2015).

    Article
     
    PubMed
     
    PubMed Central
     
    Google Scholar
     

    Russell, A., Heneghan, C. & Venkatraman, S. Investigation of an estimate of daily resting heart rate using a consumer wearable device. Preprint at medRxiv https://doi.org/10.1101/19008771 (2019).

    Quer, G., Gouda, P., Galarnyk, M., Topol, E. J. & Steinhubl, S. R. Inter- and intraindividual variability in daily resting heart rate and its associations with age, sex, sleep, BMI, and time of year: retrospective, longitudinal cohort study of 92,457 adults. PLoS ONE 15, e0227709 (2020).

    Article
     
    CAS
     
    PubMed
     
    PubMed Central
     
    Google Scholar
     

    Alexander, J., Sovakova, M. & Rena, G. Factors affecting resting heart rate in free-living healthy humans. Digit. Health 8, 20552076221129075 (2022).

    PubMed
     
    PubMed Central
     
    Google Scholar
     

    Mishra, T. et al. Pre-symptomatic detection of COVID-19 from smartwatch data. Nat. Biomed. Eng. 4, 1208–1220 (2020).

    Article
     
    CAS
     
    PubMed
     
    PubMed Central
     
    Google Scholar
     

    Dhingra, L. S. et al. Use of wearable devices in individuals with or at risk for cardiovascular disease in the US, 2019 to 2020. JAMA Netw. Open 6, e2316634 (2023).

    Article
     
    PubMed
     
    PubMed Central
     
    Google Scholar
     

    Gelles-Watnick, R. Americans’ use of mobile technology and home broadband. https://www.pewresearch.org/internet/2024/01/31/americans-use-of-mobile-technology-and-home-broadband/ (Pew Research Center, 2024).

    Kerai, A. 2023 Cell phone usage statistics: mornings are for notifications. https://www.reviews.org/mobile/cell-phone-addiction/ (REVIEWS.org, 2023).

    Sun, Y. & Thakor, N. Photoplethysmography revisited: from contact to noncontact, from point to imaging. IEEE Trans. Biomed. Eng. 63, 463–477 (2016).

    Article
     
    PubMed
     
    Google Scholar
     

    Wang, W., Den Brinker, A. C., Stuijk, S. & De Haan, G. Algorithmic principles of remote PPG. IEEE Trans. Biomed. Eng. 64, 1479–1491 (2016).

    Article
     
    PubMed
     
    Google Scholar
     

    Poh, M.-Z. & Poh, Y. C. Validation of a standalone smartphone application for measuring heart rate using imaging photoplethysmography. Telemed. Ehealth 23, 678–683 (2017).

    Article
     
    Google Scholar
     

    Yan, B. P. et al. Resting and postexercise heart rate detection from fingertip and facial photoplethysmography using a smartphone camera: a validation study. JMIR Mhealth Uhealth 5, e33 (2017).

    Article
     
    PubMed
     
    PubMed Central
     
    Google Scholar
     

    Qiao, D., Amtul Haq, A., Zulkernine, F., Jaffar, N. & Masroor, R. ReViSe: remote vital signs measurement using smartphone camera. IEEE Access 10, 131656–131670 (2022).

    Article
     
    Google Scholar
     

    Liu, X., Fromm, J., Patel, S. & McDuff, D. Multi-task temporal shift attention networks for on-device contactless vitals measurement. In Proc. Advances in Neural Information Processing Systems 33 (NeurIPS 2020) (eds Larochelle, H. et al.) (NeurIPS, 2020).

    Yan, B. P. et al. Contact-free screening of atrial fibrillation by a smartphone using facial pulsatile photoplethysmographic signals. J. Am. Heart Assoc. 7, e008585 (2018).

    Article
     
    PubMed
     
    PubMed Central
     
    Google Scholar
     

    Nowara, E. M., McDuff, D. & Veeraraghavan, A. A meta-analysis of the impact of skin type and gender on non-contact photoplethysmography measurements. In Proc. 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW) 1148–1155 (IEEE, 2020).

    US FDA Center for Devices and Radiological Health. Approach for Improving the Performance Evaluation of Pulse Oximeter Devices Taking into Consideration Skin Pigmentation, Race and Ethnicity https://www.fda.gov/media/173905/download (FDA, 2023).

    UK Department of Health and Social Care. Equity in Medical Devices: Independent Review — Final Report https://www.gov.uk/government/publications/equity-in-medical-devices-independent-review-final-report (GOV.UK, 2024).

    Ganaie, M. A., Hu, M., Kumar Malik, A., Tanveer, M. & Suganthan, P. N. Ensemble deep learning: a review. Eng. Appl. Artif. Intell. 115, 105151 (2022).

    Article
     
    Google Scholar
     

    Del Bino, S. & Bernerd, F. Variations in skin colour and the biological consequences of ultraviolet radiation exposure. Br. J. Dermatol. 169, 33–40 (2013).

    Article
     
    CAS
     
    PubMed
     
    Google Scholar
     

    Winbush, A. et al. Smartphone use in a large US adult population: temporal associations between objective measures of usage and mental well-being. Proc. Natl Acad. Sci. USA 122, e2427311122 (2025).

    Article
     
    CAS
     
    PubMed
     
    PubMed Central
     
    Google Scholar
     

    Consumer Technology Association (CTA). Physical Activity Monitoring for Heart Rate. Report No. ANSI/CTA-2065 (CTA, 2018).

    Savur, C. et al. Monitoring pulse rate in the background using front facing cameras of mobile devices. IEEE J. Biomed. Health Inform. 27, 2208–2218 (2023).

    Article
     
    PubMed
     
    PubMed Central
     
    Google Scholar
     

    Yu, Z., Li, X. & Zhao, G. Remote photoplethysmograph signal measurement from facial videos using spatio-temporal networks. In Proc. 30th British Machine Vision Conference (BMVC, 2019).

    Liu, X., Hill, B., Jiang, Z., Patel, S. & McDuff, D. EfficientPhys: enabling simple, fast and accurate camera-based cardiac measurement. In Proc. 2023 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 4997–5006 (IEEE, 2023).

    Yu, Z. et al. PhysFormer: facial video-based physiological measurement with temporal difference transformer. In Proc. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 4176–4186 (IEEE, 2022).

    Luo, C., Xie, Y. & Yu, Z. PhysMamba: efficient remote physiological measurement with SlowFast Temporal Difference Mamba. In Proc. 2024 Chinese Conference on Biometric Recognition (CCBR) 248–259 (Springer, 2024).

    Zou, B., Guo, Z., Hu, X. & Ma, H. RhythmMamba: Fast, lightweight, and accurate remote physiological measurement. In Proc. 39th AAAI Conference on Artificial Intelligence 11077–11085 (AAAI, 2025).

    Wang, K. et al. Memory-efficient low-latency remote photoplethysmography through temporal-spatial state space duality. Preprint at https://doi.org/10.48550/arXiv.2504.01774 (2025).

    Stricker, R., Müller, S. & Gross, H.-M. Non-contact video-based pulse rate measurement on a mobile service robot. In Proc. 23rd IEEE International Symposium on Robot and Human Interactive Communication 1056–1062 (IEEE, 2014).

    Wang, K. et al. Camera-based HRV prediction for remote learning environments. In Proc. 2024 IEEE Smart World Congress (SWC) 1165–1173 (IEEE, 2024).

    Palatini, P. et al. Reproducibility of heart rate measured in the clinic and with 24-hour intermittent recorders. Am. J. Hypertens. 13, 92–98 (2000).

    Article
     
    CAS
     
    PubMed
     
    Google Scholar
     

    Rogowski, O. et al. Elevated resting heart rate is associated with the metabolic syndrome. Cardiovasc. Diabetol. 8, 55 (2009).

    Article
     
    PubMed
     
    PubMed Central
     
    Google Scholar
     

    Itagi, A. B. H., Jayalakshmi, M. K. & Yunus, G. Y. Effect of obesity on cardiovascular responses to submaximal treadmill exercise in adult males. J Family Med. Prim. Care 9, 4673–4679 (2020).

    Article
     
    PubMed
     
    PubMed Central
     
    Google Scholar
     

    Martins, D., Tareen, N., Pan, D. & Norris, K. The relationship between body mass index, blood pressure and pulse rate among normotensive and hypertensive participants in the third National Health and Nutrition Examination Survey (NHANES). Cell. Mol. Biol. 49, 1305–1309 (2003).

    CAS
     
    PubMed
     
    Google Scholar
     

    Nauman, J., Aspenes, S. T., Nilsen, T. I. L., Vatten, L. J. & Wisløff, U. A prospective population study of resting heart rate and peak oxygen uptake (the HUNT study Norway). PLoS ONE 7, e45021 (2012).

    de Haan, G. & Jeanne, V. Robust pulse rate from chrominance-based rPPG. IEEE Trans. Biomed. Eng. 60, 2878–2886 (2013).

    Article
     
    PubMed
     
    Google Scholar
     

    Wang, W., Stuijk, S. & De Haan, G. Exploiting spatial redundancy of image sensor for motion robust rPPG. IEEE Trans. Biomed. Eng. 62, 415–425 (2014).

    Article
     
    PubMed
     
    Google Scholar
     

    Abdel-Rahman, F. et al. Caenorhabditis elegans as a model to study the impact of exposure to light emitting diode (LED) domestic lighting. J. Environ. Sci. Health A 52, 433–439 (2017).

    Article
     
    CAS
     
    Google Scholar
     

    Edge, J. & McKeen-Edwards, H. Light bulbs and bright ideas? The global diffusion of a ban on incandescent light bulbs. In 80th Annual Conference of the Canadian Political Science Association (University of British Colombia, 2008).

    Monk, E. The Monk Skin Tone Scale (MST). Preprint at https://osf.io/preprints/socarxiv/pdf4c_v1 (2019).

    Gilgen-Ammann, R., Schweizer, T. & Wyss, T. RR interval signal quality of a heart rate monitor and an ECG Holter at rest and during exercise. Eur. J. Appl. Physiol. 119, 1525–1532 (2019).

    Article
     
    PubMed
     
    Google Scholar
     

    Pasadyn, S. R. et al. Accuracy of commercially available heart rate monitors in athletes: a prospective study. Cardiovasc. Diagn. Ther. 9, 379–385 (2019).

    Article
     
    PubMed
     
    PubMed Central
     
    Google Scholar
     

    Speed, C. et al. Measure by measure: resting heart rate across the 24-hour cycle. PLoS Digit. Health 2, e0000236 (2023).

    Article
     
    PubMed
     
    PubMed Central
     
    Google Scholar
     

    Google for Developers. Augmented Faces introduction. https://developers.google.com/ar/develop/augmented-faces (2024).

    Lin, T.-Y., Goyal, P., Girshick, R., He, K. and Dollár, P. Focal loss for dense object detection. In Proc. IEEE Transactions on Pattern Analysis and Machine Intelligence (Vol. 42) 318–327 (IEEE, 2020).

    Golovin, D. et al. Google Vizier: a service for black-box optimization. In Proc. 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining 1487–1495 (ACM, 2017).

    Association for the Advancement of Medical Instrumentation (AAMI). Medical electrical equipment — Part 2-27: Particular requirements for the basic safety and essential performance of electrocardiographic monitoring equipment. Report No. ANSI/AAMI/IEC 60601-2-27:2011/(R)2016 (AAMI, 2016).

    Bland, J. M. & Altman, D. G. Measuring agreement in method comparison studies. Stat. Methods Med. Res. 8, 135–160 (1999).

    Article
     
    CAS
     
    PubMed
     
    Google Scholar
     

    Harrell, F. E. Regression Modeling Strategies: With Applications to Linear Models, Logistic and Ordinal Regression, and Survival Analysis 2nd edn (Springer, 2015).

Download references
Acknowledgements

We thank N. Teslovich, A. Mun, J. Hsu, X. Zhang, D. Vickers, S. Mravca, T. Giest, J. Guss, F. Thng, J. Zhan, J. Cannon, M. Kashyap, J. Pannu, T. Kung, M. J. Po, M. Shore, J. Tansuwan, L. Chen, C. A. Barrera, A. Saxena, J. Miles, M. Moran, M. V. McConnell, I. Horn, B. Ayalew, M. Howell, K. Chou, J. Saunders, J. Tsai, H. Cole-Lewis, E. Respress, P. Payne, K. Wood and N. Ezeanochie for their support of this work.
Author information
Author notes

    These authors contributed equally: Shun Liao, Paolo Di Achille, Jiang Wu, Silviu Borac, Jonathan Wang, Xin Liu, Eric S. Teasley

    These authors jointly supervised this work: Jameson K. Rogers, Ming-Zher Poh

Authors and Affiliations

    Google Research, Mountain View, CA, USA

    Shun Liao, Jiang Wu, Silviu Borac, Jonathan Wang, Eric S. Teasley, Lawrence Cai, Yuzhe Yang, Yun Liu, Brent Winslow, Anupam Pathak, Mark Malhotra & Jameson K. Rogers

    Google Research, Cambridge, MA, USA

    Paolo Di Achille, Hao-Wei Su & Ming-Zher Poh

    Google Research, Seattle, WA, USA

    Xin Liu, Daniel McDuff, Shwetak Patel & James A. Taylor

    University of Washington, Seattle, WA, USA

    Shwetak Patel

Contributions

E.S.T., J.K.R. and M.-Z.P. conceived the project and revised the manuscript. E.S.T., L.C., B.W., J.A.T. and M.-Z.P. designed and conducted the studies. S.L., P.D.A., J. Wu, S.B., J. Wang, X.L. and Y.Y. conducted experiments and data analysis and revised the manuscript. Y.L., D.M., H.-W.S., A.P., M.M. and S.P. contributed to methodology design, result interpretation and manuscript revision. M.-Z.P. and J.K.R. supervised the project.
Corresponding author

Correspondence to Ming-Zher Poh.
Ethics declarations
Competing interests

This study was funded by Alphabet and/or a subsidiary thereof (‘Alphabet’). All authors are employees of Alphabet and may own stock as part of the standard compensation. The authors have filed a patent application related to this research.
Peer review
Peer review information

Nature thanks the anonymous reviewers for their contribution to the peer review of this work.
Additional information

Publisher’s note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.
Extended data figures and tables
Extended Data Fig. 1 Accuracy of PHRM HR measurements in laboratory settings.

a, Bland–Altman plot showing the agreement between PHRM-estimated HR values and the reference ECG measurements. Colours indicate the confidence level of PHRM predictions. Dashed lines show the bias, lower, and upper limits of agreement adjusted for repeated measurements with unequal numbers of replicates. b, Box plots showing the distribution of MAPE values for individual participants, grouped by skin pigmentation. The box bounds the IQR divided by the median, and whiskers extend to a maximum of 1.5 × IQR beyond the box. The red dashed line indicates the prespecified accuracy target of MAPE < 10%. c, Boxplots showing the distribution of mean absolute error (MAE) values for individual participants, grouped by skin pigmentation. The box bounds the IQR divided by the median, and whiskers extend to a maximum of 1.5 × IQR beyond the box.
Extended Data Fig. 2 Participant-level rates of video approval and upload.

The distribution of approved videos per participant was strongly left-skewed, indicating that most participants uploaded a high proportion of their videos.
Extended Data Fig. 3 Comparison of daily RHR measurements by PHRM and reference and conventional RHR measurements in free-living conditions.

a–c, Scatter plots showing the agreement between PHRM-estimated daily RHR values and reference RHR from a Fitbit wearable HR tracker (a), supine RHR measurements from ECG (b) and sitting RHR measurements from ECG (c), respectively.
Extended Data Fig. 4 Trends in daily RHR measurements over the week.

Comparison of daily RHR estimates from the proposed PHRM system and a reference wearable RHR tracker over a seven-day period, sampled from participants exhibiting diverse trends. To enhance interpretability, participants were categorized into three groups on the basis of reference wearable RHR trends: decreasing, increasing and other patterns.
Extended Data Table 1 Accuracy of HR measurements by PHRM in diverse laboratory conditions and lighting
Full size table
Extended Data Table 2 Measurement success rate by PHRM in diverse laboratory conditions and lighting
Full size table
Supplementary information
Supplementary Information (download PDF )

This file contains the Supplementary Methods, Supplementary Results, Supplementary Discussion, Supplementary References, Supplementary Tables 1–12, Supplementary Figs. 1–5 and Supplementary Code.
Reporting Summary (download PDF )
Rights and permissions

Open Access This article is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License, which permits any non-commercial use, sharing, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons licence, and indicate if you modified the licensed material. You do not have permission under this licence to share adapted material derived from this article or parts of it. The images or other third party material in this article are included in the article’s Creative Commons licence, unless indicated otherwise in a credit line to the material. If material is not included in the article’s Creative Commons licence and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this licence, visit http://creativecommons.org/licenses/by-nc-nd/4.0/.

Reprints and permissions
About this article
Check for updates. Verify currency and authenticity via CrossMark
Cite this article

Liao, S., Di Achille, P., Wu, J. et al. Passive heart-rate monitoring during smartphone use in everyday life. Nature (2026). https://doi.org/10.1038/s41586-026-10507-6

Download citation

    Received14 March 2025

    Accepted08 April 2026

    Published01 June 2026

    Version of record01 June 2026

    DOIhttps://doi.org/10.1038/s41586-026-10507-6

Share this article

Anyone you share the following link with will be able to read this content:

Provided by the Springer Nature SharedIt content-sharing initiative
Subjects

    Biomarkers
    Physiology
    Translational research

Download PDF
Associated content
Smartphone camera takes users’ pulse passively during device use
Nature Clinical Briefing 01 Jun 2026
Your phone can use tiny skin-colour changes to measure your heart rate

    Benjamin Thompson 

Nature Nature Podcast 03 Jun 2026

    Abstract
    Main
    Overview of the system
    Study populations
    In-laboratory HR test performance
    Free-living HR test performance
    Daily RHR measurements
    Association of daily RHR with health factors
    Discussion
    Methods
    Data availability
    Code availability
    References
    Acknowledgements
    Author information
    Ethics declarations
    Peer review
    Additional information
    Extended data figures and tables
    Supplementary information
    Rights and permissions
    About this article

Advertisement

Nature (Nature)

ISSN 1476-4687 (online)

ISSN 0028-0836 (print)
nature.com footer links
About Nature Portfolio

    About us
    Press releases
    Press office
    Contact us

Discover content

    Journals A-Z
    Articles by subject
    protocols.io
    Nature Index

Publishing policies

    Nature portfolio policies
    Open access

Author & Researcher services

    Reprints & permissions
    Research data
    Language editing
    Scientific editing
    Nature Masterclasses
    Research Solutions

Libraries & institutions

    Librarian service & tools
    Librarian portal
    Open research
    Recommend to library

Advertising & partnerships

    Advertising
    Partnerships & Services
    Media kits
    Branded content

Professional development

    Nature Awards
    Nature Careers
    Nature Conferences

Regional websites

    Nature Africa
    Nature China
    Nature India
    Nature Japan
    Nature Middle East

    Privacy Policy Use of cookies Legal notice Accessibility statement Terms & Conditions Your US state privacy rights 

Springer Nature

© 2026 Springer Nature Limited
