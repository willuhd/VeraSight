==================== REVIEW =========================
----------------------------------------------------------------

committee member review (reviewer 1)

 Overall Rating

   Probably accept

 The Review

   This submission presents VeraSight, a real-time mobile sensing pipeline for
               capturing and analyzing facial micromovements using the TrueDepth
               camera of an iPhone 11. The system combines on-device FP16
               quantization and temporal compression, a VAE-based spatial
               representation, a personalized anatomically weighted Isolation Forest,
               a weight-shared GRU predictor, and a downstream LLM-based feedback
               component. The intended applications include public-speaking training,
               gaming-related stress analysis, and behavioral interviews.

   Overall, this is an ambitious and creative Teenager Show project. The authors
               demonstrate substantial hands-on effort across mobile sensing,
               wireless data transmission, machine learning, database integration,
               and interface design. In particular, the implementation of a stable 60
               Hz wireless facial tracking stream, together with the reported
               bandwidth, packet-delivery, and latency measurements, provides
               concrete evidence of a functioning sensing and communication
               prototype. The decision to build personalized behavioral baselines
               rather than relying only on generic supervised classifiers is also
               sensible, given the large differences in facial behavior across
               individuals. The system appears to have strong potential as an
               interactive demonstration.

   The main weakness is that the evaluation strongly supports the sensing and
               transmission pipeline, but does not yet validate the central
               behavioral anomaly and stress-related claims. The paper does not
               report the number of participants or sessions, the experimental tasks,
               ground-truth labels, or quantitative detection metrics such as false-
               alarm rate, precision, recall, or AUROC. It is therefore unclear how
               much the anatomically weighted Isolation Forest improves over a
               standard Isolation Forest, or whether the GRU predictor actually
               reduces false positives caused by speaking, smiling, yawning, and
               gameplay reactions. A small quantitative comparison would
               substantially strengthen the paper, even if it involved only a limited
               number of users.

   The paper should also distinguish more carefully between detecting unusual facial
               movements and measuring physiological stress. A deviation from a
               personalized facial baseline may result from stress, but it may also
               be caused by speech, fatigue, head motion, tracking noise, deliberate
               expressions, or other contextual factors. I recommend describing the
               output primarily as a “personalized facial movement anomaly score”
               unless it is validated against an independent measure of stress.

   Several implementation details are currently missing. The authors should identify
               the datasets used to train the VAE and explain how their facial
               coordinates were aligned with the ARKit mesh. The training procedure,
               architecture, loss function, and threshold selection for the GRU model
               should also be described more clearly. Similarly, the downstream LLM
               component would benefit from information about the model used, its
               execution platform, latency, input format, and safeguards against
               unsupported or inappropriate recommendations.

   A useful evaluation could compare: (1) standard Isolation Forest and AW-iForest,
               (2) single-scale and multi-scale temporal features, and (3) anomaly
               detection with and without the GRU prediction stage. These comparisons
               would directly demonstrate whether the proposed components provide
               value beyond the basic sensing pipeline.

   Finally, the submission should briefly discuss privacy and responsible use. Facial
               geometry and behavioral logs are sensitive data, and the system should
               not be interpreted as a lie detector, psychological assessment tool,
               or diagnostic system. This concern is particularly important for the
               behavioral-interview scenario. The authors should clarify that the
               system is intended for voluntary self-reflection or coaching, describe
               how data are stored and deleted, and acknowledge that facial anomalies
               do not uniquely reveal a user’s emotional or mental state.

   There are also a few minor presentation issues. Figure 1 is labeled as the system
               architecture, although the detailed architecture is actually shown in
               Figure 2. In addition, the paper reports a 7.4 ms round-trip latency
               in Section 3.1 and a 9.2 ms end-to-end latency in Section 3.2; the
               difference between these measurements should be explained.

   Despite these limitations, the project is technically ambitious, relevant to
               ubiquitous computing, and demonstrates impressive implementation
               effort for the Teenager Show. I therefore lean toward acceptance,
               provided that the authors can demonstrate that the major components
               operate together in an end-to-end prototype and revise the claims to
               avoid equating facial anomalies directly with physiological stress.

 Expertise

   Knowledgeable

----------------------------------------------------------------

committee member review (reviewer 2)

 Overall Rating

   Definitely accept

 The Review

   This paper presents VeraSight, an end-to-end mobile sensing pipeline for real-time
               facial micromovement analysis.
   Strengths:
   (1) For authors at the high-school level, the breadth of this project is genuinely
               impressive. The pipeline spans embedded sensing, on-device
               quantization, wireless compression, deep generative modeling (VAE),
               tree-based anomaly detection with domain-informed weighting, recurrent
               predictive coding, and LLM-based downstream reasoning. Building a
               functional prototype that unifies these disparate components reflects
               both intellectual curiosity and substantial hands-on engineering
               effort.
   (2) The authors do not merely simulate their pipeline. They implement it on an
               iPhone 11 using ARKit, measure actual bandwidth reduction (~66% via
               FP16+zlib), and report stable wireless throughput (~282 KB/s) and end-
               to-end latency (~7.4 ms). These concrete metrics lend credibility to
               the engineering effort and demonstrate that the prototype is
               demonstration-ready at the infrastructure level.
   Weaknesses:
   (1) While the authors evaluated network and latency performance, the anomaly
               detection is not quantitatively validated.
   (2) Critical algorithmic details are missing. For example, the VAE is said to be
               trained “offline on generic facial expression datasets,” but no
               dataset names, sizes, or preprocessing steps are provided. The Siamese
               GRU’s training objective, loss function, and supervision signal are
               entirely omitted.

 Expertise

   Knowledgeable

----------------------------------------------------------------