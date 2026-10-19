"""Narrative content for Chapter 2 of the Phase I report.

Written in the LITERATURE SURVEY OVERVIEW format: continuous narrative prose,
author-year in-text citations, each study described in terms of its method, the
strength it demonstrates and the limitation it leaves open. No bullet lists.
"""

OVERVIEW_HEADING = "2.1 Literature Survey Overview"

OVERVIEW = [
    "Ransomware detection has been an active research area for more than a decade, and the "
    "published work can be organised into four families that differ in the vantage point from "
    "which evidence is collected. Host-based detectors observe the file system and the processes "
    "of the machine that is being protected. Sandbox-based analysers characterise malware offline "
    "by executing it in an instrumented environment and recording what it does. Network-level "
    "detectors inspect traffic for the command-and-control and file-sharing patterns that "
    "crypto-ransomware exhibits. Server-side log-based detectors infer encryption from the "
    "metadata of file operations that a shared file server records while it is under attack. Each "
    "family rests on a different kind of evidence, and each family inherits the deployment "
    "constraints of the component it observes.",

    "Across these families a consistent methodological pattern is visible. The first generation "
    "of detectors, represented by CryptoLock (Scaife et al., 2016) and ShieldFS (Continella et "
    "al., 2016), combined traces of file-system operations with Shannon-entropy measurements of "
    "file contents and acted within seconds of detecting an encryption burst. The second "
    "generation, represented by UNVEIL (Kharaz et al., 2016) and EldeRan (Sgandurra et al., 2016), "
    "moved away from reading file content and towards purely behavioural signals, learning the "
    "sequence and the statistical signature of file, registry and API activity. The third strand "
    "of work moved the observation point off the machine altogether and inspected network traffic "
    "characteristics in a software-defined fabric (Cabaj et al., 2018). The most recent work, by "
    "Aranyi, Miseta and Szucs (2026), moved the observation point from the individual endpoint to "
    "the shared file server, and demonstrated that a small set of aggregate file-operation counts "
    "collected per second is sufficient for strong detection.",

    "Two limitations recur throughout the literature. The first is a deployment limitation: every "
    "detector that reads file content or installs a kernel component requires an agent on every "
    "protected machine, and such agents are frequently unavailable precisely where file servers "
    "are most exposed, namely on locked-down endpoints and on unmanaged devices. The second is a "
    "representation limitation that is less often stated but is mathematically unavoidable. When a "
    "detector reduces a stream of file operations to a small vector of per-second counts, it fixes "
    "a horizon, and any attack window and any benign window that happen to produce the same "
    "counts are identical to that representation and cannot be separated by any classifier of that "
    "representation, however well trained. Part of the available signal is therefore lost before "
    "the classifier is ever applied.",

    "This chapter surveys six representative studies in the narrative style of a literature "
    "overview: for each study it states what was proposed, what the method does, what the "
    "evaluation demonstrated and what the study leaves unresolved. The survey is followed by a "
    "comparative analysis, the consolidated literature survey table, the research gap it exposes, "
    "the ideas adopted from the literature, and the resulting problem selection.",
]

PAPER1_HEADING = "2.2 Paper I – CryptoLock: Stopping Ransomware Attacks on User Data"

PAPER1 = [
    "Scaife, Carter, Traynor and Butler (2016) proposed CryptoLock, a system that detects "
    "ransomware by monitoring file-system activity together with the entropy of file contents in "
    "real time and restores the affected files before encryption completes. The central intuition "
    "of CryptoLock is that encryption is a gradual, observable process rather than an instantaneous "
    "event. A background process samples the entropy of files that a monitored application has "
    "open, and the pair of rising entropy together with a burst of file operations is treated as "
    "the signature of an encryptor. When that signature is observed, the files that are being "
    "encrypted are copied to a shadow backup before their contents are overwritten, so that the "
    "victim is not required to pay in order to recover the data.",

    "The method is significant because it establishes the central claim on which all later work "
    "rests: file-system behaviour alone carries enough signal to identify an encryptor in its "
    "early phase. CryptoLock also shows that the useful detection point is the beginning of the "
    "encryption run rather than its completion, since encrypted data is usually unrecoverable "
    "whereas detected but unencrypted data can simply be protected. Evaluation against real "
    "families confirmed detection within a small number of file operations.",

    "Nevertheless, CryptoLock depends on inspecting the contents of files that are open on the "
    "protected host, and this imposes two costs. Read access to file content is exactly what "
    "privacy regulation and internal data-classification policy are designed to restrict, and in "
    "many enterprise environments a security tool that continuously reads user data is difficult "
    "to deploy. In addition, the mechanism is per-host: CryptoLock protects the machine on which "
    "it runs, so an organisation must install and maintain the software on every endpoint that "
    "holds valuable files, and the tool provides no visibility into a file server on behalf of a "
    "client that has no agent installed. Both costs are avoided by moving the observation point "
    "to a component that already sees every file operation performed against a shared store.",

    "For the present project, CryptoLock contributes two durable insights: the timing of "
    "detection matters more than the sophistication of the classifier, and behavioural and "
    "content evidence carry partially independent information. This project keeps only the "
    "behavioural half of that pair in order to preserve the metadata-only character of "
    "server-side logging, and must therefore recover the lost accuracy through the temporal "
    "representation.",
]

PAPER2_HEADING = "2.3 Paper II – ShieldFS: A Self-Healing, Ransomware-Aware Filesystem"

PAPER2 = [
    "Continella, Guagnelli, Zingaro, De Pasquale, Barenghi, Zanero and Maggi (2016) presented "
    "ShieldFS, a self-healing file system implemented as a Windows minifilter driver. ShieldFS "
    "intercepts every file operation that reaches the lower layers of the file-system stack, "
    "extracts a small set of low-level features from the intercepted request, and passes them to a "
    "rule engine that decides whether the operation is consistent with ransomware behaviour. When "
    "the rule engine declares an encryption run, ShieldFS rolls the file-system state back to a "
    "consistent earlier point using file-system journal entries, so that the damage is undone "
    "rather than merely reported.",

    "ShieldFS strengthens the CryptoLock result in two ways. Because the observation point is the "
    "file-system driver itself, the intercepted features describe the semantics of each request "
    "rather than the sampling behaviour of a monitoring process, which reduces the chance that an "
    "encryptor operating between two samples escapes observation. Because recovery is built into "
    "the file system, the response is automatic. Experiments against several families detected "
    "the attempt early enough for lossless restoration.",

    "The cost of this design is a dependency on privileged, platform-specific software. A "
    "minifilter driver must be compiled, signed and installed on every host, and it is tied to a "
    "particular operating system and file-system implementation, so maintenance and portability "
    "become real engineering obligations rather than administrative ones. The system also inherits "
    "the content-inspection constraint of its predecessor, since the entropy of file content is "
    "one of the features on which the rule engine depends. As a result ShieldFS is well suited to "
    "a managed desktop estate and poorly suited to an environment in which a shared file server "
    "serves many clients that cannot be instrumented, which is the environment that this project "
    "addresses.",

    "The transferable contribution of ShieldFS to the present work is architectural rather than "
    "algorithmic: the earliest reliable point at which encryption can be observed is the "
    "file-system operation itself. This project keeps that principle and changes the vantage "
    "point, reading the operations from the metadata stream that the file server already "
    "publishes so that no privileged component is needed on any client.",
]

PAPER3_HEADING = "2.4 Paper III – UNVEIL: Large-Scale Automated Detection of Ransomware"

PAPER3 = [
    "Kharaz, Arshad, Mulliner, Robertson and Kirda (2016) presented UNVEIL, a system for the "
    "large-scale automated detection of ransomware. UNVEIL executes each sample in a sandbox and "
    "records the sequence of file-system operations that the sample performs, together with a "
    "screen-lock signal that indicates when the sample locks the desktop. The recorded sequences "
    "are used to train a classifier, and the classifier is then evaluated over a large corpus of "
    "real samples to measure how reliably it separates ransomware from benign software.",

    "UNVEIL makes an important methodological point about the difference between reporting a high "
    "detection rate and demonstrating one: because the system is automated and applied to a large "
    "corpus, the figures come from the whole corpus rather than a handful of favourable samples, "
    "and the paper is candid about the behaviour that remains difficult. It also confirms that the "
    "order and shape of the sequence, not merely the total count, carries discriminating "
    "information, since a benign application that saves many files and an encryptor that rewrites "
    "many files produce similar totals but very different sequences.",

    "The principal limitation of UNVEIL for the present purpose is that it is an offline analysis "
    "technique rather than a live monitor. It characterises a sample by executing it under "
    "instrumentation, which means that it can be applied before or during deployment of a "
    "candidate program but cannot be applied to an encryption run that is already in progress on a "
    "production server. Its features are also collected inside a sandbox, which requires the kind "
    "of deep instrumentation that a production file server will not tolerate. In other words, "
    "UNVEIL answers the question of what ransomware looks like, but not the question of what to do "
    "about an attack that is happening now.",

    "For this project, UNVEIL contributes the observation that the shape of the operation stream over "
    "time matters more than its volume at any single instant. This is the direct justification for "
    "the multi-scale trailing temporal features used in Chapter 5.",
]

PAPER4_HEADING = "2.5 Paper IV – EldeRan: Automated Dynamic Analysis of Ransomware"

PAPER4 = [
    "Sgandurra, Munoz-Gonzalez, Mohsen and Lupu (2016) investigated the benefits, limitations and "
    "detection uses of automated dynamic analysis of ransomware. EldeRan executes samples in a "
    "sandbox and extracts roughly twenty behavioural features spanning three sources: file input "
    "and output activity, registry activity, and API-call activity. These features are then used "
    "to train and compare classifiers, and the paper analyses explicitly what each source of "
    "evidence contributes to the detection result.",

    "The value of EldeRan lies in its systematic treatment of the feature space. By decomposing "
    "behaviour into file, registry and API activity and measuring the contribution of each "
    "group, the work shows that file I/O alone suffices for a strong result in most cases and "
    "that the other sources add robustness mainly in ambiguous cases. This matters for server-side "
    "deployment, because it implies that a detector restricted to file operations gives up only "
    "largely redundant evidence. EldeRan also shows that behavioural features generalise across "
    "families better than strings or byte n-grams, which change with every repack.",

    "The limitation of EldeRan with respect to this project is again one of vantage point. The "
    "features are extracted from inside an instrumented sandbox during controlled execution, which "
    "requires deep telemetry that cannot be collected from a live file server without intrusive "
    "software, and the analysis is offline rather than operational. Registry and API-call evidence "
    "in particular is unavailable to a server-side observer by construction, since the server sees "
    "file operations and nothing about the client processes that caused them.",

    "The present work takes from EldeRan the principle that behavioural features generalise, and "
    "resolves the telemetry problem by moving to an evidence source that requires no client "
    "instrumentation: the server-side log contains create, rename and delete operations produced "
    "by every client, instrumented or not.",
]

PAPER5_HEADING = "2.6 Paper V – Network-Level Crypto-Ransomware Detection"

PAPER5 = [
    "Cabaj, Gregorczyk and Mazurczyk (2018) proposed a network-level method for detecting "
    "crypto-ransomware using the characteristics of HTTP traffic. In a software-defined network, "
    "the method analyses flow statistics derived from HTTP request and response headers, "
    "exploiting the observation that ransomware families commonly differ from benign applications "
    "in the number of distinct destinations contacted, in the size and regularity of objects "
    "requested, and in the proportion of requests that retrieve rather than post content. These "
    "features are inexpensive to collect and do not require inspection of the payload.",

    "This approach complements rather than replaces the endpoint methods and has real operational "
    "advantages: because the collection point is a network device, the detector sees traffic from "
    "every host on the segment without installing anything on those hosts, and it can be deployed "
    "as a service function on existing switching infrastructure. The paper reports high detection "
    "rates at a cost low enough for line-rate deployment.",

    "However, a network vantage point is blind to the case that matters most for this project. "
    "Ransomware that encrypts files already present on a host, or on a mounted file share, "
    "generates little or no network traffic beyond the initial delivery of the payload, so a "
    "network detector may see nothing at all during the encryption run itself. The method also "
    "requires visibility into the network fabric and becomes unreliable behind encryption or "
    "tunnelling, and its features are weakest exactly where an attacker has most control, namely "
    "in the low-and-slow profile that deliberately mimics legitimate traffic. For a shared file "
    "server, the file operations themselves are a strictly better evidence source because they are "
    "produced locally by the attack itself and cannot be suppressed by the attacker without "
    "ceasing to encrypt.",

    "The present work therefore treats network-level detection as complementary context and "
    "confines its own observation to the server-side file-operation stream, adopting the "
    "methodological lesson that features should be chosen so that the attacker cannot cheaply "
    "suppress them.",
]

PAPER6_HEADING = "2.7 Paper VI – The Base Paper: Server-Side File-Operation Logs"

PAPER6 = [
    "Aranyi, Miseta and Szucs (2026) constructed a virtualised small- and medium-enterprise file "
    "server environment, drove it with four automated user profiles that reproduce the working "
    "patterns of typical office roles under on-site and remote conditions, and executed five real "
    "ransomware samples against it at randomly chosen times. File create, rename and delete "
    "operations were collected by Fluent Bit and stored in InfluxDB, aggregated into fixed "
    "one-second windows, and classified by five classical machine-learning models after "
    "hyperparameters were tuned by grid search. The best model, XGBoost, reached a sensitivity of "
    "91.87 per cent on the executed samples, detecting 192 of 209 attack seconds, at "
    "sub-millisecond inference cost, and this was achieved without any client-side access.",

    "This study is the direct antecedent of the present work and its most important contribution is "
    "architectural. It demonstrates that the file server is a sufficient observation point: the "
    "three metadata event types that a server already logs are enough to separate encryption from "
    "ordinary office activity with high accuracy, no file content is read, no endpoint agent is "
    "required, and the inference cost is small enough for continuous evaluation.",

    "Two of the limitations that the study itself identifies are structural rather than incidental, "
    "and this project treats them as its point of departure. The first is the ceiling implied by a "
    "three-count representation. If an attack window and a benign window are assigned the same "
    "three integers, they are the same point in feature space and no classifier of those features "
    "can separate them; the paper's own tables contain attack windows whose counts lie inside the "
    "benign envelope, so part of the signal is unreachable no matter how the classifier is tuned. "
    "The second is that the design leaves several important questions unmeasured. The paper claims "
    "in its abstract that the detector is family-independent but never evaluates transfer to a "
    "family that was not present in training; it names adversarial rate limiting, or low-and-slow "
    "encryption, as the principal limitation without measuring the rate at which detection fails; "
    "it uses a fixed decision threshold of 0.5 despite the ten-to-one asymmetry between the cost "
    "of a missed attack and the cost of a false alarm; and it reports a single run, so the ranking "
    "of the five models is a point estimate with no dispersion attached to it.",

    "Finally, the base paper fixes a single one-second aggregation horizon. This choice determines "
    "the entire behaviour of the detector: it makes burst encryption obvious, and it makes slow "
    "encryption invisible, because an encryptor that renames and deletes a few files per second "
    "produces counts that fall inside the ordinary range of office activity. The present project "
    "argues that this is an artefact of the horizon rather than an inherent property of the "
    "evidence, and that the remedy is to give the classifier access to the recent history of the "
    "operation stream rather than to a single instant of it.",
]

SURVEY_TABLE_HEADING = "2.9 Literature Survey Table"

SURVEY_TABLE_CAPTION = "Table 2.2: Literature survey table"

SURVEY_TABLE_INTRO = [
    "Table 2.2 consolidates the surveyed work in the standard survey-table format, recording for "
    "each study the title, the authors, the year of publication, the algorithm or method used and "
    "the key finding, together with the way in which the present project extends or answers that "
    "finding.",
]

SURVEY_TABLE_HEADERS = [
    "S.No", "Title", "Author(s)", "Year", "Algorithm Used", "Key Findings",
]

SURVEY_TABLE_ROWS = [
    [
        "1",
        "CryptoLock (and drop it): Stopping Ransomware Attacks on User Data",
        "Scaife et al.",
        "2016",
        "Real-time file-system monitoring with Shannon-entropy based detection",
        "I concluded that CryptoLock proves file-system behaviour combined with content entropy "
        "identifies an encryptor within the first few operations, and that detection before "
        "encryption completes is what preserves the data. It needs a per-host agent and read "
        "access to content, so I aim to keep only its behavioural half and to recover the lost "
        "accuracy through temporal context.",
    ],
    [
        "2",
        "ShieldFS: A Self-Healing, Ransomware-Aware Filesystem",
        "Continella et al.",
        "2016",
        "File-system minifilter rule engine with entropy features and journal rollback",
        "I concluded that ShieldFS shows the earliest reliable observation point is the file-system "
        "operation itself, and that automatic rollback is preferable to alerting alone. Its "
        "minifilter driver must be compiled, signed and installed on every client, so I aim to "
        "keep its principle but read the operations from the server log, which requires no "
        "client software at all.",
    ],
    [
        "3",
        "UNVEIL: A Large-Scale, Automated Approach to Detecting Ransomware",
        "Kharaz et al.",
        "2016",
        "Operation-sequence learning with screen-lock signal classification",
        "I concluded that UNVEIL demonstrates, over a large real corpus, that the order and shape "
        "of the operation sequence discriminates ransomware far better than the total number of "
        "operations. It is an offline sandbox analysis, so I aim to convert its sequential "
        "insight into a live detector using multi-scale trailing aggregates.",
    ],
    [
        "4",
        "Automated Dynamic Analysis of Ransomware: Benefits, Limitations and Use for Detection",
        "Sgandurra et al.",
        "2016",
        "Behavioural classification on about twenty file, registry and API-call features",
        "I concluded that EldeRan establishes behavioural features as generalisable across "
        "families and shows by ablation that file input and output activity alone carries most of "
        "the detection signal. Its registry and API evidence needs deep instrumentation, so I aim "
        "to use only the file-operation subset the server already produces.",
    ],
    [
        "5",
        "SDN-Based Crypto-Ransomware Detection Using HTTP Traffic Characteristics",
        "Cabaj et al.",
        "2018",
        "Classification of HTTP flow and header statistics",
        "I concluded that network telemetry is inexpensive and needs no endpoint agent, but it is "
        "blind to a purely local encryption run and weakest against low-and-slow behaviour. I "
        "therefore aim to observe file-operation metadata, which the attacker must produce in "
        "order to encrypt at all.",
    ],
    [
        "6",
        "Ransomware Detection Based on Server-Side File Operation Logs Using Machine Learning",
        "Aranyi et al.",
        "2026",
        "Decision Tree, Random Forest, SVM, AdaBoost and XGBoost on three per-second counts",
        "I concluded that server-side create, rename and delete logs are sufficient for 91.87 per "
        "cent sensitivity with sub-millisecond inference and no client access. Its three-count "
        "representation has a provable ceiling, and transfer, evasion, drift and dispersion are "
        "untested, so I aim to replace the counts with 26 multi-scale trailing features and to "
        "measure all four.",
    ],
    [
        "7",
        "Adaptive Server-Side Ransomware Detection (ASSRD) - This Project",
        "Mega B, Methi M, Nikita Ray B and Nithya Sri G",
        "2026",
        "Multi-scale trailing temporal features with cost-sensitive Decision Tree, Random Forest, "
        "calibrated SVM, AdaBoost and XGBoost, cost-selected threshold and PSI drift monitoring",
        "I concluded in this project that temporal context raises AUROC from 0.9564 to 0.9997 and "
        "reduces cost from 2504 to 226, that cost-selected thresholding lowers cost further to "
        "62, and that the detector is defeated below roughly fifteen files per second. I aim to "
        "extend it with executed samples, in-place modification events and non-count features.",
    ],
]

COMPARATIVE_HEADING = "2.8 Comparative Analysis"

COMPARATIVE_INTRO = [
    "The surveyed works differ along four axes that determine what they observe and what they prove: "
    "the telemetry they read, the way they represent it, whether they operate live, and whether "
    "they were evaluated on real samples. Table 2.1 places the studies against these axes.",
]

RESEARCH_GAP_HEADING = "2.10 Research Gap"

RESEARCH_GAP = [
    "Read together, the six studies expose four gaps that no surveyed work closes. The first is "
    "the count-only ceiling of the base paper. Its feature vector is three integers per one-second "
    "window, and because two windows with identical counts are indistinguishable to any classifier "
    "of that vector, the maximum achievable separability is bounded by the proportion of attack "
    "windows inside the benign envelope. This is a property of the representation, so it cannot "
    "be closed by tuning or by adding data; it can only be closed by changing the representation.",

    "The second gap concerns generalisation. The base paper asserts family-independent detection "
    "without measuring it, and no surveyed work evaluates transfer to a wholly unseen family using "
    "an episode-aware split, so a reported sensitivity cannot be distinguished from one fitted to "
    "the families on which it was measured. The third gap concerns adversarial behaviour: "
    "rate-limited encryption, in which an attacker deliberately stays below any burst threshold, is "
    "named as the principal weakness of count-based detection, yet no surveyed work measures the "
    "rate at which detection collapses.",

    "The fourth gap is methodological and affects every study in the survey: results are reported "
    "from a single run with a single random seed, so the ranking of models is a point estimate "
    "whose confidence interval is never stated, and a difference of a few percentage points on one "
    "split is not evidence that one classifier is better. Related to this, no surveyed work "
    "reports attribution, so an operator is not told which property of a window caused an alarm, "
    "and none monitors concept drift, so a deployed detector can decay silently.",
]

IDEAS_HEADING = "2.11 Ideas Obtained from the Literature"

IDEAS = [
    "Five ideas were adopted from the survey and shaped the design of this project. The first, "
    "from Scaife et al. (2016) and Continella et al. (2016), is that file-operation behaviour is a "
    "strong and intrinsically privacy-preserving signal, so the project adopts server-side create, "
    "rename and delete events as its only input and reads no file content. The second, from Kharaz "
    "et al. (2016) and Sgandurra et al. (2016), is that the shape of the operation stream over time "
    "is more informative than its volume at any single instant, which motivates replacing the "
    "per-second counts with multi-scale trailing aggregates. The third, following Aranyi et al. "
    "(2026), is that a fixed set of classical classifiers provides broad coverage of inductive "
    "biases at negligible cost, so all five are evaluated rather than one being assumed superior.",

    "The fourth idea is that the two error types of a ransomware detector are not equally "
    "important. A missed attack can cost an entire file store while a false alarm costs only the "
    "analyst time required to dismiss it, so the evaluation is made cost-sensitive with a missed "
    "attack charged ten times a false alarm and the operating threshold is chosen to minimise that "
    "cost rather than fixed at 0.5. The fifth is that a deployed detector must be monitored and "
    "explained: it must report which observable property triggered an alarm and must detect the "
    "point at which its assumptions no longer hold. Table 2.3 summarises the surveyed methods and "
    "their principal limitations.",
]

PROBLEM_HEADING = "2.12 Problem Selection and the Final Project"

PROBLEM = [
    "Combining the base paper with the gaps above leads to a single well-posed problem. The project "
    "keeps server-side file-operation logging as its telemetry, the only surveyed evidence source "
    "that requires no client instrumentation and no access to file content. It replaces the three "
    "per-second counts with multi-scale trailing temporal features so that the count-only ceiling "
    "is removed at its cause rather than worked around at the model. It retains the "
    "five-classifier comparison, makes it cost-sensitive, and selects the operating threshold by "
    "minimising cost on a held-out split instead of fixing it at 0.5. Finally it performs the "
    "experiments the base paper names but does not execute: leave-one-family-out transfer, an "
    "evasion study, a multi-seed study, and a drift study that measures recovery. The resulting "
    "system is the Adaptive Server-Side Ransomware Detection (ASSRD) framework, described in the "
    "following chapters.",
]