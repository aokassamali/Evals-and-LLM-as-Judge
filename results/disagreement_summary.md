# Disagreement summary (ref = scifact_dev200_gpt-5.2_t0.0)

Shared IDs across gold + 5 runs: **198**

## scifact_dev200_deepseek-r1_8b_t0.0 vs scifact_dev200_gpt-5.2_t0.0

- n: **198**
- label flips: `{'NEI<->(S/R)': 77, 'same': 119, 'SUPPORTS<->REFUTES': 2}`
- evidence behavior: `{'cite_all': 2, 'normal': 85, 'empty': 95, 'shotgun': 16}`
- claim flags among disagreements: `{'numeric': 13, 'causal': 22, 'hedged': 1}`
- correct label but wrong evidence (vs gold): **1**

### Examples: SUPPORTS<->REFUTES

- `scifact_dev_1290_4687948` gold=SUPPORTS  ref=SUPPORTS  scifact_dev200_deepseek-r1_8b_t0.0=REFUTES  other_ev=1 (normal)
  - There is an inverse relationship between hip fractures and statin use.
- `scifact_dev_249_1568684` gold=REFUTES  ref=REFUTES  scifact_dev200_deepseek-r1_8b_t0.0=SUPPORTS  other_ev=2 (normal)
  - Chenodeosycholic acid treatment reduces whole-body energy expenditure.

### Examples: NEI<->(S/R)

- `scifact_dev_475_18678095` gold=SUPPORTS  ref=NEI  scifact_dev200_deepseek-r1_8b_t0.0=SUPPORTS  other_ev=3 (normal)
  - Glycolysis is one of the primary glycometabolic pathways in cells.
- `scifact_dev_100_4381486` gold=SUPPORTS  ref=SUPPORTS  scifact_dev200_deepseek-r1_8b_t0.0=NEI  other_ev=8 (cite_all)
  - All hematopoietic stem cells segregate their chromosomes randomly.
- `scifact_dev_513_13230773` gold=REFUTES  ref=REFUTES  scifact_dev200_deepseek-r1_8b_t0.0=NEI  other_ev=0 (empty)
  - High cardiopulmonary fitness causes increased mortality rate.
- `scifact_dev_274_11614737` gold=REFUTES  ref=REFUTES  scifact_dev200_deepseek-r1_8b_t0.0=NEI  other_ev=5 (normal)
  - Combination nicotine replacement therapies with varenicline or bupropion lead to significantly higher long-term abstinence rates at 52 weeks than varenicline monotherapy.
- `scifact_dev_501_17930286` gold=SUPPORTS  ref=SUPPORTS  scifact_dev200_deepseek-r1_8b_t0.0=NEI  other_ev=8 (shotgun)
  - Headaches are not correlated with cognitive impairment.

## scifact_dev200_gemma3_4b_t0.0 vs scifact_dev200_gpt-5.2_t0.0

- n: **198**
- label flips: `{'same': 74, 'NEI<->(S/R)': 91, 'SUPPORTS<->REFUTES': 33}`
- evidence behavior: `{'normal': 167, 'shotgun': 18, 'empty': 9, 'cite_all': 4}`
- claim flags among disagreements: `{'causal': 35, 'numeric': 24, 'hedged': 4}`
- correct label but wrong evidence (vs gold): **5**

### Examples: SUPPORTS<->REFUTES

- `scifact_dev_1336_27910499` gold=REFUTES  ref=REFUTES  scifact_dev200_gemma3_4b_t0.0=SUPPORTS  other_ev=1 (normal)
  - UCB T cells reduce TCR diversity after transplantation.
- `scifact_dev_1359_11614737` gold=REFUTES  ref=REFUTES  scifact_dev200_gemma3_4b_t0.0=SUPPORTS  other_ev=2 (normal)
  - Varenicline monotherapy is more effective after 12 weeks of treatment compared to combination nicotine replacement therapies with varenicline or bupropion.
- `scifact_dev_1303_12631697` gold=REFUTES  ref=REFUTES  scifact_dev200_gemma3_4b_t0.0=SUPPORTS  other_ev=4 (normal)
  - Tirasemtiv has no effect on fast-twitch muscle.
- `scifact_dev_578_8764879` gold=REFUTES  ref=REFUTES  scifact_dev200_gemma3_4b_t0.0=SUPPORTS  other_ev=5 (shotgun)
  - In mouse models, the loss of CSF1R facilitates MOZ-TIF2-induced leuekmogenesis.
- `scifact_dev_1137_33370` gold=REFUTES  ref=REFUTES  scifact_dev200_gemma3_4b_t0.0=SUPPORTS  other_ev=5 (shotgun)
  - TNFAIP3 is a tumor suppressor in glioblastoma.

### Examples: NEI<->(S/R)

- `scifact_dev_1274_4406819` gold=SUPPORTS  ref=NEI  scifact_dev200_gemma3_4b_t0.0=SUPPORTS  other_ev=3 (normal)
  - The tip of the inner tube of the toxic type VI secretion system (T6SS) antibacterial effector in Escherichia coli (E. coli) carries toxic effector proteins.
- `scifact_dev_820_8646760` gold=NEI  ref=NEI  scifact_dev200_gemma3_4b_t0.0=SUPPORTS  other_ev=2 (normal)
  - N-terminal cleavage increases success identifying transcription start sites.
- `scifact_dev_1280_4387784` gold=NEI  ref=NEI  scifact_dev200_gemma3_4b_t0.0=SUPPORTS  other_ev=2 (normal)
  - The ureABIEFGH gene cluster encodes urease maturation proteins : UreD/UreH, UreE, UreF, and UreG.
- `scifact_dev_279_14376683` gold=SUPPORTS  ref=NEI  scifact_dev200_gemma3_4b_t0.0=SUPPORTS  other_ev=1 (normal)
  - Commelina yellow mottle virus' (ComYMV) genome consists of 7489 baise pairs.
- `scifact_dev_1316_27910499` gold=NEI  ref=NEI  scifact_dev200_gemma3_4b_t0.0=SUPPORTS  other_ev=2 (normal)
  - Transferred UCB T cells acquire a memory-like phenotype in recipients.

## scifact_dev200_llama3.1_8b_t0.0 vs scifact_dev200_gpt-5.2_t0.0

- n: **198**
- label flips: `{'same': 73, 'NEI<->(S/R)': 98, 'SUPPORTS<->REFUTES': 27}`
- evidence behavior: `{'normal': 166, 'empty': 5, 'cite_all': 15, 'shotgun': 12}`
- claim flags among disagreements: `{'causal': 34, 'numeric': 21, 'hedged': 4}`
- correct label but wrong evidence (vs gold): **15**

### Examples: SUPPORTS<->REFUTES

- `scifact_dev_274_11614737` gold=REFUTES  ref=REFUTES  scifact_dev200_llama3.1_8b_t0.0=SUPPORTS  other_ev=1 (normal)
  - Combination nicotine replacement therapies with varenicline or bupropion lead to significantly higher long-term abstinence rates at 52 weeks than varenicline monotherapy.
- `scifact_dev_1359_11614737` gold=REFUTES  ref=REFUTES  scifact_dev200_llama3.1_8b_t0.0=SUPPORTS  other_ev=2 (normal)
  - Varenicline monotherapy is more effective after 12 weeks of treatment compared to combination nicotine replacement therapies with varenicline or bupropion.
- `scifact_dev_1320_16284655` gold=REFUTES  ref=REFUTES  scifact_dev200_llama3.1_8b_t0.0=SUPPORTS  other_ev=2 (normal)
  - Transplanted human glial progenitor cells are incapable of forming a neural network with host animals' neurons.
- `scifact_dev_578_8764879` gold=REFUTES  ref=REFUTES  scifact_dev200_llama3.1_8b_t0.0=SUPPORTS  other_ev=1 (normal)
  - In mouse models, the loss of CSF1R facilitates MOZ-TIF2-induced leuekmogenesis.
- `scifact_dev_1137_33370` gold=REFUTES  ref=REFUTES  scifact_dev200_llama3.1_8b_t0.0=SUPPORTS  other_ev=3 (normal)
  - TNFAIP3 is a tumor suppressor in glioblastoma.

### Examples: NEI<->(S/R)

- `scifact_dev_1179_31272411` gold=NEI  ref=NEI  scifact_dev200_llama3.1_8b_t0.0=SUPPORTS  other_ev=1 (normal)
  - The PRR MDA5 has a central DExD/H RNA helices domain.
- `scifact_dev_593_19675911` gold=SUPPORTS  ref=REFUTES  scifact_dev200_llama3.1_8b_t0.0=NEI  other_ev=10 (cite_all)
  - Incidence of heart failure decreased by 10% in women since 1979.
- `scifact_dev_1266_37480103` gold=SUPPORTS  ref=NEI  scifact_dev200_llama3.1_8b_t0.0=SUPPORTS  other_ev=2 (normal)
  - The risk of breast cancer among parous women increases with placental weight of pregnancies, and this association is strongest for premenopausal breast cancer.
- `scifact_dev_238_2251426` gold=NEI  ref=NEI  scifact_dev200_llama3.1_8b_t0.0=SUPPORTS  other_ev=2 (normal)
  - Cells undergoing methionine restriction may activate miRNAs.
- `scifact_dev_183_12827098` gold=REFUTES  ref=NEI  scifact_dev200_llama3.1_8b_t0.0=REFUTES  other_ev=3 (normal)
  - Bone marrow cells contribute to adult macrophage compartments.

## scifact_dev200_qwen3_8b_t0.0 vs scifact_dev200_gpt-5.2_t0.0

- n: **198**
- label flips: `{'SUPPORTS<->REFUTES': 14, 'same': 136, 'NEI<->(S/R)': 48}`
- evidence behavior: `{'normal': 127, 'empty': 62, 'shotgun': 7, 'cite_all': 2}`
- claim flags among disagreements: `{'numeric': 16, 'causal': 18, 'hedged': 2}`
- correct label but wrong evidence (vs gold): **2**

### Examples: SUPPORTS<->REFUTES

- `scifact_dev_903_10648422` gold=REFUTES  ref=REFUTES  scifact_dev200_qwen3_8b_t0.0=SUPPORTS  other_ev=2 (normal)
  - PD-1 triggering on monocytes reduces IL-10 production by monocytes.
- `scifact_dev_742_32159283` gold=SUPPORTS  ref=SUPPORTS  scifact_dev200_qwen3_8b_t0.0=REFUTES  other_ev=1 (normal)
  - Macrolides have no protective effect against myocardial infarction.
- `scifact_dev_57_4709641` gold=REFUTES  ref=REFUTES  scifact_dev200_qwen3_8b_t0.0=SUPPORTS  other_ev=5 (shotgun)
  - APOE4 expression in iPSC-derived neurons increases AlphaBeta production and tau phosphorylation, delaying GABA neuron degeneration.
- `scifact_dev_1088_37549932` gold=REFUTES  ref=REFUTES  scifact_dev200_qwen3_8b_t0.0=SUPPORTS  other_ev=2 (normal)
  - Silencing of Bcl2 is important for the maintenance and progression of tumors.
- `scifact_dev_100_4381486` gold=SUPPORTS  ref=SUPPORTS  scifact_dev200_qwen3_8b_t0.0=REFUTES  other_ev=2 (normal)
  - All hematopoietic stem cells segregate their chromosomes randomly.

### Examples: NEI<->(S/R)

- `scifact_dev_552_1471041` gold=NEI  ref=NEI  scifact_dev200_qwen3_8b_t0.0=SUPPORTS  other_ev=2 (normal)
  - IgA plasma cells that are specific for transglutaminase 2 accumulate in the duodenal mucosa on commencement of a gluten-free diet.
- `scifact_dev_133_16280642` gold=SUPPORTS  ref=SUPPORTS  scifact_dev200_qwen3_8b_t0.0=NEI  other_ev=2 (normal)
  - Assembly of invadopodia is triggered by focal generation of phosphatidylinositol-3,4-biphosphate and the activation of the nonreceptor tyrosine kinase Src.
- `scifact_dev_208_13519661` gold=SUPPORTS  ref=REFUTES  scifact_dev200_qwen3_8b_t0.0=NEI  other_ev=2 (normal)
  - CHEK2 is not associated with breast cancer.
- `scifact_dev_1163_15305881` gold=SUPPORTS  ref=SUPPORTS  scifact_dev200_qwen3_8b_t0.0=NEI  other_ev=3 (normal)
  - The DdrB protein from Deinococcus radiodurans is an alternative SSB.
- `scifact_dev_879_8426046` gold=REFUTES  ref=SUPPORTS  scifact_dev200_qwen3_8b_t0.0=NEI  other_ev=3 (normal)
  - Occupancy of ribosomes by IncRNAs do not make functional peptides.

