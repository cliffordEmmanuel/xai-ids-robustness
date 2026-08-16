# Class Balancing: Rationale

How class imbalance is handled: **primarily at the model level via class weights**
(cost-sensitive learning), with dataset resampling **off by default**
(`balance: none`) and available only as an optional sensitivity check.

## The problem

Both datasets are heavily imbalanced toward benign traffic — extreme in
CICIDS2017 (often >99% benign across ~2.8M flows), moderate in UNSW-NB15. Left
unbalanced, a classifier can score high accuracy by predicting "benign" almost
always while barely learning the attack (minority) class. Since detecting — and
then *explaining* — attacks is the whole point, the model must actually learn the
minority class. Balancing the **training** data addresses this.

## The choice, and the honest framing

Imbalance is handled **at the model level using class weights** (e.g. XGBoost's
`scale_pos_weight`, a weighted loss for the MLP), so the default is
`balance: none` — no resampling. Resampling (`undersample_train_only`,
`smote_train_only`) is kept only as an optional sensitivity check.

Being precise about what the literature does and does not support:

- There is **no** general result that any one balancing method is best; the
  imbalance literature is context-dependent. Many IDS studies on CICIDS2017 /
  UNSW-NB15 use SMOTE or SMOTE-hybrids. This is therefore a *reasoned choice*, not
  a claim that "the literature says do X."

Class weighting (over resampling) is justified by a combination of general and
dataset-specific reasons:

1. **Resampling is not a guaranteed improvement** for strong classifiers such as
   XGBoost \parencite{elorSMOTENotSMOTE2022}; class weighting achieves cost-
   sensitivity without altering the data. Resampling methods remain standard
   baselines \parencite{heLearningImbalancedData2009,
   vanhulseExperimentalPerspectivesLearning2007,
   chawlaSMOTESyntheticMinority2002} and are available for comparison.
2. **No data is discarded** (unlike undersampling) and **no synthetic samples are
   created** (unlike SMOTE).
3. **Realism (domain-specific).** SMOTE interpolates new points that, for network
   flows, may not correspond to realisable traffic — problematic for a study
   about explanations of *genuine* attack behaviour. Class weights avoid this
   entirely.
4. **Scale (dataset-specific).** CICIDS2017's size and extreme imbalance
   \parencite{sharafaldinTowardGeneratingNew2018} make SMOTE computationally
   heavy; class weights add negligible cost.
5. **Data quality (dataset-specific).** CICIDS2017 has documented labelling and
   generation issues \parencite{engelenTroubleshootingIntrusionDetection2021};
   not fabricating samples on top of noisy data is safer.

Empirically, model performance (attack recall/F1 on the frozen test set) is
reported so the adequacy of class weighting can be checked directly; resampling is
run only if needed. A related, more important design requirement: the **attack
subset is drawn from correctly-classified attack flows (true positives)**, since
tampering with the explanation of an already-misclassified flow is not a
meaningful "hide the reason" scenario.

## The rule: train only

Balancing is applied to the **training set only**. The **test set is left
untouched and realistically imbalanced**, so evaluation reflects deployment
conditions. Balancing the test set would inflate metrics against a distribution
the model will never see. This is why the pipeline balances train and freezes an
imbalanced test set (and the attack subset is drawn from that frozen test set).

## Trade-offs (state as limitations)

- Undersampling **discards majority data**, losing potentially useful benign
  examples.
- SMOTE (if used) can create **unrealistic synthetic flows** and offers no
  guaranteed gain.
- The choice is configurable (`none` / `undersample_train_only` /
  `smote_train_only`); reporting a SMOTE sensitivity check strengthens the claim.

## References

- He & Garcia (2009), *Learning from Imbalanced Data*, IEEE TKDE.
- van Hulse, Khoshgoftaar & Napolitano (2007), *Experimental Perspectives on
  Learning from Imbalanced Data*, ICML.
- Chawla et al. (2002), *SMOTE: Synthetic Minority Over-sampling Technique*, JAIR.
- Elor & Averbuch-Elor (2022), *To SMOTE, or not to SMOTE?*, arXiv:2201.08528.
- Sharafaldin, Lashkari & Ghorbani (2018), *Toward Generating a New Intrusion
  Detection Dataset...* (CICIDS2017), ICISSP.
- Engelen, Rimmer & Joosen (2021), *Troubleshooting an Intrusion Detection
  Dataset: the CICIDS2017 Case Study*, IEEE S&P Workshops.
