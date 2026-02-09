# Methodology

This document describes the statistical design of the project: how the target is defined, how data are split, which features and models are used, how models are compared and turned into decisions, and how they are explained and monitored. Implementation details live in the docstrings of `src/credit_risk`; this document states the reasoning and the assumptions behind them.

## 1. Problem and notation

For applicant $i$ with attribute vector $x_i \in \mathbb{R}^p$, the outcome is $y_i = 1$ if the borrower was 90 or more days past due within two years of observation and $y_i = 0$ otherwise. The quantity of interest is the probability of default

$$\mathrm{PD}(x) = \Pr(Y = 1 \mid X = x),$$

estimated by a model $\hat p(x)$. A lending policy approves an applicant when $\hat p(x) < t$ for a cut-off $t$. Throughout, $n_1$ and $n_0$ denote the numbers of defaulters and non-defaulters in a sample, and $\pi = n_1/(n_0+n_1)$ the sample default rate.

Two properties of $\hat p$ are evaluated separately. Discrimination is the ability to rank defaulters above non-defaulters and is invariant to monotone transforms of $\hat p$. Calibration is the agreement between $\hat p$ and observed default frequencies and is what pricing, provisioning and cost-based cut-offs depend on.

## 2. Data and validation design

**Source.** The Credit Risk Benchmark Dataset has 16,714 borrowers and ten numeric attributes: revolving utilization, age, three counts of past-due episodes (30-59, 60-89 and 90 or more days), debt ratio, monthly income, number of open credit lines, number of real estate loans and number of dependents. There are no application dates, no loan amounts and no free text.

**Validation.** Every input file passes `data.schema.validate_frame`. Hard violations (missing columns, non-numeric types, negative counts, a non-binary target) stop the run. Soft violations (for example ages above 100 or utilization above 10) are logged, because such values occur in real bureau data and removing them silently would change the population.

**Splits.** The default design is a stratified 80/20 train-test split with a fixed seed. Model selection and hyperparameter search only ever see the training part. Cross-validation on the training part uses stratified $K$-fold splits. When hyperparameters are tuned, nested cross-validation (5 outer folds, 3 inner folds) separates selection from evaluation; reporting the inner-loop score instead would be optimistically biased (Varma and Simon, 2006; Cawley and Talbot, 2010). The package records the mean gap between inner and outer AUC as an estimate of that optimism.

**Out-of-time design.** A random split measures sampling error only. In production a model is applied to applicants who arrive after its development window, so it is also exposed to population drift. `evaluation.oot.oot_validation` fits the model on application months before a cut-off month $\tau$, keeps a stratified random holdout from that development window, and evaluates on all months $\ge \tau$. The gap between in-time and out-of-time results, read against the DeLong confidence intervals of each AUC and the PSI of the score, separates the two sources of error. Because the benchmark has no dates, `data.vintage.add_vintage` assigns synthetic months. With drift parameter $\kappa = 0$ the assignment is random; with $\kappa > 0$ rows are ordered by $\kappa z_i + \varepsilon_i$, where $z_i$ is the standardized rank of revolving utilization and $\varepsilon_i \sim N(0,1)$, so later months contain more highly utilized borrowers. The draws come from a random stream separate from the one that generated the attributes, so with $\kappa = 0$ the months are independent of every attribute.

**Synthetic data.** Tests and CI never use the real file. `data.synthetic.make_synthetic` draws a sample with the same schema: correlated delinquency counts from a shared gamma-distributed stress factor, lognormal income and debt ratio, and a logistic outcome whose intercept is solved numerically to hit a requested default rate. The coefficient signs follow the directions seen in the benchmark; their magnitudes are not estimates. Numbers computed on synthetic data are reported only as checks of the code and are labeled as such.

## 3. Features

**Engineered attributes.** With household size $h = \text{dependents} + 1$:

| Feature | Definition |
| --- | --- |
| `debt_per_dependent` | debt ratio $/\, h$ |
| `inc_per_dependent` | monthly income $/\, h$ |
| `credit_to_income` | open credit lines $/\,(\text{income} + 1)$ |
| `late_severity` | $s = n_{30\text{-}59} + 2\,n_{60\text{-}89} + 3\,n_{90+}$ |
| `log_*` | $\log(1+x)$ of income, utilization and debt ratio |

The three delinquency counts have pairwise correlations close to 0.99 in the benchmark. Entering them separately makes logistic coefficients unstable and their signs uninterpretable; the severity score keeps the ordering information in one variable with weights that increase with the depth of delinquency.

**Weight of Evidence.** For the scorecard, each attribute is cut at quantiles into at most ten bins. Bins holding less than 5% of the sample are merged into a neighbour, and adjacent bins are then merged until the default rate is monotonic in the attribute. With $g_b, b_b$ the numbers of non-defaulters and defaulters in bin $b$ and $G, D$ their totals,

$$\mathrm{WoE}_b = \ln\frac{g_b/G}{b_b/D}, \qquad \mathrm{IV} = \sum_b \left(\frac{g_b}{G} - \frac{b_b}{D}\right)\mathrm{WoE}_b .$$

A smoothing constant of 0.5 added to every count prevents infinite values in pure bins. Information Value ranks attributes; values below about 0.02 are conventionally treated as unpredictive (Siddiqi, 2006).

**Text embeddings.** Because the benchmark has no text, a synthetic loan-purpose statement is attached for development (Section 9). `features.text_embedding.TextEmbedder` maps text to $d$ dense features either by latent semantic analysis (TF-IDF on word unigrams and bigrams, then truncated SVD; Deerwester et al., 1990) or by Sentence-BERT embeddings (Reimers and Gurevych, 2019) reduced with PCA. Vocabulary, SVD and PCA are fitted inside the pipeline, so cross-validation folds never see test-fold text.

## 4. Models

**Penalized logistic regression.** The champion is

$$\operatorname{logit} \hat p(x) = \beta_0 + \beta^\top z(x),$$

where $z$ standardizes the engineered features with statistics from the training fold only. Coefficients minimize the class-weighted log loss plus an L1 or L2 penalty with inverse strength $C$ (Tibshirani, 1996). Class weights are balanced, which shifts the intercept; Section 5 explains why calibration is assessed and corrected afterwards.

**Scorecard.** A logistic regression on WoE-encoded attributes is converted to points. With base score $S_0$ at good:bad odds $O_0$ and $\mathrm{pdo}$ points to double the odds,

$$\text{score} = \text{offset} + \text{factor}\cdot\ln(\text{odds}_{\text{good}}), \qquad \text{factor} = \frac{\mathrm{pdo}}{\ln 2}, \qquad \text{offset} = S_0 - \text{factor}\cdot\ln O_0 .$$

The defaults are $S_0 = 600$, $O_0 = 50$ and $\mathrm{pdo} = 20$. Each bin receives integer points, and the total score maps back to a PD.

**Gradient boosting.** Histogram gradient boosting from scikit-learn, XGBoost (Chen and Guestrin, 2016) and LightGBM (Ke et al., 2017) fit additive tree ensembles to the log loss (Friedman, 2001). They capture interactions and non-monotone effects at the cost of a less transparent structure, which Section 7 addresses.

**Monotonic constraints.** For an input $x_j$ with expected direction $c_j \in \{-1, 0, +1\}$ the ensembles can be restricted so that $c_j\,\big(\hat p(x) - \hat p(x')\big) \ge 0$ whenever $x$ and $x'$ differ only in $x_j$ and $x_j > x_j'$. All three libraries enforce this during tree growth by rejecting splits whose child values violate the order and by bounding leaf values. The directions are set by credit judgment: PD is non-decreasing in utilization, debt ratio and delinquency severity, and non-increasing in age and income; the number of open lines, dependents and credit-to-income are left free because their effect is not one-signed. The constraints are verified after fitting by the share of ICE curves, on a quantile grid of each constrained input, that move against $c_j$. A constrained model cannot produce a reason code that contradicts the stated direction, which simplifies review and adverse action explanations at a usually small cost in AUC that the comparison report measures.

**Tuning.** Search spaces cover regularization strength and penalty type for the logistic model, and learning rate, tree size, minimum leaf size, subsampling and L2 regularization for the ensembles. Optuna's tree-structured Parzen estimator (Akiba et al., 2019) maximizes inner cross-validated AUC.

## 5. Evaluation

**Discrimination.** ROC-AUC is the probability that a randomly chosen defaulter receives a higher PD than a randomly chosen non-defaulter. The credit literature reports the Gini coefficient (accuracy ratio) $G = 2\,\mathrm{AUC} - 1$ and the Kolmogorov-Smirnov statistic $\mathrm{KS} = \max_t \lvert F_1(t) - F_0(t)\rvert$, where $F_1$ and $F_0$ are the score distributions of defaulters and non-defaulters (Thomas et al., 2002). PR-AUC and lift and gain tables by score decile complete the picture for the default class.

**Paired comparison.** Two models scored on the same applicants produce correlated AUC estimates. Their difference is tested with the nonparametric method of DeLong et al. (1988), computed with the midrank algorithm of Sun and Xu (2014), which gives the variance of $\widehat{\mathrm{AUC}}_A - \widehat{\mathrm{AUC}}_B$ from the placement values of each observation under both models. When the comparison is made across cross-validation folds instead of on one test set, fold differences $d_1, \dots, d_J$ are tested with the corrected resampled statistic

$$t = \frac{\bar d}{\sqrt{\left(\frac{1}{J} + \frac{n_{\text{test}}}{n_{\text{train}}}\right)\hat\sigma_d^2}}$$

of Nadeau and Bengio (2003), because overlapping training sets make the naive paired $t$-test anti-conservative.

**Sampling uncertainty.** Every reported metric is given with a bootstrap interval. Test-set rows are resampled with replacement within each class, so the number of defaulters is fixed as in the stratified split, and all metrics and all models are evaluated on the same $B$ replicates. For a statistic $\hat\theta$ with replicates $\hat\theta^*_1, \dots, \hat\theta^*_B$ the percentile interval is $[q_{\alpha/2}, q_{1-\alpha/2}]$ of the replicates and the basic interval is $[2\hat\theta - q_{1-\alpha/2},\, 2\hat\theta - q_{\alpha/2}]$ (Efron and Tibshirani, 1993). Approval rate, bad rate among approved applicants and profit per applicant are computed at the cut-off chosen on the full test set and held fixed, so their intervals do not include the variability of choosing the cut-off. For AUC the bootstrap interval agrees closely with the DeLong interval, which the tests check. Intervals of two models on the same applicants overlap more than a paired test would suggest; differences are tested with the paired methods above, not by comparing intervals.

**Calibration.** The Brier score $\frac{1}{n}\sum_i(\hat p_i - y_i)^2$ (Brier, 1950) combines calibration and refinement. Calibration alone is measured on $M$ quantile bins of $\hat p$ by the expected calibration error

$$\mathrm{ECE} = \sum_{m=1}^{M} \frac{n_m}{n}\,\lvert \bar p_m - \bar y_m \rvert$$

and by the Hosmer-Lemeshow statistic $\sum_m (O_m - E_m)^2 / [E_m(1 - E_m/n_m)]$, referred to a $\chi^2_{M-2}$ distribution (Hosmer and Lemeshow, 1980). Miscalibrated models are corrected on held-out data with Platt scaling (Platt, 1999) or isotonic regression (Zadrozny and Elkan, 2002); Niculescu-Mizil and Caruana (2005) show that boosted trees in particular benefit from this step.

**Prior shift.** The benchmark default rate is far higher than in a typical consumer portfolio, and class weighting changes the effective prior further. If the target population has default rate $\pi'$ and the model was calibrated at rate $\pi$, the PD should be adjusted before use by

$$\operatorname{logit} \hat p'(x) = \operatorname{logit} \hat p(x) + \ln\frac{\pi'}{1-\pi'} - \ln\frac{\pi}{1-\pi},$$

which is exact when only the class priors change (Saerens et al., 2002). Rankings, and therefore AUC, Gini and KS, are unaffected. A model fitted with class weights $w_0, w_1$ on a sample with default rate $\pi$ is calibrated to the effective rate $\pi_w = w_1\pi / (w_1\pi + w_0(1-\pi))$, which equals $0.5$ for balanced weights; $\pi_w$, not $\pi$, is the source rate in the adjustment. When the target rate is unknown, it is estimated from unlabeled target scores by the EM procedure of the same paper: starting from $\pi^{(0)} = \pi_w$, PDs are adjusted to $\pi^{(t)}$ and $\pi^{(t+1)}$ is set to their mean, until the rate stops changing. The fixed point is the maximum likelihood estimate of the target prior under the prior-shift assumption. After adjustment, calibration in the large is checked by $z = (O - E)/\sqrt{\sum_i \hat p_i(1-\hat p_i)}$, the standardized difference between observed and expected defaults. The test assumes independent defaults, so in a portfolio with correlated defaults it rejects too often.

## 6. Decision layer

Let a performing loan earn net margin $m$ and a defaulted loan lose $\mathrm{LGD}$, both as fractions of exposure at default $\mathrm{EAD}$. The expected value of approving applicant $x$ is

$$\mathbb{E}[\text{gain} \mid x] = \mathrm{EAD}\,\big[(1 - \mathrm{PD}(x))\,m - \mathrm{PD}(x)\,\mathrm{LGD}\big],$$

which is positive exactly when $\mathrm{PD}(x) < \mathrm{PD}^* = m / (m + \mathrm{LGD})$. With calibrated PDs, $\mathrm{PD}^*$ is the optimal cut-off. Because calibration is never perfect, the package also evaluates the empirical profit curve $\Pi(t) = \frac{1}{n}\sum_i \mathbb{1}[\hat p_i < t]\, \mathrm{gain}_i$ over a grid of cut-offs and reports the approval rate and the bad rate among approved applicants at its maximum. The default values $m = 0.08$, $\mathrm{LGD} = 0.45$ and $\mathrm{EAD} = 1$ in `configs/default.yaml` are illustrative assumptions, not estimates, and the chosen cut-off is only as good as they are.

**Uncertainty-aware decisions.** A point PD says nothing about how well the data support it. Two distribution-free methods are fitted on a calibration sample that is not used for training, and both are valid when calibration and test applicants are exchangeable.

*Mondrian conformal prediction* (Vovk et al., 2005) uses the nonconformity scores $a_0 = \hat p$ for label 0 and $a_1 = 1 - \hat p$ for label 1. With $n_c$ calibration applicants of class $c$, the p-value of label $c$ for a new applicant is

$$\pi_c = \frac{\#\{i : y_i = c,\ a_{c,i} \ge a_c\} + 1}{n_c + 1},$$

and the prediction set at level $\alpha$ is $\Gamma^\alpha = \{c : \pi_c > \alpha\}$. Because calibration is done within each class, $\Pr(y \in \Gamma^\alpha \mid y = c) \ge 1 - \alpha$ for both classes, so the guarantee is not carried by the majority class. The set gives a three-way rule: approve when $\Gamma^\alpha = \{0\}$, decline when $\Gamma^\alpha = \{1\}$, and refer to an underwriter when the set contains both outcomes or is empty. The referral rate at a given $\alpha$ is a direct measure of how much of the population the model cannot separate.

*Inductive Venn-Abers prediction* (Vovk and Petej, 2014) adds the test score $s$ to the calibration scores twice, once with label 0 and once with label 1, fits an isotonic regression to each extended sample and evaluates both at $s$. The result is an interval $[p_0, p_1]$ that contains a calibrated probability in the sense of Venn predictors. The single estimate $p = p_1 / (1 - p_0 + p_1)$ minimizes the worst-case log loss over the interval (Vovk et al., 2015). The interval is narrow where many calibration applicants have similar scores and wide in sparse regions, typically the extreme scores, which is where a fixed PD is least reliable. Isotonic regression depends only on the order of the scores, so each fit is cached on the position of $s$ among the calibration scores.

## 7. Explainability

**Attributions.** For the logistic model and the scorecard, exact Shapley values of the log-odds have the closed form $\phi_j(x) = \beta_j\,(z_j(x) - \mathbb{E}_{\text{bg}}[z_j])$, with the expectation taken over a background sample from the training data. For tree ensembles, TreeSHAP computes Shapley values in polynomial time (Lundberg and Lee, 2017; Lundberg et al., 2020). In both cases $\sum_j \phi_j(x)$ equals the model's log-odds minus the base value, and the tests check this additivity.

**Reason codes.** Attributions of engineered features are summed back to the borrower attribute they come from (for example `log_rev_util` and `rev_util` to utilization), so that a notice never cites a transformed variable. For a declined applicant, the attribute groups with the largest positive contributions to the PD, up to four, become the principal reasons required in adverse action notices under Regulation B.

**Effect shapes.** Partial dependence and individual conditional expectation curves (Goldstein et al., 2015) show how the predicted PD changes along a grid of one attribute while the others are held at observed values. The spread of centred ICE curves indicates interactions. They are also used to check that the PD does not increase with age above 62, which Regulation B would treat as assigning a negative value to an elderly applicant.

## 8. Fairness and monitoring

**Fairness.** Age is the only protected attribute in the data. Applicants are grouped into the bands under 25, 25-34, 35-44, 45-54, 55-61 and 62 and over. For each band the report gives the approval rate, the disparate impact ratio against the band with the highest approval rate (the four-fifths rule flags ratios below 0.8), and the approval rate among applicants who did not default. The difference of the last quantity between bands is the equal opportunity gap of Hardt et al. (2016) in a lending setting. These are descriptive statistics on one sample; they do not establish or exclude disparate treatment, and proxy effects for other protected attributes cannot be measured with this data.

**Stability.** With reference bin shares $e_k$ (quantile bins from the development sample) and current shares $a_k$, the population stability index is

$$\mathrm{PSI} = \sum_{k} (a_k - e_k)\ln\frac{a_k}{e_k}.$$

It is computed for each input (characteristic stability) and for the PD. Values below 0.1 are read as stable, 0.1 to 0.25 as moderate shift and above 0.25 as significant (Siddiqi, 2006).

**Champion-challenger.** A candidate model replaces the champion only after a paired evaluation on the same labeled out-of-sample applicants, with each model at its own cut-off. Three conditions must hold: the DeLong lower confidence bound of $\mathrm{AUC}_{\text{challenger}} - \mathrm{AUC}_{\text{champion}}$ exceeds $-\delta_{\mathrm{AUC}}$, the paired bootstrap upper bound of the Brier score difference stays below $\delta_B$, and the bootstrap lower bound of the profit difference per applicant is positive. The first two are non-inferiority tests; the third requires a demonstrated business gain. The swap-set table reports the bad rate of applicants the challenger would newly approve and newly decline.

## 9. Text features and the ablation design

The synthetic purpose statement for applicant $i$ is generated from one of eleven categories $k$ with probability

$$\pi_{ik} \propto p_k \exp\{\gamma\, s_k\,(2y_i - 1)\},$$

where $p_k$ is a prior share, $s_k$ a risk shift (negative for home improvement or car purchase, positive for overdue bills or emergencies) and $\gamma \ge 0$ the signal strength. A short qualifier about income or payment history is added with a probability that also depends on $y_i$ through $\gamma$. Because the text is conditioned on the outcome and not on the structured attributes, it carries information that the structured features do not, and the amount of that information is controlled by $\gamma$.

The ablation fits structured-only, text-only and combined pipelines on identical repeated stratified $K$-fold splits and tests the fold AUC differences against the structured baseline with the corrected resampled $t$-test. The case $\gamma = 0$ is a negative control: a well-specified pipeline must show no significant gain. The design demonstrates the method and guards against leakage; it does not estimate the value of text on real applications, which would require real application text and a time-ordered validation sample.

## 10. Assumptions and limitations

- The two-year outcome window is taken as given by the dataset; censoring and early closure are not observed.
- Without application dates the out-of-time design can only be exercised on synthetic vintages. Performance on real later vintages is likely to be lower, and production use requires out-of-time validation on dated data.
- The benchmark default rate is not representative, so PDs need the prior-shift adjustment above and recalibration on the target population.
- Only approved and booked accounts have outcomes. Rejected applicants are missing from the benchmark, so the model is fitted on a selected population. Reject inference (Section 11) is studied on synthetic applicants only.
- LGD, EAD and margin are constants set by assumption. In practice they vary by applicant and product.
- The text field and the credit policy used by the LLM layer are written for this project and are not real data.

## 11. Reject inference

Let $A \in \{0, 1\}$ indicate approval by the historical policy, so that $y$ is observed only when $A = 1$. A model fitted on approved applicants (known good/bad, KGB) estimates $\Pr(y = 1 \mid x, A = 1)$, which equals the target $\Pr(y = 1 \mid x)$ when selection is missing at random given the model inputs, $A \perp y \mid x$. Five methods are implemented:

- *KGB*: logistic regression on approved applicants.
- *Reweighting*: approved applicants weighted by $1/\hat\Pr(A = 1 \mid x)$ from a logistic approval model, with weights capped to limit variance. It targets the through-the-door population and is consistent under missing at random.
- *Fuzzy augmentation*: each reject enters twice, as bad with weight $\hat p_{\text{KGB}}(x)$ and as good with weight $1 - \hat p_{\text{KGB}}(x)$.
- *Hard cut-off augmentation*: rejects with $\hat p_{\text{KGB}}$ above a cut-off are labeled bad and the rest good.
- *Parceling*: rejects are grouped into quantile bands of $\hat p_{\text{KGB}}$ and labeled bad at random with probability $\min(1, k\,\bar p_b)$, where $\bar p_b$ is the band mean and $k \ge 1$ an inflation factor (Siddiqi, 2006).

The first four use only $x$ and the observed outcomes, so if the old policy used information $h$ that is absent from the data and related to $y$ (missing not at random), none of them can remove the bias; parceling removes it only to the extent that $k$ is chosen correctly, and $k$ cannot be estimated without outcomes for rejects (Hand and Henley, 1993; Crook and Banasik, 2004). The synthetic through-the-door generator makes the mechanism explicit. The old policy ranks applicants by a linear score of delinquency, utilization and income plus $0.7\,h$, with $h = \eta(2y - 1) + \varepsilon$, and approves a fixed share; $\eta = 0$ gives missing at random and $\eta > 0$ missing not at random. Since the outcomes of rejects are kept for evaluation, each method is scored against the truth on a holdout of all applicants. The study supports the usual practical conclusion: reject inference is a sensitivity analysis, and the reliable remedies are outcome data on rejects, from a bureau or from a small randomized approval test below the cut-off.

## References

- Akiba, T., Sano, S., Yanase, T., Ohta, T., and Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2623-2631.
- Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1-3.
- Cawley, G. C., and Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research*, 11, 2079-2107.
- Chen, T., and Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.
- Crook, J., and Banasik, J. (2004). Does reject inference really improve the performance of application scoring models? *Journal of Banking and Finance*, 28(4), 857-874.
- Deerwester, S., Dumais, S. T., Furnas, G. W., Landauer, T. K., and Harshman, R. (1990). Indexing by latent semantic analysis. *Journal of the American Society for Information Science*, 41(6), 391-407.
- DeLong, E. R., DeLong, D. M., and Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated receiver operating characteristic curves: A nonparametric approach. *Biometrics*, 44(3), 837-845.
- Efron, B., and Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman and Hall.
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189-1232.
- Goldstein, A., Kapelner, A., Bleich, J., and Pitkin, E. (2015). Peeking inside the black box: Visualizing statistical learning with plots of individual conditional expectation. *Journal of Computational and Graphical Statistics*, 24(1), 44-65.
- Hand, D. J., and Henley, W. E. (1993). Can reject inference ever work? *IMA Journal of Mathematics Applied in Business and Industry*, 5(1), 45-55.
- Hardt, M., Price, E., and Srebro, N. (2016). Equality of opportunity in supervised learning. *Advances in Neural Information Processing Systems*, 29.
- Hosmer, D. W., and Lemeshow, S. (1980). Goodness of fit tests for the multiple logistic regression model. *Communications in Statistics: Theory and Methods*, 9(10), 1043-1069.
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., and Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30.
- Lundberg, S. M., and Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.
- Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N., and Lee, S.-I. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2(1), 56-67.
- Nadeau, C., and Bengio, Y. (2003). Inference for the generalization error. *Machine Learning*, 52(3), 239-281.
- Niculescu-Mizil, A., and Caruana, R. (2005). Predicting good probabilities with supervised learning. *Proceedings of the 22nd International Conference on Machine Learning*, 625-632.
- Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In *Advances in Large Margin Classifiers*, MIT Press, 61-74.
- Reimers, N., and Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *Proceedings of EMNLP-IJCNLP*, 3982-3992.
- Saerens, M., Latinne, P., and Decaestecker, C. (2002). Adjusting the outputs of a classifier to new a priori probabilities: A simple procedure. *Neural Computation*, 14(1), 21-41.
- Siddiqi, N. (2006). *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring*. Wiley.
- Sun, X., and Xu, W. (2014). Fast implementation of DeLong's algorithm for comparing the areas under correlated receiver operating characteristic curves. *IEEE Signal Processing Letters*, 21(11), 1389-1393.
- Thomas, L. C., Edelman, D. B., and Crook, J. N. (2002). *Credit Scoring and Its Applications*. SIAM.
- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267-288.
- Varma, S., and Simon, R. (2006). Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics*, 7, 91.
- Vovk, V., Gammerman, A., and Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
- Vovk, V., and Petej, I. (2014). Venn-Abers predictors. *Proceedings of the 30th Conference on Uncertainty in Artificial Intelligence*.
- Vovk, V., Petej, I., and Fedorova, V. (2015). Large-scale probabilistic predictors with and without guarantees of validity. *Advances in Neural Information Processing Systems*, 28.
- Zadrozny, B., and Elkan, C. (2002). Transforming classifier scores into accurate multiclass probability estimates. *Proceedings of the 8th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 694-699.
