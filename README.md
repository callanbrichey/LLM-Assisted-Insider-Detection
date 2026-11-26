# LLM-Assisted Insider Detection in a Simulated Market

This project was undertaken in completion of **COMP0257 Advanced Topics in Financial Modeling and Technologies** in the **MSc Computational Finance** programme at **University College London (UCL)**.

This study uses a Kyle/Back-inspired market simulation and a Gemini workflow to detect days when an informed trader is active.

## Analysis

- **Market simulation.** Simulates 6,000 trading days at 4,680 steps per day, with labelled informed-trader activity, prices and order flow.
- **Agent-generated code.** An ETL agent writes a Python function that summarises intraday observations. A modelling agent generates five candidate programs per iteration. Python tests the programs and selects the valid candidate whose predictions are closest to the group consensus. Earlier code and evaluation feedback inform later iterations.
- **Classifier evaluation.** Uses the first 4,500 days for development and reserves the last 1,500 days for a single local evaluation. The recorded results include an always-inactive baseline and an absolute daily log-return rule whose threshold maximises F1 on the development data.

### Methodology

**Market simulation.** We draw on [Kyle (1985)][kyle-1985] for the sequential auction structure and linear price response to order flow, and on [Back (1992)][back-1992] for the continuous-time treatment of informed trading. The unobserved fundamental value follows a geometric Brownian motion (GBM) across trading days and is held fixed within each day. To model intermittent informed trading, the simulator draws $K_d \sim \mathrm{Bernoulli}(\alpha)$ once per day, with $\alpha=0.4$ by default. It records $K_d$ as `Signal`, the label the classifier is trying to recover.

Let $v_d$ be the log fundamental value, $p_{d,i}$ the log market price before auction $i$ and $\Delta t = \frac{1}{252N}$ the intraday step in years for $N$ auctions per day. The intraday update has the Euler–Maruyama form described by [Higham (2001)][higham-2001]. The code computes signed net order flow $q_{d,i}$ and updates the log price as follows.

```math
\begin{aligned}
q_{d,i} &= K_d\,\beta_i(v_d-p_{d,i})\,\Delta t + u_{d,i},
& u_{d,i} &\sim \mathcal{N}(0,~\sigma_U^2\Delta t),\\
p_{d,i+1} &= p_{d,i} + \sqrt{\alpha}\,\lambda q_{d,i},
& \lambda &= \frac{\sigma_P}{\sigma_U}.
\end{aligned}
```

Here $\sigma_P$ and $\sigma_U$ are the annualised volatilities of fundamental log returns and noise order flow. The variance proxy $\Sigma_i=\max\{\frac{(1-\frac{i}{N})\sigma_P^2}{252},\,10^{-12}\}$ declines towards the close, increasing the informed trading coefficient $\beta_i=\frac{\sigma_U^2\lambda}{\sqrt{\alpha}\,\Sigma_i}$. On inactive days, only noise trades move the price. Each day's final auction price becomes the next day's opening price. [Simulation code](Code/simulation.py)

**ETL agent.** We split by trading day before ETL. Gemini receives up to 250 intraday rows and summary metadata, both drawn from the development period. The workflow draws on Program of Thoughts ([Chen et al., 2023][chen-2023]) by asking Gemini to explain its choices and express the calculations as a Python function. A local worker applies the function to development rows to compute daily log return, price change and the mean, variance, skewness and kurtosis of intraday log returns and net order flow. The workflow keeps `Signal` separate and attaches the simulator's labels to the daily features. If the function fails on development data or its output fails validation, the workflow passes the error back to Gemini for revision, adapting the feedback loop in Reflexion ([Shinn et al., 2023][shinn-2023]). [ETL code](Code/etl_agent.py) · [Local worker](Code/generated_code.py)

**Modelling agent.** Gemini receives summary metadata and up to 100 sampled rows from each development training and validation set, with validation labels masked. A Step-Back prompt ([Zheng et al., 2024][zheng-2024]) asks it to identify plausible market microstructure models and use those hypotheses to motivate feature choices. The prompt draws on chain-of-thought prompting ([Wei et al., 2022][wei-2022]) for the written rationale and Program of Thoughts ([Chen et al., 2023][chen-2023]) for the classifier code. In the first four rounds, the local worker fits each candidate on the earliest 80% of an expanding development window and validates on the remaining 20%. The fifth round reuses the last development split. [Development splits](Code/prepare_modelling_data.py) · [Modelling code](Code/modelling_agent.py)

**Candidate selection and feedback.** We adapt self-consistency ([Wang et al., 2023][wang-2023]) to select among up to five candidate programs per round. The workflow requests five candidates concurrently, checks the generated code locally and retries only failed slots, for up to five waves. It selects a program once all five candidates pass or the attempts are exhausted with at least three valid candidates. For each validation day, it averages the binary predictions of the valid candidates and selects the program whose predictions have the smallest sum of squared differences from those averages.

When generated code fails validation, the workflow feeds the errors into the next attempt for those slots. Between development rounds, it carries the selected response, including its code, and its performance scores into the next prompt. The agent is asked to reflect on those results and revise its approach, drawing on Reflexion ([Shinn et al., 2023][shinn-2023]) and the use of accumulated experience to guide planning in [Park et al. (2023)][park-2023].

**Final evaluation.** After the fifth round, the local worker applies the selected ETL function to the reserved days, using the last development day's close to compute the first test day's return. The workflow refits the selected classifier on all development days and evaluates it once on the reserved test period. Test labels are used to calculate metrics after prediction. Gemini receives development data throughout the workflow. [Main workflow](Code/main.py)

## Data and Results

The input data are synthetic. The simulation produces 28.08 million intraday observations. The repository includes a 250-row sample and summary metadata for the full simulation alongside the recorded daily outputs. A full run regenerates the 2.04 GB intraday CSV at `Outputs/data/synthetic_intraday_market_data.csv`. The public download is pending and will be listed under [GitHub Releases](https://github.com/callanbrichey/LLM-Assisted-Insider-Detection/releases).

### Recorded Run

The results below describe one Gemini 3.5 Flash-Lite run, chosen for reporting after comparing completed runs on simulated data. The final classifier combined histogram gradient boosting, a random forest and logistic regression by taking a weighted average of their predicted probabilities. It set its decision threshold using the observed share of active days and fitted probabilities in the development period, then constrained the threshold to the range 0.36–0.44.

| Method | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Always predict inactive | 60.1% | — | 0.0% | 0.000 |
| Absolute daily log-return rule | 65.1% | 55.0% | 68.3% | 0.610 |
| Gemini-generated classifier | 72.5% | 69.7% | 55.3% | 0.616 |

On the final 1,500 days, the classifier detected 331 of 599 active days and raised 144 false alarms. The return rule, using a threshold of approximately 0.00711502, detected 409 active days and raised 334 false alarms. The classifier achieved higher accuracy, precision and F1, while the return rule achieved higher recall.

It should be noted that the closing price observed in the reported run represents a lower-tail outcome under our configured simulator. To estimate the probability of a terminal price at or below what was observed, we begin by letting $V_t$ be the hidden fundamental value of the traded asset. Applying It&ocirc;'s lemma to our GBM and then integrating gives

```math
\begin{aligned}
\mathrm{d}\log V_t
&= \left(\mu - \frac{1}{2}\sigma^2\right)\mathrm{d}t
   + \sigma\,\mathrm{d}W_t, \\
\log\!\left(\frac{V_T}{V_0}\right)
&\sim \mathcal{N}\!\left(
    \left(\mu - \frac{1}{2}\sigma^2\right)T,\,
    \sigma^2 T
\right).
\end{aligned}
```

Let $t_*$ be the closing time of the last day with an active insider. At that day's final auction, the traded log price becomes $\log V_{t_*}$ plus a noise shock with variance $\alpha \sigma^2 \Delta t$. Order-flow noise on the remaining inactive days adds variance $\alpha \sigma^2 (T - t_*)$. This gives the conditional distribution

```math
\begin{aligned}
\log\!\left(\frac{P_T}{P_0}\right) \left. \right| t_*
&\sim \mathcal{N}\!\left(
    \left(\mu - \frac{1}{2}\sigma^2\right)t_*,\,
    \sigma^2 t_* + \alpha \sigma^2 \left( \Delta t + T - t_* \right)
\right).
\end{aligned}
```

Plugging in the default simulation parameters $V_0 = \$ 100$, $\mu = 0.1$, $\sigma = 0.2$, $\alpha = 0.4$, $\Delta t = \frac{1}{252 \times 4680}$, $T = \frac{6000}{252}$, and the observed $t_* = \frac{5999}{252}$, we find a log return mean of about 1.904 and a standard deviation of about 0.976. The observed final closing price gives $\log \left( \frac{108.30}{100} \right) \approx 0.080$. Applying the standard normal cumulative distribution function (CDF) gives

```math
\Phi\!\left(\frac{\log(108.30/100)-1.9044}{0.9759}\right) \approx 0.0307.
```

In other words, given that day 5,999 was the last active insider day, the simulation closes at or below \$108.30 approximately 3.07% of the time. The [full-period figure](Figures/MarketSimulationFullPeriod.pdf) puts that close in context by showing the traded price, the estimated fundamental path and the pointwise 5th to 95th percentiles of the fundamental GBM.

The reported run illustrates how iterative feedback can inform an LLM's revisions to a classifier for detecting informed trading in noisy data. It also raises the question of how sensitive classifier performance is to the realised market path. A follow-up could run the workflow with and without validation feedback on the same set of fresh simulations, with equal candidate budgets. This would help distinguish the contribution of feedback from variation across paths, including runs with lower-tail closes.

The simulator's market structure also shapes what the modelling worker can learn. With fixed price impact, each intraday log-price change is proportional to net order flow, so several return and order-flow features encode the same movement. This helps explain why the absolute-return rule remains competitive. Future simulations could vary price impact with observable market depth, independently of insider activity. The same order flow would then move prices more in a thin market. Comparing the generated classifier with both the absolute-return rule and a liquidity-adjusted rule at common alert budgets would test whether the modelling worker can use that added information to detect more active days.

## Project Files

- [Main workflow](Code/main.py)
- [Recorded ETL response](Outputs/logs/ETL_log.txt) and [modelling transcript](Outputs/logs/modelling_log.txt) with raw Gemini responses
- [Simulation sample](Tables/df_sample.csv) and [simulation metadata](Tables/df_metadata.csv)
- [Daily summary](Outputs/data/daily_summary.csv), [evaluation metrics](Outputs/data/evaluation_metrics.csv) and [test predictions](Outputs/data/evaluation_predictions.csv)
- [Simulation figures](Figures/) and [final test confusion matrix](Figures/FinalTestConfusionMatrix.pdf)

## Run Locally

You can inspect the [recorded results](#recorded-run) without an API key. To regenerate the intraday CSV, simulation figures and tables without Gemini, run `python -B main.py --simulation-only` from `Code/` in a separate clone. This preserves the saved results in your main copy.

Install the dependencies in the Python environment you intend to use.

```powershell
python -m pip install numpy pandas matplotlib "seaborn>=0.12" scipy scikit-learn google-genai
```

Open PowerShell at the root of your clone and activate the Python environment where you installed the dependencies. Use the block below and enter your own Gemini API key at the masked prompt. The block clears the key from the terminal environment when it exits.

**Live run safety.** The use of a disposable virtual machine (VM) for `--gemini` with shared folders and clipboard sharing disabled is recommended. The live workflow sends synthetic samples and metadata to Gemini, then automatically runs the Python it returns. That code runs with the file and network permissions of the account that starts the workflow. The worker starts without the Gemini API key in its environment.

```powershell
Set-Location -LiteralPath '.\Code' -ErrorAction Stop

try {
    $geminiSecret = Read-Host 'Enter your Gemini API key' -AsSecureString
    $env:GEMINI_API_KEY = [System.Net.NetworkCredential]::new('', $geminiSecret).Password

    python -B main.py --gemini
} finally {
    Remove-Item Env:GEMINI_API_KEY -ErrorAction SilentlyContinue
    Remove-Variable geminiSecret -ErrorAction SilentlyContinue
}
```

- **Short run.** Add `--days 40 --nsteps 100` to the Python command. Add `--seed` to reproduce the simulated inputs on subsequent runs with the same settings. Gemini can generate different code on each live run.
- **Resume after ETL.** Add `--resume-modelling` to restart at modelling iteration 1 using the local ETL checkpoint and its simulation CSV.
- **Model and API usage.** The default is Gemini 3.5 Flash-Lite. ETL makes up to ten single-response attempts. The workflow caps its generation requests at 15 starts per minute across ETL and modelling. A full run uses 26 generation calls and 22 file uploads when all stages succeed on their first attempts. Retries can use more requests. [Project rate limits](https://aistudio.google.com/rate-limit)

Runs write to `Figures/`, `Tables/` and `Outputs/`, overwriting files as each stage finishes. Git ignores the full intraday CSV because of its size. Keep the recorded run together when updating the portfolio files, because an interrupted run can leave outputs from different runs.

## References

- Back, K. (1992). [*Insider Trading in Continuous Time*][back-1992]. The Review of Financial Studies.
- Chen, W., et al. (2023). [*Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks*][chen-2023]. Transactions on Machine Learning Research.
- Higham, D. J. (2001). [*An Algorithmic Introduction to Numerical Simulation of Stochastic Differential Equations*][higham-2001]. SIAM Review.
- Kyle, A. S. (1985). [*Continuous Auctions and Insider Trading*][kyle-1985]. Econometrica.
- Park, J. S., et al. (2023). [*Generative Agents: Interactive Simulacra of Human Behavior*][park-2023]. UIST.
- Shinn, N., et al. (2023). [*Reflexion: Language Agents with Verbal Reinforcement Learning*][shinn-2023]. NeurIPS.
- Wang, X., et al. (2023). [*Self-Consistency Improves Chain of Thought Reasoning in Language Models*][wang-2023]. ICLR.
- Wei, J., et al. (2022). [*Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*][wei-2022]. NeurIPS.
- Zheng, H. S., et al. (2024). [*Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models*][zheng-2024]. ICLR.

[back-1992]: https://www.jstor.org/stable/2962132
[chen-2023]: https://arxiv.org/pdf/2211.12588
[higham-2001]: https://epubs.siam.org/doi/10.1137/S0036144500378302
[kyle-1985]: https://www.jstor.org/stable/1913210
[park-2023]: https://arxiv.org/pdf/2304.03442
[shinn-2023]: https://arxiv.org/pdf/2303.11366
[wang-2023]: https://arxiv.org/pdf/2203.11171
[wei-2022]: https://arxiv.org/pdf/2201.11903
[zheng-2024]: https://arxiv.org/pdf/2310.06117
