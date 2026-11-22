# A/B test design: does the coach help newcomers get merged?

The model and the coach show what is *associated* with a first PR getting merged. They don't show that advice *causes* more merges. A newcomer who reads guidelines carefully might have been merged anyway. The only way to know whether the coach helps is a randomised experiment. This document is the plan for one. It hasn't been run.

## The question

Does posting MergeMate's advice on a newcomer's first pull request raise the chance that it is merged within 30 days?

## How newcomers meet the coach

Partner repositories install a GitHub bot. When someone opens their **first** pull request to that repository, the bot is triggered:

- **Treatment:** the bot posts the coach's advice as a comment (what to fix, with the project's own rules quoted).
- **Control:** the bot posts nothing.

I chose a bot over the web app because the web app only reaches people who go looking for help. They are already unusual, so the result would only apply to them. The bot reaches every newcomer in a partner repository.

## Who is randomised

Each **newcomer** is randomised, not each repository. With only about 45 eligible repositories, randomising whole repositories would leave far too few units to detect anything.

Randomisation is **stratified by repository**: within each repository, newcomers are split 50/50. The repository is the strongest predictor of a merge (the repo's recent merge rate is the model's top feature), so this keeps both groups balanced between easy and hard projects.

The known risk of randomising people rather than repositories is **spillover**. Maintainers who see bot comments on some PRs may change how they treat every newcomer, which would make the two groups more alike and hide part of the effect. I'd watch for it by comparing each repository's control-group merge rate during the test with its rate before the test.

## Eligibility

- The repository's maintainers opt in.
- Repositories that ban AI-generated content are excluded (2 of the 50: gleam and meilisearch).
- The PR is the author's first in that repository, using the same definition as the model.
- Bots and accounts flagged as bots are excluded, with the same rules as the data cleaning.

## Metrics

| Type | Metric | Why |
|---|---|---|
| **Primary** | Merged within 30 days of opening | Same target as the model, so the experiment and the model measure the same thing |
| Secondary | Time to first human reply | Clearer PRs may get answered faster |
| Secondary | Returns with new work within 90 days | Merged newcomers were twice as likely to come back (22% vs 11%); this tests whether the coach moves that too |
| Guardrail | PRs closed as spam or low quality | The coach must not make things worse |
| Guardrail | Maintainers opting out, negative reactions to the bot | The coach must not annoy the people it depends on |

A guardrail getting clearly worse is a reason to stop early, whatever the primary metric shows.

## Sample size

- **Baseline:** 37.5% of newcomer PRs merged within 30 days (2026).
- **Smallest effect worth detecting:** +6 percentage points (37.5% → 43.5%). Anything smaller probably isn't worth the maintainers' attention, and a larger threshold would risk missing a real, useful effect.
- **Settings:** two-sided test, α = 0.05, 80% power.

| | |
|---|---|
| Newcomers needed per group | 1,050 |
| **Total** | **2,100** |
| Newcomer PRs per month (50 repos, 2026 rate) | about 350, a little fewer after exclusions |
| **Time to enrol** | **about 6 months**, plus 30 days for the last outcomes |

## Analysis

1. **Sample-ratio check first.** The groups should split close to 50/50. If they don't, the assignment is broken, and no other result can be trusted.
2. **Primary result:** the difference in 30-day merge rates between the groups, with a 95% confidence interval.
3. **Variance reduction (CUPED).** MergeMate's own score is computed for every PR *before* any advice is shown, so it can't be affected by the treatment. Using it as a covariate removes part of the noise that comes from some PRs simply being easier than others. The expected saving is around 10% of the sample. The exact figure depends on how strongly the score correlates with the outcome, which I'd measure on 2026 data before launch.
4. **No peeking.** The sample size is fixed in advance and the result is analysed once, at the end. Checking repeatedly and stopping when the result looks good inflates false positives. That's the same discipline as using the 2026 test set only once.
5. **Within-repository analysis** as a robustness check, since randomisation was stratified by repository.

## Ethics and transparency

- Repositories take part only if their maintainers agree.
- Every bot comment is labelled as automated and links to an explanation of how it works.
- The coach only advises. It never writes code, PR descriptions or replies for the contributor, which keeps it within the AI policies of the projects that allow AI assistance.
- Newcomers in the control group lose nothing they would otherwise have had.

## What would change the plan

- **Fewer partner repositories than expected:** enrolment slows in proportion. I'd either run longer or accept a larger minimum effect, decided *before* launch, not during.
- **Strong spillover:** switch to randomising repositories, and accept that the test can then only detect large effects.