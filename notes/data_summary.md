# MIND-small: Data Summary

Source: MIND-small (Wu et al., 2020), downloaded from the Recommenders project's
Hugging Face copy (Recommenders/MIND). The original Azure link returns
"409: Public access is not permitted". Microsoft Research License terms apply.

## Time window
- Train file covers Nov 9 to Nov 14, 2019 (week 5 of the 6-week log, minus its last day)
- User history comes from weeks 1 to 4 and is fixed per user

## Users and impressions
- 50,000 users, 156,965 impressions
- 5,843,444 (user, article) pairs after splitting impressions
- Click rate: 4.04%
- Every impression has at least 1 click (median 1, mean 1.5, max 35),
  so the click rate is conditional on a click occurring
- Articles per impression: median 24, mean 37.2, range 2 to 299
- Impression IDs are not in time order; sort by `time` when order matters

## User history
- Median 11 articles, mean 18.5, max 558
- 892 users (1.8%) have empty history; plan: exclude them from the simulator

## Articles
- 51,282 articles, 17 categories
- news 31%, sports 28%; kids, middleeast, northamerica are tiny (merge into "other")
- 2,666 (5.2%) missing abstracts; use title only for these
- Titles: median 10 words. Abstracts: median 24 words
- 20,288 articles shown in impressions; all have text
- 30,994 articles appear only in user histories; 0 appear in neither

## Cold start
- Only 2,201 of the 20,288 shown articles appear in any MIND-small user's history
- 89% of articles shown during the training week have no prior clicks
  from these users, so a policy must judge most articles from content alone
