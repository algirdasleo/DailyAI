# DailyAI - yesterday's AI news in under 4 minutes

DailyAI ingests AI/tech news from the open web and produces a daily
executive brief - one document, scannable in 2-4 minutes, that leaves the
reader knowing what mattered yesterday in AI. Built for a mixed audience and is simple to read.

## How to run

```bash
cp .env.example .env        # add your keys
uv sync                     # install deps
uv run dailyai              # run pipeline and save the brief
uv run pytest               # run the pipeline tests
```

Or with Docker:

```bash
docker compose -f docker/docker-compose.yml up
```

## How it works

![alt text](pipeline_graph.png)

**Steps:**

1. **Collect articles from the last 24h.**

   > Pulls from RSS/Atom feeds, APIs (Hacker News, Hugging Face papers), and
   > a scraper. All feeds are normalized into a shared `Article` format.

2. **Remove duplicate articles.**

   > Drops the same article when it's posted multiple times, by comparing URLs and fuzzy-matching titles.

3. **Embed & store each article.**

   > Captures each article's meaning as a vector embedding
   > and saves it in the database, so it's ready for
   > clustering and reusable later.

4. **Cluster articles into stories.**

   > Groups articles reporting on the same story by looking at how close
   > their embeddings are. The more sources land in one cluster, the more
   > popular the story is. This is a signal that later pushes it rank higher.

5. **Drop stories already published before.**

   > Compares each story's embedding against the stories already covered in
   > recent briefs, and skips it if it's too similar. This way the brief
   > doesn't cover the same ground twice.

6. **Rank stories & keep the top ones.**

   > An LLM orders the stories by importance, using source count,
   > authority, and recency as hints. As a fallback, if that call fails, a
   > rule-based sort on those same metrics is used instead.

7. **Fetch each story's full article text.**

   > Retrieves the full article for each story that was selected.
   > If none available, falls back to the description provided by the source.

8. **Summarize each story.**

   > Each story's article is summarized by LLMs in parallel, turning the
   > full text into the short, factual material the brief will rely on.

9. **Draft the brief.**

   > One LLM call turns the summaries into the finished brief, adding a
   > "why it matters" note and a citation for each story.

10. **Evaluate the brief's quality.**

    > Checks every citation against the real source list, then runs three
    > evaluations: claim support (is each claim traceable to a source),
    > cluster coherence (did clustering merge unrelated stories), and a
    > quality review (scannability, takeaways, jargon).

11. **Save the brief & its metrics.**

    > Writes the finished brief to a markdown file, and stores its
    > evaluation results, quality metrics in the database.

## Example output

See [`briefs/`](briefs/) for a brief generated from a real run.

## Limitations

> Current implementation limitations - what breaks, why, and how it could be fixed.

### Evaluation finds problems but no-one fixes them before launch

- **Problem:** Evaluation step brings up possible brief issues, but the brief is shipped regardless and the issues are ignored.
- **Improvement:**
  1. Add a revision step that passes the flagged issues (unsupported claims, fabricated URLs) back to the drafter to fix them, then evaluate again.
  2. Prevent publishing based on a quality threshold. This way, a poor quality brief is not shipped.

### Hand-maintained AI relevance keyword list can not keep up

- **Problem:** Relevance is decided by a hand-edited keyword regex. The list becomes outdated as new labs and models appear.
- **Improvement:**
  1. Replace the keyword list with a few-shot LLM classifier that understands context instead of matching keywords.
  2. Store each article's text with the LLM's relevant/not-relevant verdict to build the labeled dataset for step 3.
  3. After collecting enough examples, train a logistic regression model to replace the LLM for relevance filtering.

### Duplicated story and updated story are treated as the same

- **Problem:** The step that skips recently covered stories can not tell a repeat from a developing story. Both look similar by embedding, but the developing story carries new information.
- **Improvement:**
   1. On a "recently covered" match, link the story to the earlier one instead of dropping it. Then compare today's facts against the ones already published for that story.
   2. If there are new facts, keep it as an update. If no new facts, drop it as a story repeat.

### Fuzzy title deduplication weakens the popularity signal

- **Problem:** Dedup drops articles with identical headlines. But when several outlets use the identical article headlines, that is coverage of one story and not a duplicate.
- **Improvement:** Only treat an identical headline as a duplicate when it comes from the same outlet.

### 24h window is too short for capturing trending HuggingFace papers

- **Problem:** Papers rarely trend the day they launch. Upvotes and discussion accumulate over days.
- **Improvement:** A rolling 7-day or monthly window ranked by trending momentum would catch the papers that genuinely are important.

## Production improvements

> Solution improvements for a production-ready version.

### Scalability and performance

#### SQLite was for the MVP and does not scale

- **Currently:** SQLite locks the DB file for writes and does not survive a multi-replica deploy, since each Pod would have its own separate database file.
- **Improvement:** Use Postgres. The app then is not limited by concurrent write blocks, can run with multiple replicas and use one shared database.

#### The whole run is one big sequential step

- **Currently:** Pipeline.run() does everything in one long process without keeping track of which steps were completed. If it crashes along the way, the whole process restarts from scratch.
- **Improvement:** Break the steps into activities and form a workflow using a workflow engine like Temporal. This way, the progress is tracked, stored and if something fails, workflow can continue from that step.


---
### Accuracy

#### Evaluation finds problems but no-one fixes them before launch (point from Limitations)

- **Currently:** A broken brief ships like a clean one. The evaluation scores are saved but not used.
- **Improvement:** After evaluation add a new step that fixes issues found in evaluation step.

#### No output quality regression testing

- **Currently:** Hard to tell whether a new feature or an adjustment improved or worsened the output quality.
- **Improvement:** Add a human-labeled golden dataset and score against it on every change.

---
### Data freshness and update mechanisms

#### Feeds are re-fetched even when nothing changed

- **Currently:** Every run re-fetches every feed, even if nothing changed.
- **Improvement:** Use HTTP caching headers where available. Otherwise fetch and diff items against what is already stored.

#### A dead source fails silently

- **Currently:** A dead source is logged once, then silently returns nothing.
- **Improvement:** Track each source's last successful fetch and alert when one dies.

---
### Monitoring and observability

#### Evaluation results are not easy to see

- **Currently:** Evaluation metrics and issues are stored per brief but never used. Checking them means querying the DB directly.
- **Improvement:** Build a dashboard for showing the trend of the stored metrics. This dashboard could be expanded and used for the other observability use-cases aswell.

#### Some pipeline decisions leave no explanation

- **Currently:** The pipeline does not record why it filtered, deduped, clustered, or ranked something a certain way.
- **Improvement:** Store the decisions reasoning, the same way evaluation already are. This way, it is easy to debug and improve the solution.

---
### Cost optimization

#### Every LLM task uses same model

- **Currently:** Different LLM tasks use the same model.
- **Improvement:** Adjust models per task. For example, smaller model for summarization, stronger model for drafting, evaluation.
