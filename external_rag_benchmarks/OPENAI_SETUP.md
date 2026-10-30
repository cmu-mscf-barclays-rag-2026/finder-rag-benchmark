# School API access

The invitation must first be accepted with the school email account. In the OpenAI Platform, select Group 004 and the project assigned by the instructor. Create an authorized project key and configure it locally as OPENAI_API_KEY. Configure OPENAI_MODEL with a model allowed by that project. Never paste a key into chat or include it in GitHub. Restart the terminal/app after changing persistent environment variables.

This session currently has no API key, so no OpenAI benchmark request has run. The screenshot alone does not establish project model access, credit balance or budget.

From this folder, preview the request plan without contacting the API:

```bash
python run_openai_reader.py --model YOUR_APPROVED_MODEL --limit 1
```

After configuring the project key and model, make one test request:

```bash
python run_openai_reader.py --model YOUR_APPROVED_MODEL --limit 1 --execute
```

The full comparison has 80 conditions: twenty questions times BM25, dense, hybrid and no context. With the same approved settings and sufficient school allowance, finish remaining conditions with --limit 80 --execute. Completed requests resume without being sent again. Responses, model IDs and token usage are saved under results/openai. A comparison table is produced only after all conditions complete.

The experiment reuses the existing LongBench questions and saved retrieved contexts, changing only the answer model and its native API execution. This compares answer-model behavior without rebuilding retrieval. FiQA's document-relevance scores need no answer model. Model-specific reasoning defaults may differ; inspect the response usage when comparing costs.

Requests use the school's API allowance. Output-token limits do not establish a dollar spending cap. There are no automatic retries. An incomplete response stops the run for inspection. Keep the existing local-model results as the baseline.

Official setup reference: https://developers.openai.com/api/docs/quickstart
