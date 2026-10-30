# Sources and attribution

## FiQA

- [Original FiQA challenge](https://sites.google.com/view/fiqa/home)
- [BEIR FiQA corpus and queries](https://huggingface.co/datasets/BeIR/fiqa), revision `979c07a7cb5ccc6ca009792241fa1250b98055dd`.
- [BEIR FiQA relevance judgments](https://huggingface.co/datasets/BeIR/fiqa-qrels), revision `252958f2d646e22cab6d0c72dd3f0d5de6d0655a`.

The BEIR cards list CC BY-SA 4.0. The original challenge states noncommercial use; retain this discrepancy and clarify applicable terms before commercial use. Included files preserve source texts and select queries from the published splits; they add no new relevance judgments. This benchmark contains financial community answers, rather than a corpus of corporate 10-K filings.

## LongBench v2

- [Project](https://longbench2.github.io/)
- [Official dataset](https://huggingface.co/datasets/zai-org/LongBench-v2), revision `2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9`.
- Bai et al., *LongBench v2 Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks*, [paper](https://arxiv.org/abs/2412.15204).

The dataset card lists Apache 2.0. The included pilot preserves original contexts, questions, options and answer letters and adds a project split. No new supporting-span labels are claimed. Raw full-dataset downloads are omitted from the upload.

## Models and counting tokenizer

- [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`.
- [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- [MiniLM counting tokenizer](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.

These model cards list Apache 2.0. Model weights are downloaded locally and are not redistributed. The counting tokenizer and Apache license text are included in `assets/`. Qwen embedding pooling follows the published last-token, left-padding approach. A MiniLM embedding model is not run in this folder.
