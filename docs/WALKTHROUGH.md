# Code walkthrough

This document goes through the project file by file and block by block. Every important block gets two explanations:

- **Plain:** the idea, with an everyday comparison.
- **Technical:** the exact mechanics, the data shapes, and what would break if the code were written differently.

The first two sections give you the whole system at once. After that, the files appear in the order the pipeline uses them.

---

## Contents

1. [The whole system in one pass](#1-the-whole-system-in-one-pass)
2. [The blocks worth knowing cold](#2-the-blocks-worth-knowing-cold)
3. [config.py](#3-configpy)
4. [data.py](#4-datapy)
5. [prompts.py](#5-promptspy)
6. [teacher.py](#6-teacherpy)
7. [llm.py](#7-llmpy)
8. [train.py](#8-trainpy)
9. [export.py](#9-exportpy)
10. [student.py](#10-studentpy)
11. [calibration.py](#11-calibrationpy)
12. [routing.py](#12-routingpy)
13. [evaluate.py](#13-evaluatepy)
14. [bench.py](#14-benchpy)
15. [serve.py](#15-servepy)
16. [Tests, config and build files](#16-tests-config-and-build-files)

---

## 1. The whole system in one pass

**Plain.** Think of a hospital. A senior doctor (the big LLM) is excellent but expensive and slow. A nurse (DistilBERT) is quick and cheap.

1. The senior doctor goes through a big stack of past cases and writes down a diagnosis for each one.
2. The nurse studies those diagnoses until they can mostly reproduce them.
3. We test how honest the nurse is when they say "I'm sure", and adjust for it.
4. We set a rule for when the nurse has to call the doctor.

From then on, the nurse sees every patient and only calls the doctor for the cases they aren't sure about.

**Technical.** There are two phases: offline (build) and online (serve).

### Offline

```
python -m cascade.teacher
  load_config → load_clinc → (per split) read cache → make_labeler (lazy)
  → messages() → apply_chat_template → vLLM generate with choice constraint
  → parse_label → append_jsonl → summarize → reports/teacher_constrained.json
  data: 23,850 strings → 23,850 rows {"idx", "raw", "label", "secs"} in data/teacher/*.jsonl

python -m cascade.train
  load_clinc → teacher_labels("train") → drop -1s → batches() with dynamic padding
  → DistilBERT forward [B, L] → logits [B, 151] → cross-entropy vs teacher id
  → AdamW + warmup/decay → per-epoch agreement on validation → save_pretrained
  data: 15,250 (text, teacher_id) pairs → artifacts/student/

python -m cascade.export
  to_onnx: wrap model as LogitsOnly → torch.onnx.export (dynamo, dynamic batch/seq)
  → strip value_info → quantize_dynamic (INT8 weights)
  → parity: torch argmax vs FP32 ONNX vs INT8 ONNX on 3,100 validation texts
  data: ~257 MB FP32 graph → ~65 MB INT8 graph

python -m cascade.evaluate
  Student(int8).logits(validation) → [3100, 151]
  Student(int8).logits(test)       → [5500, 151]
  fit_temperature(val logits, val teacher ids) → T
  softmax(val logits / T) → conf, pred → pick_threshold(budget=0.01) → τ
  softmax(test logits / T) → conf, pred → route(pred, conf, test teacher, τ) → final
  score teacher / student / cascade against gold; ECE before/after; plots
  data: → artifacts/routing.json {"temperature", "threshold", "onnx"}, reports/*
```

### Online

```
POST /classify {"text"}
  Student.encode([text]) → input_ids, attention_mask  [1, L] int64
  Student.forward → logits [1, 151]
  softmax(logits / T)[0] → probs [151] → top, conf
  conf ≥ τ  → return student label                       source="student"
  conf < τ  → ChatLabeler.label(text) via OpenAI-compatible endpoint
              → valid label   → return it                  source="llm"
              → error/invalid → return student label       source="student_fallback"
```

### Key facts

- **Label set:** 151 labels (150 intents + `oos`). The label index order comes from the dataset's `ClassLabel` feature, and it is saved as `labels.json` next to the student so serving uses the same order.
- **Teacher labels at every stage:** the student is trained on them, and the temperature and threshold are fitted against them. Gold labels are only used to score the results.
- **What escalation means offline:** at evaluation time, "ask the LLM" reads the cached teacher label for that test utterance. The results therefore show exactly what a live cascade would return, provided the live LLM matches the labeling LLM.

---

## 2. The blocks worth knowing cold

If you only have time for a few, learn these. Each one is explained in full in its own section.

1. **`prompts.messages`.** The label list goes in the system message, so every prompt shares the same long prefix. This is what makes prefix caching pay off. ([§5](#5-promptspy))
2. **`teacher.choice_constraint` and `VLLMLabeler`.** Constrained decoding is why invalid outputs drop to zero. ([§6](#6-teacherpy))
3. **The resume loop in `teacher.main`.** It lets labeling survive a free GPU session dying. ([§6](#6-teacherpy))
4. **The training loop in `train.main`.** Hard teacher labels, AdamW with decay groups, warmup, clipping, and no checkpoint picking on validation. ([§8](#8-trainpy))
5. **`export.to_onnx` and `quantize_dynamic`.** How the model leaves PyTorch, how it gets 4× smaller, and how we prove it still behaves the same. ([§9](#9-exportpy))
6. **`calibration.fit_temperature` and `ece`.** What calibration does and how it's measured. ([§11](#11-calibrationpy))
7. **`routing.frontier` and `pick_threshold`.** The threshold rule, its accuracy guarantee, and how ties are handled. ([§12](#12-routingpy))
8. **The middle of `evaluate.main`.** What is fitted on which split, and why. ([§13](#13-evaluatepy))
9. **`serve.classify`.** The live routing decision and the fallback. ([§15](#15-servepy))

---

## 3. `config.py`

### `load_config`

**Plain.** It's the settings sheet. It reads the YAML file once and works out where every file should go, so it doesn't matter which folder you run the command from.

**Technical.**

- `ROOT` is the repository root, two `parent`s up from this file.
- Every entry under `paths:` that isn't absolute is resolved against `ROOT` and turned into a `pathlib.Path`.
- The rest of the code does `cfg["paths"]["reports"] / "metrics.json"` and never thinks about the working directory.

Without this, running `python -m cascade.evaluate` from inside `docs/` would quietly write `reports/` there.

### `base_parser`

Every CLI gets a `--config` flag. Passing a different YAML file is how I ran the smoke test against throwaway paths without touching the real ones.

### `seed_everything`

**Plain.** It makes "random" choices come out the same every run, so results can be repeated.

**Technical.**

- It seeds Python's `random`, NumPy and torch. Torch is imported lazily, so the teacher script can use this module on a machine without torch.
- It does **not** make GPU or MPS kernels bit-exact. Some of those kernels are nondeterministic, so two training runs can differ slightly in the last decimal places.

---

## 4. `data.py`

### `load_clinc`

**Plain.** Downloads the dataset and gets the official list of label names.

**Technical.**

- `load_dataset("clinc/clinc_oos", "plus")` returns a `DatasetDict` with `train` (15,250), `validation` (3,100) and `test` (5,500).
- `features["intent"].names` is the ordered list of 151 label strings, with `oos` at index 42.
- That order defines what "label id 17" means everywhere in the project, including the student's output layer.

### `read_jsonl` / `append_jsonl`

**Plain.** It's a notebook you only ever add lines to. If the power cuts out, everything already written stays written.

**Technical.**

- Appends are line-oriented JSON, and the parent directory is created on demand.
- Append-only matters for crash safety. A process killed mid-run loses at most the chunk it was working on, never the file.
- A JSON array or a Parquet file would need rewriting on every update, so a crash during a write could corrupt everything.

### `teacher_labels`

**Plain.** It converts the teacher's answers from words into numbers, and leaves a blank where the teacher gave a bad answer.

**Technical.**

- It returns a Python list of length `n`. Entry `i` is the label id for utterance `i`, or `-1` if the row is missing or the label is `None`.
- `-1` works as a sentinel because no real label has that id.
- Downstream code relies on this sentinel:
  - `train` drops these rows.
  - `evaluate` masks them out of calibration fitting, and refuses to run if more than 1% are missing.
  - In routing, a `-1` teacher label never equals a student prediction, so it counts as a disagreement, which errs on the safe side.

---

## 5. `prompts.py`

### `SYSTEM` and `messages` (load-bearing)

**Plain.** Picture a stack of 23,850 forms. You could staple the same three-page instruction sheet to the front of every form, or you could hand out the instruction sheet once and then just pass along the one-line question. The second is far faster. I put the long, identical part of the prompt first, so the model only does that work once.

**Technical.**

- The system message holds the instructions plus all 151 labels, one per line, which comes to several hundred tokens. The user message is just the utterance.
- After the chat template is applied, every prompt begins with the same token prefix.
- vLLM's automatic prefix caching (`enable_prefix_caching=True`) stores the attention key/value blocks for that prefix and reuses them. Per utterance, the GPU only has to process the few tokens of the message itself.
- If the labels came *after* the utterance, every prompt would differ from roughly the tenth token onward. The cache would hit almost nothing, and labeling would be several times slower.
- `test_label_list_sits_in_shared_prefix` locks this behavior in with a test.

The line about `oos` matters: it gives the model explicit permission to say "none of these". Without it, a 7B model tends to force every message into the closest intent.

### `label_schema`

**Technical.** It builds a JSON Schema of the form `{"label": <enum of 151 strings>}` with `additionalProperties: false`. The HTTP backend passes it as `response_format`, and servers that support structured output (vLLM, Ollama) compile it into a grammar. The model can then only generate one of the allowed strings.

### `parse_label`

**Plain.** It reads the teacher's answer and decides whether it's one of the allowed labels. It's forgiving about capital letters and stray punctuation, but it never guesses.

**Technical.** The steps are:

1. Strip whitespace.
2. If the text looks like JSON, pull out `"label"`.
3. Strip quotes, backticks and periods, lowercase everything, and turn runs of spaces and hyphens into `_`.
4. Accept the result only on an exact match with the label set.

There is deliberately no fuzzy matching. If "The intent is book_flight" were rescued by a substring search, the free-form invalid rate would look better than the model really is. A substring search can also pick the wrong label when one label name is contained in another (`pto_request` vs `pto_request_status`).

---

## 6. `teacher.py`

### `choice_constraint` (load-bearing)

**Plain.** It's the difference between an essay question and a multiple-choice bubble sheet. On a bubble sheet you can't write something that isn't one of the options.

**Technical.**

- vLLM's structured output uses the allowed strings to build a token-level automaton. At each decoding step, the logits of every token that can't continue a valid choice are masked to −∞. Greedy decoding can therefore only produce one of the 151 strings, exactly.
- The `try/except ImportError` covers an API rename. Newer vLLM uses `StructuredOutputsParams` passed as `structured_outputs=`, while older versions use `GuidedDecodingParams` passed as `guided_decoding=`.
- Without this fallback, the script would only work with one range of vLLM versions, and Kaggle's preinstalled packages change often.

### `VLLMLabeler` (load-bearing)

**Plain.** It loads the big model onto the GPU once, then feeds it thousands of messages at a time.

**Technical.**

- `model="Qwen/Qwen2.5-7B-Instruct-AWQ"` uses 4-bit AWQ weights, about 5.5 GB. FP16 weights would be about 15 GB, which doesn't leave room for the KV cache on a 16 GB T4. vLLM reads the quantization method from the checkpoint's config.
- `dtype="half"`: the T4 (Turing, compute capability 7.5) has no bfloat16 support, so activations run in FP16.
- `max_model_len=2048` caps the sequence length vLLM reserves KV cache for. The prompt is well under that, and a smaller cap leaves more memory for running many sequences at once.
- `gpu_memory_utilization=0.90` is the share of GPU memory vLLM pre-allocates for weights plus the KV cache's memory blocks (PagedAttention).
- `SamplingParams(temperature=0, max_tokens=24)` gives greedy decoding, so the same input always gets the same label. 24 tokens is more than the longest label (`replacement_card_duration`) needs.
- `llm.generate(prompts, ...)` hands the whole chunk to vLLM's scheduler, which uses continuous batching to keep the GPU full. This batching is where the throughput comes from. Sending prompts one at a time would leave the GPU mostly idle.
- `vllm` is imported inside `__init__`, not at the top of the file. That way the rest of the project, and the HTTP backend, work on machines where vLLM isn't installed (vLLM's GPU build needs Linux and CUDA).

### `make_labeler`

It picks the backend. The HTTP path builds a `ChatLabeler` that points at whatever OpenAI-compatible server the config names.

### `summarize`

**Technical.** For each split, it reports:

- the number of rows labeled
- the invalid rate (`label is None`)
- accuracy against gold, with invalid labels counted as wrong
- OOS recall
- total wall-clock minutes, as the sum of per-item time

Rows with `idx >= n` are ignored, so a `--limit` run summarizes only what it was asked to do.

### `main`: the resume loop (load-bearing)

**Plain.** It's like a bookmark. Before starting, it checks which pages are already done and skips them. After each chunk it writes down what it finished. If Kaggle kills the session halfway through, you rerun the same command and it carries on.

**Technical.**

- `done` is the set of indices already in `data/teacher/{split}.jsonl`. `todo` is everything below `n` that's missing.
- The work is processed in chunks: 2,048 for vLLM, or `workers × 16` for HTTP. After each chunk:
  - the rows are appended to disk;
  - per-item time (`chunk seconds / chunk size`) is stored on every row, so the total time stays correct across resumed sessions.
- The labeler is built lazily, only when the first chunk actually needs labeling. A rerun on a complete cache never loads the 7B model, so it finishes in seconds.
- HTTP rows whose raw output is `None` (network failures) are **not** written, so the next run retries them. Rows with a bad label *are* written, because a bad label is a real model output that should be counted.
- `--mode free` drops the constraint and writes to `data/teacher_free/`, so the free-form comparison never pollutes the training labels.
- The report file is merged, not overwritten, so labeling splits in separate sessions builds up one complete report.

---

## 7. `llm.py`

### `ChatLabeler`

**Plain.** It's a phone line to any LLM server that "speaks OpenAI". That covers Ollama on your laptop, a vLLM server, or a hosted endpoint. It's used both for labeling without a GPU and for live escalations.

**Technical.**

- It uses the `openai` client library with a custom `base_url`. Local servers ignore the API key, so the key is a dummy.
- `response_format` with a strict JSON schema is the HTTP equivalent of vLLM's choice constraint.
- `extra_body` passes server-specific fields straight through. The Qwen3.8 config uses it to send `reasoning_effort: "none"`, which turns off the model's thinking mode. Thinking on, each label would cost seconds of hidden reasoning tokens. Thinking off, the answer is a handful of tokens.
- `timeout=30, max_retries=1`: a request that hangs mustn't block the service for long.
- Any exception returns `None`, never raises. During labeling, `None` means "retry later". In serving, it means "fall back to the student".
- `__call__` runs requests through a `ThreadPoolExecutor`. These calls spend almost all their time waiting on the network, so threads give real concurrency despite Python's GIL. Results come back in input order because `pool.map` preserves order.

---

## 8. `train.py`

### `pick_device`

It uses CUDA, then Apple MPS, then CPU. On an M-series Mac, MPS makes training take minutes instead of most of an hour.

### `batches`: dynamic padding

**Plain.** It groups sentences into batches and pads each batch only to the length of its own longest sentence, not to a fixed size.

**Technical.**

- `collate` tokenizes one batch at a time with `padding=True` (pad to the longest in the batch) and `truncation` at 64 tokens. It returns `input_ids` and `attention_mask` of shape `[B, L_batch]` plus `labels` of shape `[B]`.
- CLINC utterances are short, about 8–15 tokens. Padding every batch to 64 would mean most of each batch's compute goes into pad tokens.
- The attention mask makes pad positions irrelevant to the result.

### `agreement`

It returns the share of validation utterances where the student's argmax equals the teacher's label. It is logged every epoch so you can see learning level off. It is **not** used to choose a checkpoint (see below).

### `main`: the training loop (load-bearing)

**Plain.** The student studies flashcards: the message on the front, the teacher's answer on the back. It goes through the deck 4 times in random order. It starts gently, takes big steps once warmed up, slows down toward the end, and is stopped from making any wildly large correction in one go.

**Technical.**

- **Data.** It trains on the training texts paired with the **teacher's** label ids. Rows where the teacher gave no valid label are dropped. Gold labels are never read here.
- **Model.** `AutoModelForSequenceClassification` on `distilbert-base-uncased`:
  - 6 transformer layers, hidden size 768, 12 attention heads.
  - The classification head runs the first-token vector through `pre_classifier` (768→768) → ReLU → dropout 0.2 → `classifier` (768→151).
  - About 67M parameters in total.
  - `id2label`/`label2id` are stored in the checkpoint config, so the exported model is self-describing.
- **Loss.** Cross-entropy on hard labels. The model computes it internally when `labels` is in the batch.
- **Weight decay groups.** Decay goes on parameters with `ndim > 1` (weight matrices and embeddings) and not on `ndim ≤ 1` (biases and LayerNorm scales).
  - Decaying LayerNorm scales pulls them toward zero, which fights the normalization they exist to do.
  - I originally matched on parameter names. That missed DistilBERT's `sa_layer_norm` and `output_layer_norm`, because they aren't called `LayerNorm`. Grouping by `ndim` can't miss a parameter that way.
- **Optimizer and schedule.** AdamW at lr 5e-5. The learning rate rises linearly over the first 6% of steps, then falls linearly to 0.
  - Warmup matters because the classifier head starts random. Large early updates would push big, noisy gradients back into the pretrained layers.
- **Gradient clipping at 1.0.** If the gradient's overall norm exceeds 1.0, it is scaled down. This guards against one bad batch spiking the weights.
- **No best-epoch selection.** I train a fixed 4 epochs and keep the last one. Validation is reserved for fitting the temperature and the threshold. Using it to pick a checkpoint too would make it a third thing tuned on the same data, so the threshold estimate would be optimistic.
- **Saving.** The model goes back to CPU before `save_pretrained`, with the tokenizer and `labels.json` saved next to it. Everything downstream loads from this one folder.

---

## 9. `export.py`

### `LogitsOnly`

**Plain.** It strips the packaging so only the useful part gets shipped.

**Technical.** Hugging Face models return a `SequenceClassifierOutput` object. ONNX export needs plain tensors in and out. The wrapper's `forward(input_ids, attention_mask)` returns just the `[B, 151]` logits tensor, which also fixes the graph's input and output names.

### `to_onnx` (load-bearing)

**Plain.** It translates the model from PyTorch's own format into a standard format (ONNX) that a fast, small inference engine can run without PyTorch installed.

**Technical.**

- **Example input.** The sample has **two** texts of different lengths. `torch.export` traces with real tensors and specializes any dimension whose size is 0 or 1. A single-text sample would fix the batch dimension at 1 in the exported graph.
- **Dynamic shapes.** `dynamic_shapes` marks dim 0 (batch) and dim 1 (sequence) as symbolic with `torch.export.Dim`, so the same graph accepts `[1, 9]` in serving and `[256, 23]` in evaluation.
- **Opset and file layout.** It uses `opset_version=18`. `external_data=False` keeps the weights inside the single `.onnx` file, which works because it's under the 2 GB protobuf limit.
- **Stripping `value_info`.** The new exporter writes shape annotations for intermediate tensors, and ONNX Runtime's quantizer re-runs shape inference over them. On this model, one annotation clashed with the inferred shape (768 vs 151) and quantization crashed. Clearing `graph.value_info` removes only the annotations. The real shapes are recomputed, and the computation itself is unchanged.

### `quantize_dynamic`

**Plain.** It's like storing every measurement rounded to the nearest centimetre instead of to the micrometre. The numbers get much smaller to store and faster to work with, and the answer is nearly always the same.

**Technical.**

- **Weights.** For each `MatMul`/`Gemm` weight, ONNX Runtime stores int8 values plus a scale and zero point.
- **Activations.** These are quantized on the fly at runtime, using the actual min/max of each input. That's the "dynamic" part, and it means no calibration dataset is needed.
- **Speed.** The CPU then uses int8 matrix multiplies (VNNI on x86, the dot-product instructions on ARM), which are faster than FP32.
- **Size.** The file shrinks from about 257 MB to about 65 MB.
- **Why not static quantization.** Static quantization would also fix activation ranges ahead of time. That needs representative data, and it is fragile around LayerNorm and softmax in transformers. Dynamic is the standard choice for BERT-style models on CPU.

### `torch_argmax` and the parity report (load-bearing)

**Plain.** It's a taste test: the copy has to give the same answers as the original.

**Technical.**

- It runs all 3,100 validation texts through PyTorch, the FP32 ONNX graph and the INT8 graph, and reports the share of argmax predictions that agree with PyTorch.
- FP32 ONNX should come out at exactly 1.0. Anything less means the export is broken.
- INT8 will be slightly under 1.0. The few flips are utterances that were already close calls between two labels.
- This check is the evidence behind the claim that the quantized model is the same model, only faster.

---

## 10. `student.py`

### `Student`

**Plain.** It's the fast engine that runs the small model on a CPU, with no PyTorch needed.

**Technical.**

- `ORT_ENABLE_ALL` turns on ONNX Runtime's graph optimizations, for example fusing the attention, GELU and LayerNorm subgraphs into single kernels.
- `intra_op_num_threads` sets how many threads one operator (one big matmul) can use. `inter_op_num_threads=1`, because this graph is a straight chain with nothing to run in parallel at the graph level.
- `encode` casts token ids and the mask to `int64`, because that's the dtype the graph was exported with. Feeding `int32` raises an error.
- `logits` runs batches of 256 and returns `float64`. Calibration math (log of small probabilities, exp of scaled logits) is more stable in double precision, and at this size the cost doesn't matter.

---

## 11. `calibration.py`

### `softmax`

**Technical.** It computes `exp(z / T) / Σ exp(z / T)` with the row max subtracted first. Subtracting the max doesn't change the result mathematically, but it prevents `exp(1000)` from overflowing. `test_softmax_rows_sum_to_one` feeds in logits of ±1000 to check this.

Dividing by `T` never changes which entry is largest, so **temperature scaling cannot change any prediction**. It only changes how confident each prediction looks. `test_temperature_keeps_argmax` checks this.

### `fit_temperature` (load-bearing)

**Plain.** Picture a weather forecaster who says "90% chance of rain" on days when it actually rains 70% of the time. You don't fire them. You learn to tone down everything they say by the same amount. Temperature is that single "tone it down" dial. Above 1 makes the model less sure of itself; below 1 makes it more sure.

**Technical.**

- It finds the `T` that minimizes the negative log-likelihood of the teacher's labels on validation, using `softmax(logits / T)`.
- It searches over `log T` in [−3, 3], so T can range from about 0.05 to about 20, using SciPy's bounded Brent method.
- Searching in log space keeps T positive and treats "twice as hot" and "twice as cold" symmetrically.
- NLL is convex in `1/T`, so this one-dimensional search finds the global minimum.
- Only one number is learned from 3,100 examples, so it is very hard to overfit. More flexible methods (vector scaling would learn 302 numbers) can fit validation better and test worse.
- `test_fit_recovers_known_temperature` generates labels from a known T = 2.5 and checks that the fit recovers it to within 0.15.

### `reliability` and `ece` (load-bearing)

**Plain.** ECE is a grade for honesty. Group all the answers by how confident the model claimed to be. In each group, compare the claimed confidence with how often it was actually right. ECE is the average gap, weighted by how many answers fall in each group. Zero means perfectly honest.

**Technical.**

- 15 equal-width bins over (0, 1].
- For each non-empty bin it records (mean confidence, accuracy, count).
- `ECE = Σ_b (n_b / n) · |acc_b − conf_b|`.
- The tests check two known cases:
  - Confidence 0.75 with 75% accuracy gives ECE = 0.
  - Confidence 0.99 with 50% accuracy gives ECE = 0.49.
- Known weaknesses of ECE:
  - It depends on the binning.
  - It only looks at the top-1 confidence.
  - It can be gamed by a model that always outputs its average accuracy.

  That's why `summary` also reports NLL and the Brier score, which are proper scoring rules and can't be gamed that way.

### `nll`, `brier`

- **NLL** is the average of −log(probability given to the correct label). It is clipped at 1e-12 so a single zero doesn't produce infinity.
- **Brier** is the mean squared distance between the probability vector and the one-hot target, summed over all 151 classes.

---

## 12. `routing.py`

### `frontier` (load-bearing)

**Plain.** Line up every validation message from most confident to least. Then walk down the line. At each step, ask: "if the student answered everyone up to here, how many people would it handle, and how many times would it disagree with the teacher?" That gives a menu of every possible threshold and its cost.

**Technical.**

- Sort confidences descending, with a stable sort so ties keep a deterministic order.
- `misses` is the running count of `student != teacher` in that order.
- A threshold rule `conf ≥ t` must accept **all** tied items together. You can't let one message at confidence 0.93 through and hold back another at 0.93. So only positions where the next confidence is strictly lower (or the end of the list) are valid cut points. `np.r_[ranked[1:] < ranked[:-1], True]` marks exactly those.
- It returns three aligned arrays: thresholds, coverage (accepted / n) and disagreement (misses / n).
- Both coverage and disagreement only go up as the threshold goes down. `test_frontier_groups_ties` checks the tie handling.

### `pick_threshold` (load-bearing)

**Plain.** The rule: let the student answer as many messages as possible, as long as it disagrees with the teacher on no more than 1 in 100 messages overall.

**Technical.**

- Because disagreement never decreases along the frontier, the valid cut points form a prefix of it. The last one in that prefix has the highest coverage.
- If no cut point fits (even the single most confident message is a disagreement and the budget is 0), it returns `inf`: escalate everything.

**Why this rule, and the guarantee behind it.** The cascade's answer differs from the teacher's only on messages where the student answered *and* disagreed. On each of those, the cascade's correctness changes by at most one message compared with the teacher. So:

```
|accuracy(cascade) − accuracy(teacher)| ≤ disagreement share ≤ budget
```

With a 1% budget, the cascade can't be more than 1 percentage point worse than the teacher **on the data the threshold was picked on**. On test, that bound becomes an estimate, and the test results show how well it carried over. The actual gap is usually smaller than the bound, because when the student and teacher disagree, the student is sometimes the one that's right.

None of this needs gold labels, so the same rule works in a real deployment.

### `route`

`np.where(conf >= threshold, student, teacher)`, plus the accept mask. It's the whole cascade in one line.

### `accuracy_curve`

**Technical.**

- It uses running counts of student-correct and teacher-correct in confidence order, each with a leading 0, so that index `k` means "k accepted".
- Cascade accuracy at `k` = (student-correct among the top k + teacher-correct among the rest) / n. The second term is `t_hit[-1] − t_hit[k]`.
- It is evaluated at every tie-group boundary, plus 0% coverage (pure teacher).
- This is the curve in `reports/coverage.png`. It shows the whole tradeoff, not just the one point I chose.

---

## 13. `evaluate.py`

### `main`: what is fitted where (load-bearing)

**Plain.** Validation is the practice exam, used to set the dials. Test is the real exam, which we look at once, after every dial is set.

**Technical.** In order:

1. **Load the student.** Load the INT8 student, or FP32 with `--fp32`. The model being calibrated must be the one that ships, because quantization nudges the logits slightly, and a temperature fitted on FP32 would be slightly wrong for INT8.
2. **Load labels and compute logits.** For validation and test, load gold and teacher ids and compute logits (`[3100, 151]` and `[5500, 151]`). Stop if more than 1% of teacher labels are missing.
3. **Fit `T`** on validation logits against validation **teacher** labels, using only rows where the teacher label is known.
4. **Pick `τ`** on validation: calibrated confidence, student argmax and teacher ids, under the disagreement budget.
5. **Score on test.** Apply `T` and `τ` to the test logits once, and route against the cached teacher labels.
6. **Score three systems against gold:** teacher only, student only and the cascade. For each, report accuracy, in-scope accuracy and OOS recall.
7. **Measure calibration** on test before and after temperature scaling, against teacher labels (what the router actually relies on) and against gold (what people usually mean by calibration).
8. **Write the outputs:** `metrics.json`, `RESULTS.md`, the two plots, and `artifacts/routing.json`. `routing.json` is the only thing the server needs besides the model.

A few metrics are worth explaining:

- **`escalated_oos_share`** is the share of escalated messages that are actually out-of-scope. Compare it with the out-of-scope share of the whole test set, which is 18.2%. If escalation picked messages at random, it would be about 18%. In my main run it was 30.4%, so low confidence does pick out out-of-scope messages. With the weaker 7B teacher it was only 19.3%.
- **`gap_pp`** is the cascade's accuracy minus the teacher's. A negative value is a loss.
- **`oos_recall_gain_pp`** is the cascade's OOS recall minus the student's alone, which is what escalation adds.
- **The threshold** is stored at full precision in `routing.json`. Rounding it for display only would change which messages pass right at the boundary.

---

## 14. `bench.py`

### `time_one`

**Plain.** It's a stopwatch. We time a thousand single requests and report the typical time (median) and the slow-day time (95th percentile).

**Technical.**

- **Batch size 1**, because that's how live requests arrive. Batched throughput would look far better per item and would answer a different question.
- **Each text is encoded on its own,** so there's no padding. The sequence length is the real length, as in serving.
- **50 warm-up runs first.** They let ONNX Runtime allocate memory and start its thread pool, and they warm the CPU caches. Without them, the first few calls would distort the tail.
- **Two measurements:**
  - `model_only` times just `session.run` on pre-tokenized input.
  - `tokenize_to_decision` times tokenization, the forward pass, the softmax with T, and the threshold compare: everything the server does before it decides whether to escalate.
- **The p95 matters** because a cascade's users feel the tail, not the average.

### `main`

It times INT8 and FP32 at each configured thread count and records the CPU model, from `sysctl` on macOS or `/proc/cpuinfo` on Linux. A latency figure without the hardware attached means nothing.

---

## 15. `serve.py`

### `build_app` (factory)

**Plain.** It assembles the service from parts. Tests can swap in fake parts, so they don't need a real model or LLM.

**Technical.**

- `uvicorn cascade.serve:build_app --factory` calls this function at startup.
- `student`, `teacher` and `routing` can be injected, and `tests/test_serve.py` passes fakes.
- Environment variables override the config:
  - `CASCADE_LLM_URL` / `CASCADE_LLM_MODEL` point escalations somewhere else.
  - `CASCADE_NO_LLM` disables escalation entirely.
  - `CASCADE_THREADS` sets ONNX Runtime threads (default 1).
- The default of one thread per request keeps latency predictable when several requests run at once.

### `classify` (load-bearing)

**Plain.** The nurse looks at the patient. If they're sure, they answer. If not, they phone the doctor. If the doctor doesn't pick up, the nurse gives their best guess and writes "doctor unavailable" on the chart.

**Technical.**

1. Encode the one text, run the forward pass, and compute `softmax(logits / T)`. This produces a 151-vector.
2. Take the argmax and its probability.
3. If `conf < τ`, call `teacher.label(text)`:
   - a valid label → `source="llm"`;
   - `None` (timeout, connection error, unparseable output) → keep the student's answer with `source="student_fallback"`.
4. Return the label, the confidence, the source, and `latency_ms` measured around the whole thing, including any escalation.

More details:

- **Graceful degradation.** The service never returns a 500 because the LLM is down. The caller can still see from `source` that the answer is lower-trust.
- **Sync endpoint.** It is a plain `def`, so FastAPI runs it in its worker threadpool. The blocking LLM call doesn't stall the event loop, and ONNX Runtime's `run` is safe to call from several threads at once.
- **Input validation.** `Query` uses `Field(min_length=1, max_length=2000)`, so empty strings get a 422 and huge payloads are rejected before tokenization.

### `health`

It reports whether the service is up, plus T, τ, and whether an LLM is configured. This lets a deploy check confirm the right routing file was loaded.

---

## 16. Tests, config and build files

### `tests/`

| File | What it checks |
|---|---|
| `test_calibration.py` | Softmax stability, argmax unchanged by T, temperature recovery, ECE on two hand-computed cases |
| `test_routing.py` | Tie grouping in the frontier, budget respected, escalate-everything case, `route`, accuracy-curve endpoints |
| `test_prompts.py` | Plain, messy, JSON and invalid outputs; label list in the shared prefix |
| `test_serve.py` | Confident → student, unsure → LLM, LLM down → fallback, empty text → 422 |

None of the tests needs a GPU, a downloaded model or a running LLM, so the suite runs in seconds anywhere.

### `configs/default.yaml`

Every tunable setting lives in this one file (see [PROJECT.md §3](PROJECT.md#3-configuration)). The seed is fixed at 13.

`configs/qwen3.8.yaml` is the same file with three changes: the Qwen3.8 teacher, thinking turned off, and its own data, artifact and report folders, so the two runs never overwrite each other.

### `pyproject.toml`

- Python 3.10–3.13. 3.14 is excluded because some ML wheels can lag a new Python release.
- `vllm` is an optional extra (`teacher`), because the GPU build needs Linux and CUDA.
- `onnxscript` is required by the torch ONNX exporter.

### `Makefile`

It holds short names for each stage. `make pipeline` runs train → export → evaluate → bench.
