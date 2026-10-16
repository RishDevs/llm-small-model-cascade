# What this project is and why it matters

## The short version

I built a system that classifies what a user wants from a short message, like "what's my checking balance" or "book me a flight to Denver." It uses two models:

- **A large open language model.** In my main run it's Qwen3.8 with 27.8 billion parameters. It's accurate, but it's slow and needs a lot of memory.
- **A small model about 400 times smaller.** It runs on a normal CPU in about 2 milliseconds.

The large model teaches the small one. In production, the small model answers every message it's confident about and passes the rest to the large model. The goal is that the combined system is nearly as accurate as the large model alone, while the large model only sees a small share of the traffic.

## The problem it solves

LLMs are now the easy default for text classification. You write a prompt that lists the categories, send the message, and read the answer back. No training data and no model to maintain, and it works well.

The trouble starts at volume. Every request costs a GPU forward pass or an API call: hundreds of milliseconds to seconds, and real money. Most of that traffic is easy. "Set an alarm for 7" doesn't need a 7B model. A support system, a voice assistant or an email router can pay LLM prices for millions of messages a day when only a small fraction of them actually need an LLM.

The obvious fix is to train a small classifier. That runs into two problems:

1. **You need labeled data.** Teams usually don't have it, which is often why they reached for an LLM in the first place.
2. **A small model doesn't know when it's out of its depth.** It answers everything with the same apparent confidence, including messages it has no business answering.

This project handles both problems:

- The LLM produces the training labels, so no human labeling is needed.
- The small model's confidence is calibrated, so it can be trusted to decide when to hand a message back.

## Why it isn't trivial

Each of these is a real place the project could have gone wrong:

**1. LLM labels drift when they're free text.** Asked for a label in free text, a 7B model sometimes adds extra words or invents a label that isn't on the list. More often, it just picks a worse one. I constrain decoding so the model can only produce one of the 151 valid label strings. On the same 2,000 utterances, that cut invalid outputs from 0.45% to zero and raised the teacher's accuracy from 67.8% to 74.6%. The accuracy gain mattered far more than the validity.

**2. Confidence scores lie.** A fine-tuned transformer will happily say it's 99% sure and be wrong far more often than 1% of the time. If routing trusts raw softmax output, the threshold is meaningless. I fit a temperature on held-out data so that "90% confident" should mean right about 90% of the time, then measure whether it helped. In this run it didn't need to. Trained on the teacher's labels, the student was already well calibrated against the teacher (ECE 0.021). Temperature scaling barely moved anything, and ECE came out slightly worse. The point is that the check exists and the result is reported, not that it always wins.

**3. Choosing the threshold is a tradeoff that needs a rule.** Set the threshold too low and accuracy drops. Set it too high and you're back to calling the LLM for everything. I pick it with a rule that has a guarantee: the student is only allowed to answer where it disagrees with the teacher on at most 1% of traffic. Because of that, the cascade's accuracy can't fall more than 1 percentage point below the teacher's on the validation data where the threshold is picked. The test set then shows how well that holds on data the threshold has never seen. I didn't tune for a number. I tuned for a bound.

**4. Out-of-scope messages are the hard part.** Real users ask things the system was never built for. CLINC150 is designed to test exactly this:

- Only 1.6% of training examples are out-of-scope.
- 18% of the test set is out-of-scope.

A small model trained on so few out-of-scope examples tends to force those messages into some real intent. In my main run, the cascade's out-of-scope recall was 89.6%, against 79.3% for the student alone. The student is less sure on these messages, so they make up 30% of escalations against 18% of traffic, and the teacher is better at saying "none of these."

**5. Speed claims are easy to fake.** I measure latency on CPU at batch size 1, which is how a real request arrives, and I report the 95th percentile, not just the median. I also check that the INT8 model makes the same predictions as the full-precision one, and I report the share of predictions that change.

## What makes it more than a tutorial

Most distillation or fine-tuning tutorials stop at "I trained a small model and it got X% accuracy." This project is a complete system, and each piece is measured:

- **No human labels in the loop.** The student, the calibration and the threshold are all fitted against the teacher's labels. Gold labels are used only to score the result, so the setup matches a team that has an LLM and no labeled data.
- **Routing with a stated guarantee.** The threshold rule bounds the accuracy loss instead of hand-picking a number.
- **Teacher choice measured.** The same pipeline with a 7B teacher and a 27.8B teacher shows how much teacher quality drives coverage: 45% against 68% of LLM calls avoided.
- **A test set built to break it.** The out-of-scope share jumps from 3% to 18% between validation and test, so the routing has to hold up under a real distribution shift.
- **Built for free hardware.** Labeling is resumable, so it survives a free GPU session being killed. The student runs on any laptop CPU.
- **Fails safely.** If the LLM is down, the service still answers from the student and marks the response as a fallback.
- **Every speed-up is checked.** The ONNX export and the INT8 quantization are each compared against the original model's predictions before they're used.

## Where this applies

The same pattern works anywhere an LLM classifies high-volume text:

- intent detection in a chatbot or voice assistant
- routing support tickets
- first-pass content moderation
- tagging emails or documents

The LLM stays in the loop for the hard cases. The cheap model takes the bulk.

## What it isn't

- **Not a general replacement for the LLM.** The student only knows the 151 labels it was trained on.
- **Not a way to beat the teacher.** The goal is to match the teacher's accuracy at a fraction of the cost. When the cascade does better than the teacher on some examples, that's a side effect, not the design.
- **Not production infrastructure.** It's a working service with tests, but it doesn't include auth, autoscaling or monitoring.
