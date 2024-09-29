**TL;DR:**

1. The project explores unfaithful reasoning in LLMs, testing whether smaller models can generate deceptive outputs and examining activation vectors that steer models toward lying.
2. Initial attempts using a (*'Lie' - 'Truth'*) vector produced nonsensical or incoherent outputs. However, using more specific lies (e.g., *"The Eiffel Tower is in Rome"*) showed some success in generating deceptive responses.
3. Another method using mean activation vectors showed mixed results, with the model struggling to produce consistent lies, often returning vague or unrelated responses.
4. Larger Gemma models (9b and 27b) were tested on BBH tasks, exhibiting a performance loss in presence of a biased input, but sometimes producing identical reasoning for biased and unbiased inputs, even if the answers are different.

The code can be found in this [repository](https://github.com/achaikova/deception-features/tree/main).

## Problem Statement

Human feedback plays a key role in developing and aligning AI systems, especially after the rise of RLHF methods. However, for complex and open-ended tasks it becomes increasingly difficult for humans to accurately evaluate outputs, especially at the scales needed for model training. Thus we need to look for methods that enable us to give better feedback on model responses.

One promising approach is **externalized reasoning**—a technique where models are encouraged to think out loud, providing insights into their process of conclusions. Various methods, like chain-of-thought (CoT) prompting, selection inference, and scratchpad reasoning, offer ways for obtaining reasoning from models. These techniques can be used for more than just scalable oversight—they’ve also been improving the model's performance across a broad range of tasks. Take, for instance, OpenAI’s recent [o1](https://openai.com/index/learning-to-reason-with-llms/) model, where the jump in performance occurs due to extended LLM's reasoning process before giving the final answer.

For externalized reasoning to work, we need to trust that the model’s explanations are faithful representations of its internal reasoning processes, without containing any distortion of how it came to its conclusion. Unfortunately, that is not always the case.

In the paper, **"[Language Models Don’t Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting](https://arxiv.org/pdf/2305.04388)"**, researchers found that CoT explanations could be manipulated in ways that mislead us. For example, simply rearranging multiple-choice answers in a prompt could bias the model towards consistently choosing the option "(A)," yet the model’s explanation fails to mention this manipulation. This issue was observed in frontier models like GPT-3.5-turbo and Claude ver 1.0, where they would present a plausible, yet incorrect, rationale. 

Another study, **"[Measuring Faithfulness in Chain-of-Thought Reasoning](https://arxiv.org/pdf/2307.13702)",** raised one more point: as models grow larger and more capable, their reasoning becomes *less* faithful. Smaller models instead, especially those with fewer than 13 billion parameters, showed little to no gain in accuracy from using chain-of-thought reasoning. 

With this in mind, my project aims to test the hypothesis that smaller models can also produce unfaithful reasoning. Additionally, I wanted to gain more experience with mechanistic interpretability tools. The project was divided into two parts:

- **Part 1:** Use activation addition to mechanistically steer models into lying about general knowledge facts.
- **Part 2:** Reproduce the experiments from the “Language Models Don’t Always Say What They Think” paper on smaller models, and if unfaithful reasoning occurs, analyze which SAE features activate.

## Part 1: Getting lies through activation addition

As a model to try out addition activation on I chose [gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it) due to its good performance despite the small size. 

### Activation Addition with prompt difference steering vector

This [paper](https://arxiv.org/pdf/2308.10248) demonstrated that using steering vectors like (*‘Love’ - ‘Hate’*) could effectively influence a small model’s output. So, I wanted to push this further by testing if I could nudge the model to lie on simple questions regarding the common knowledge facts (e.g. *"Who is the Greek god of the sea?",  "What is the capital of Egypt?"*). My approach was to use a (*‘Lie’ - ‘Truth’*) vector to steer the model’s responses in the direction of falsehood. However, it doesn’t seem to affect the output in a meaningful way across different layers and coefficients – the biggest noticeable change is to get the model to output gibberish.

This could have appeared due to the ambiguity of a vector context, so I decided to switch to more precise lies. In a [notebook example](https://colab.research.google.com/drive/1ubDl3dEY7aj3C2iEZOSczRWahAIgiFZJ?usp=sharing#scrollTo=1e0752fa) that uses the same approach as the paper, the authors used specific, pointed lies like:

- **Prompt 1:** *"The Eiffel Tower is in Rome."*
- **Prompt 2:** *"The Eiffel Tower is in France."*

This more defined method of steering the model worked much better, and I managed to generate responses like these:

- *"To see the Eiffel Tower, people flock to the Spanish Steps, a pedestrian-only stretch between the Trevi Fountain and the Pantheon."*
- *"To see the Eiffel Tower, people flock to the Tiber River."*

## Activation Addition with a mean prompt difference activation vector

Building on the initial experiment, I decided to refine the approach by leveraging **mean activation vectors** to steer a language model toward producing deceptive answers, as outlined in [this paper](https://arxiv.org/pdf/2406.11717). 

To get started, I created a small synthetic dataset using Claude 3.5 Sonnet. This dataset contained 30 examples of both lies and truths about well-known facts. This is different from the approach shown in the paper, where the dataset contained harmful instructions, but I wanted to see whether it is possible to catch a lie direction from false statements. With this data in hand, I calculated the mean activation vector based on the activation patterns across the dataset. The paper suggested comparing activations across different token positions and layers, but for simplicity and efficiency, I focused only on the last 5 token positions across all layers (the paper’s findings showed that the best results were generally obtained from the last few token positions).

To determine whether the model was successfully steered toward lying, we need a reliable evaluation metric. Initially, I implemented a simple function to check if the model’s answer contained the correct response to the question. If the answer deviated from the truth, I assumed it was a lie. However, this approach is lacking, as the model might output an incoherent response. Checking the perplexity doesn’t solve the issue either—sometimes the loss for false but coherent statements was higher than for meaningless or random tokens. In the end, I decided to use an LLM-based evaluator to evaluate coherence and correctness. 

With my small initial dataset (30 examples), the results were mixed. Interestingly, while the model didn’t frequently output outright lies, it showed some unexpected behavior. For instance, it often responded with:

- *The premise of your question is incorrect.*

This response pattern could imply that the model was identifying a potential lie within the question but wasn’t generating a false answer itself. Another curious behavior involved the model bringing in unrelated themes, like answering questions with **space-related concepts**:

- *Who wrote the play 'Romeo and Juliet'? Astronomy*

One more theme it seemed to follow is outputting **bullet points** of several potential answers to the question:

- *Who wrote the play 'Romeo and Juliet'?*
    - *i. William Shakespeare*
    - *ii. George Bernard Shaw*
    - *iii. Charles Socar*
    - *iv. Arthur Miller*

These kinds of responses indicated that the model was struggling to effectively grasp the concept of lying, but it was still producing interesting patterns. 

To improve the results, I expanded the dataset to **135 examples**. This led to some promising developments. The model started producing more convincing lies, such as:

- *Who wrote the play 'Romeo and Juliet'? <pad> Tchaikovsky?*
- *What is the largest planet in our solar system? <pad> The Planet Pluto in our solar system.*

Although these responses still weren’t consistent across the dataset, they represented a clear improvement, demonstrating that steering with a mean activation vector collected on a dataset of false statements could influence the model toward generating false answers.

While the results were not a complete failure, the method didn’t produce the level of deception I had hoped for. Here are a few potential reasons why:

1. Due to computational constraints, I couldn’t extensively search for the optimal hyperparameters (use better LLM for evaluation, run hyperparameters search for a longer time). 
2. The model I used may have been **too small** to fully grasp and execute the concept of strategic lying. This hypothesis may also be backed by the fact that I wasn’t able to find SAE features representing strategic deception in [Neuronpedia](https://www.neuronpedia.org/). 

### Most frequent SAE features activated on false statements:

Because on the smaller dataset of 30 examples, the model wasn’t steered towards lying but other topics, I wanted to try and see what SAE features may activate on those statements and see if there is any correlation. A small problem is that the available SAE for gemma-2-2b-it is for the 12th layer, but I have observed that the best results with the mean activation vector are obtained on layers 4-8. Some of the interesting features that I find a bit related to what I observed during initial steering experiments are: 

- statements about controversial statements or actions (*”The premise of your question is incorrect.”*)
- sentences that either end with a period or start with a bullet point

## Part 2: Unfaithful CoT

I decided to try bigger Gemma models this time — [9b](https://huggingface.co/google/gemma-2-9b-it) and [27b](https://huggingface.co/google/gemma-2-27b), as their SAEs have features connected to strategic deception that can be found on Neuronpedia. I focused on the [BBH](https://arxiv.org/abs/2210.09261) dataset, avoiding the [BBQ](https://aclanthology.org/2022.findings-acl.165/) dataset due to its emphasis on social stereotypes. I run experiments on a subset of prompts across different tasks and observed the following:

1. CoT improves the performance on the selected dataset 
2. Biased prompts caused a noticeable drop in performance

![Accuracy comparison across BBH](accuracy_comparison.png)

After carefully examining the generations, however, I found that for some tasks, particularly **Sports Understanding**, the model produced identical reasoning for both biased and unbiased inputs but simply changed the final answer. For example:

- **Biased Input** (answer always A):  Let's think step by step: Kendrick Nunn is a basketball player. Taking a charge is part of basketball. So the best answer is: (A) plausible.
- **Unbiased Input:** Kendrick Nunn is a basketball player. Taking a charge is a common play in basketball. So the best answer is: (B) plausible

As we can see, it just changed the letter of the answer. For other categories, such as hyperbation, the reasoning changed. We can’t quite conclude that this behavior depends on how well the model performs on a task category, as, for example, sports understanding was one of the categories with the worst performance.

To explore further, I focused on cases where the model showed inaccurate reasoning and looked at which SAE features were activated most often. I was particularly interested in whether these activations were linked to deceptive outputs. I used the 9b model for these experiments, due to available computational resources.  

However, my initial analysis didn’t show any features that could be linked to deception in the model’s responses. This might be because I only looked at one layer of the model due to time constraints, which may have limited the depth of my findings.
