## There are codebases with more efficient RL sampling and generation than yours. LeanRL does not support all of that yet. Why should I use LeanRL?

Yes. Some open source codebases are more optimized for sampling and generation today. That is a good thing, and we actively learn from them. LeanRL’s focus is different. We prioritize a codebase that is easy to understand, modular, and safe to modify without hidden coupling or framework magic. If you care about moving fast on research, debugging quickly, and extending algorithms with confidence, LeanRL is built for that workflow.

Not being the most optimized does not mean being impractical. LeanRL is designed to be production grade and scalable, so you can train and post train large models with standard, widely used methods. Over time, we plan to incorporate proven optimizations as long as they do not compromise clarity and maintainability.


## There are differences between your implementation of methods like GRPO. Why is that the case?

That is correct. RL training is sensitive to small implementation details, and some important details are often overlooked when applying RL to large models, unlike classic RL settings such as games. As a result, in some places LeanRL makes deliberate choices to improve stability and performance, even if that means it does not match a specific reference implementation line for line.

When the differences are intentional, we document them and name variants explicitly. For example, you may see SGRPO, which indicates a GRPO style method with stability focused implementation choices and address some of GRPO limitations. 


## I found a bug. What should I do?

That is wonderful. Please open a GitHub issue with steps to reproduce, expected behavior, and actual behavior. If you can include logs, config files, or a minimal script, it will help a lot. Pull requests are also welcome, and we will review them as quickly as possible.

## I have a few research ideas and want guidance. Can you help?

We can try. If you are comfortable sharing your idea publicly, open a GitHub issue and include enough context for others to follow along. If you prefer to discuss privately, you can email Rasool. Contact details are available on the [Rasool's website](https://rasoolfa.github.io/).

