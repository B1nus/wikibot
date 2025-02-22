# Usage
```
$ python wikibot.py
Hello I am wikibot! Ask me anything.
>
```
Write your question. For example:
```
> What does the fox say?
```
# MVP
- Homemade zig matrices
- REINFORCE
- No self attention
- No tokenizer
- 128 character context window
- Get through entire dataset

# Goals
- Take less space than the dataset (14 gb)
- Correct responses

# Hmmm
- Because we have small context windows, removing the tokenizer would be possible and prefered
- on-policy reinforcement learning sounds really cool
- Try a simple neural network without attention
- the wikibot doesn't have to be smart, maybe I can train it locally?
- How to teach the wikibot about itself
- How to handle the large dataset
- How to make it understand the situation it's in so it gives good responses

To get it to learn wikipedia it might be necessay to have multiple versions of wikipedia written in different wordings by an llm.

# Resources
- [wikipedia dataset](https://github.com/GermanT5/wikipedia2corpus)
- [llm from scratch](https://m.youtube.com/watch?v=kCc8FmEb1nY&pp=ygUWYnVpbGQgbGxtIGZyb20gc2NyYXRjaA%3D%3D)
- [tokenization](https://www.youtube.com/watch?v=zduSFxRajkE)
- [gpt 2 from scratch](https://m.youtube.com/watch?v=l8pRSuU81PU)
- [let's reproduce gpt-2](https://www.youtube.com/watch?v=l8pRSuU81PU)
- [find tuning llama](https://www.llama.com/docs/how-to-guides/fine-tuning/)
- [llama2.c](https://github.com/karpathy/llama2.c/blob/master/run.c)

# Future plans
- Voice output
- Voice input
