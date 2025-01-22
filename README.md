# Usage
```
$ python wikibot.py
```
Write your question. For example:
```
Hello I am wikibot! Ask me anything.
> What does the fox say?
```
## Available commands
```
$ python wikibot.py [model.pkl]     # Start wikibot using wikibot.pkl or [model.pkl]
$ python wikibot.py train           # Train wikibot and store it in ./wikibot.pkl
$ python wikibot.py train tokenizer # Train the tokenizer and store it in tokenizer.pkl
```

# Goals
- Take less space than the dataset (14 gb)
- Correct responses

# Hmmm
- How to teach the wikibot about itself
- How to handle the large dataset
- How to make it understand the situation it's in so it gives good responses

# Resources
- [wikipedia dataset](https://github.com/GermanT5/wikipedia2corpus)
- [llm from scratch](https://m.youtube.com/watch?v=kCc8FmEb1nY&pp=ygUWYnVpbGQgbGxtIGZyb20gc2NyYXRjaA%3D%3D)
- [tokenization](https://www.youtube.com/watch?v=zduSFxRajkE)
- [gpt 2 from scratch](https://m.youtube.com/watch?v=l8pRSuU81PU)

# Future plans
- Voice output
- Voice input
