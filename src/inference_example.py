from transformers import pipeline

pipe = pipeline("sentiment-analysis", model="../results")

print(pipe("This movie was awesome!"))
