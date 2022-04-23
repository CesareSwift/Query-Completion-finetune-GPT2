This project used [gpt2.ipynb](https://github.com/CesareSwift/Query-Complete-finetune-GPT2/blob/main/gpt2.ipynb "gpt2.ipynb") to finetune gpt2 with the dataset you want (in this sample, I used a csv file which contained 200MB news data to finetune gpt2). 

After the finetuning, the model was saved using pytorch, then used the predict.py to load the model and generated a whole sentence based on the input sequences (which are usually 2-3 words) to finish the query completion.


