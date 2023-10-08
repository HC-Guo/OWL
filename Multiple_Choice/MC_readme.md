# Introduction for Multiple-Choice in Owl-Bench


# Multiple Choice Structure
  - data: The data folder is used to save the evaluation questions.
  - model: The model receives the input and gives the corresponding answer.
  - result: This folder contains the model's answers to each question and whether they were correct or not.
  - utils: Code for Data preprocessing.
  - eval.py
  - requirements.txt
  
## Eval_model
## Run
Please run the eval.py.

```
python3 eval.py

```
```
Main  parameters:
         main(path='/path/to/your/repo'
         ntrain=0,
         api_key="xxx",  # if model is chatgpt ,this paarameter is necessary.
         cot=False,
         temperature=0.2,
         model_name="llama_13b",  #["chatgpt","moss","chatglm6b","chatglm2_6b","llama_13b"]
         model_path="/path/to/models",
         cuda_visible_device="4,5,6,7")
```


  

 