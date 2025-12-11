# NLP-Final-Project

Instructions to run:

Train the extractive model:
1) pip install -r requirements.txt
2) python src/train_extractive.py configs/extractive_config.yaml

Can change spoiler type to phrase, multi or passage through extractive_config.yaml file

Inference it:
3) python src/run_predictor_demo.py 

Evaluate bleu scores:
4) python evaluate_bleu.py --predictions results/predictions.json

Train the refinement model:
5) python src/train_refinement_new.py

Can change spoiler type to phrase, multi or passage inside the train_refinement file itself.