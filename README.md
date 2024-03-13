## SemEval-2023 Task 6 Sub-task 3: Court Judgement Prediction with Explanation (CJPE).

### Setup environment
```
conda create -p .env/ python=3.8
pip install -r requirements.txt
```

Install torch-scatter if using the Hierarchical Longformer model. See: https://github.com/rusty1s/pytorch_scatter.

### Run Longformer TF-IDF

#### Build vocabulary

For ILDC:
```
python build_vocab.py --dataset ildc --frequency-threshold 350
```

#### Train & evaluate model:
```
python train_longformer_tfidf.py \
    --max-length 4096 \
    --model allenai/longformer-large-4096 \
    --save-dir models/longformer-tfidf/longformer-large-seed100 \
    --log-dir results/longformer-tfidf/longformer-large-seed100 \
    --truncation-side left \
    --dataset ildc \
    --seed 100 \
    --epochs 5 \
    --batch-size 8 \
    --gradient-accumulation-steps 8 \
    --tfidf-vectorizer tfidf_vectorizer-threshold350.pkl
```
If the model doesn't fit the GPU memory, add --fp16 to reduce the model size.

### Run pure text classification using Pretrained Language model
```
python train_text_classification.py \
    --max-length 512 \
    --model roberta-large \
    --save-dir models/text-classification/roberta-large \
    --log-dir results/text-classification/roberta-large \
    --truncation-side left \
    --dataset ildc \
    --seed 100 \
    --epochs 5 \
    --batch-size 8 \
    --gradient-accumulation-steps 8
```

### Train hierarchical model
```
python train_hierarchical_model.py \
    --max-length 4096 \
    --model-path allenai/longformer-large-4096 \
    --save-dir models/hierarchical/longformer-large \
    --log-dir results/hierarchical/longformer-large \
    --truncation-side left \
    --dataset ildc \
    --seed 100 \
    --epochs 5 \
    --max-n-chunks 3 \
    --batch-size 2 \
    --gradient-accumulation-steps 32 
```

### Dataset
```
The dataset was taken from ILDC :- https://github.com/Exploration-Lab/CJPE.
We requested for the database of Indian Legal Document Corpus from ashutoshm.iitk@gmail.com, vijitvm21@gmail.com. Through this database we ran the base model and obtained the results
```
