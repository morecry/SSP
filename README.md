#Repository of the paper "SSP: Self-Supervised Post training for Conversational Search" (ACL 2023)
## Prerequisites

Install dependencies:

```bash
pip install -r requirements.txt
```
 We use [ANCE](https://github.com/microsoft/ANCE) as a backbone. So you should first download the checkpoint and store them in `./checkpoints`.

```bash
mkdir checkpoints
wget https://webdatamltrainingdiag842.blob.core.windows.net/semistructstore/OpenSource/Passage_ANCE_FirstP_Checkpoint.zip
unzip Passage_ANCE_FirstP_Checkpoint.zip
mv "Passage ANCE(FirstP) Checkpoint" ad-hoc-ance-msmarco
```

## Data Preparation

Please prepare the CAsT-19 and CAsT-20 dataset, then preprocess them and generate the candidate documents embeddding as follows. (Same as [ConvDR](https://github.com/thunlp/ConvDR).)


### TREC CAsT

#### CAsT shared files download

Use the following commands to download the document collection for CAsT-19 & CAsT-20 as well as the MARCO duplicate file:

```bash
cd datasets/raw
wget https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz -O msmarco.tsv
wget http://trec-car.cs.unh.edu/datareleases/v2.0/paragraphCorpus.v2.0.tar.xz
wget http://boston.lti.cs.cmu.edu/Services/treccast19/duplicate_list_v1.0.txt
```

#### CAsT-19 files download

Download necessary files for CAsT-19 and store them into `./datasets/raw/cast-19`:

```bash
mkdir datasets/raw/cast-19
cd datasets/raw/cast-19
wget https://raw.githubusercontent.com/daltonj/treccastweb/master/2019/data/evaluation/evaluation_topics_v1.0.json
wget https://raw.githubusercontent.com/daltonj/treccastweb/master/2019/data/evaluation/evaluation_topics_annotated_resolved_v1.0.tsv
wget https://trec.nist.gov/data/cast/2019qrels.txt
```

#### CAsT-20 files download

Download necessary files for CAsT-20 and store them into `./datasets/raw/cast-20`:

```bash
mkdir datasets/raw/cast-20
cd datasets/raw/cast-20
wget https://raw.githubusercontent.com/daltonj/treccastweb/master/2020/2020_automatic_evaluation_topics_v1.0.json
wget https://raw.githubusercontent.com/daltonj/treccastweb/master/2020/2020_manual_evaluation_topics_v1.0.json
wget https://trec.nist.gov/data/cast/2020qrels.txt
```

#### CAsT preprocessing

Use the scripts `./data/preprocess_cast19` and `./data/preprocess_cast20` to preprocess raw CAsT files:

```bash
mkdir datasets/cast-19
mkdir datasets/cast-shared
python data/preprocess_cast19.py  --car_cbor=datasets/raw/dedup.articles-paragraphs.cbor  --msmarco_collection=datasets/raw/msmarco.tsv  --duplicate_file=datasets/raw/duplicate_list_v1.0.txt  --cast_dir=datasets/raw/cast-19/  --out_data_dir=datasets/cast-19  --out_collection_dir=datasets/cast-shared
```

```bash
mkdir datasets/cast-20
mkdir datasets/cast-shared
python data/preprocess_cast20.py  --car_cbor=datasets/raw/dedup.articles-paragraphs.cbor  --msmarco_collection=datasets/raw/msmarco.tsv  --duplicate_file=datasets/raw/duplicate_list_v1.0.txt  --cast_dir=datasets/raw/cast-20/  --out_data_dir=datasets/cast-20  --out_collection_dir=datasets/cast-shared
```

### Generate Document Embeddings

Our code is based on ANCE and we have a similar embedding inference pipeline, where the documents are first tokenized and converted to token ids and then the token ids are used for embedding inference. We create sub-directories `tokenized` and `embeddings` inside `./datasets/cast-shared` and `./datasets/or-quac` to store the tokenized documents and document embeddings, respectively:

```bash
mkdir datasets/cast-shared/tokenized
mkdir datasets/cast-shared/embeddings
mkdir datasets/or-quac/tokenized
mkdir datasets/or-quac/embeddings
```

Run `./data/tokenizing.py` to tokenize documents in parallel:

```bash
# CAsT
python data/tokenizing.py  --collection=datasets/cast-shared/collection.tsv  --out_data_dir=datasets/cast-shared/tokenized  --model_name_or_path=checkpoints/ad-hoc-ance-msmarco --model_type=rdot_nll
# OR-QuAC
python data/tokenizing.py  --collection=datasets/or-quac/collection.tsv  --out_data_dir=datasets/or-quac/tokenized  --model_name_or_path=bert-base-uncased --model_type=dpr
```

After tokenization, run `./drivers/gen_passage_embeddings.py` to generate document embeddings:

```bash
# CAsT
python -m torch.distributed.launch --nproc_per_node=$gpu_no python drivers/gen_passage_embeddings.py  --data_dir=datasets/cast-shared/tokenized  --checkpoint=checkpoints/ad-hoc-ance-msmarco  --output_dir=datasets/cast-shared/embeddings  --model_type=rdot_nll
# OR-QuAC
python -m torch.distributed.launch --nproc_per_node=$gpu_no python drivers/gen_passage_embeddings.py  --data_dir=datasets/or-quac/tokenized  --checkpoint=checkpoints/ad-hoc-ance-orquac.cp  --output_dir=datasets/or-quac/embeddings  --model_type=dpr
```

Note that we follow the ANCE implementation and this step takes up a lot of memory. To generate all 38M CAsT document embeddings safely, the machine should have at least 200GB memory. It's possible to save memory by generating a part at a time, and we may update the implementation in the future.


### QReCC Dataset
We use [QReCC](https://github.com/apple/ml-qrecc/blob/main/dataset/qrecc_data.zip) dataset for SSP and warm-up the conversational retriever.  

```bash
# download dataset
cd QReCC/
wget https://github.com/apple/ml-qrecc/blob/main/dataset/qrecc_data.zip
unzip qrecc_data.zip
mv qrecc_data/* .
rm -r qrecc_data/
cd ..

# preprocess 
python data_process.py

cd ..

# create topic indicator
python create_topic_shift.py

# creater reference marker
python mark_key_sent.py
```


## SSP on CAsT-19

```bash
# ssp warm-up
python drivers/run_convdr_train.py  --output_dir=checkpoints/ConvDR-KD-QRECC-postrain-19_1  --model_name_or_path=checkpoints/ad-hoc-ance-msmarco --teacher_model=checkpoints/ad-hoc-ance-msmarco --train_file=QRECC/query_refomulation_aug_19.json  --query=no_res  --per_gpu_train_batch_size=64  --learning_rate=2e-5   --log_dir=logs/convdr_kd_cast19  --num_train_epochs=2 --model_type=rdot_nll  --overwrite_output_dir --max_concat_length=256 --max_query_length=32 --use_debias --use_mark --use_topic
python drivers/run_convdr_train.py  --output_dir=checkpoints/ConvDR-KD-QRECC-postrain-19_1  --model_name_or_path=checkpoints/ConvDR-KD-QRECC-postrain-19_1 --teacher_model=checkpoints/ad-hoc-ance-msmarco --train_file=QRECC/query_refomulation.json  --query=no_res  --per_gpu_train_batch_size=64  --learning_rate=2e-5   --log_dir=logs/convdr_kd_cast19  --num_train_epochs=2 --model_type=rdot_nll  --overwrite_output_dir --max_concat_length=256 --max_query_length=32

# cast finetune
python drivers/run_convdr_train.py  --output_dir=checkpoints/convdr-ssp-cast19  --model_name_or_path=checkpoints/ConvDR-KD-QRECC-postrain-19_1  --teacher_model=checkpoints/ad-hoc-ance-msmarco --train_file=datasets/cast-19/eval_topics.jsonl  --query=no_res  --per_gpu_train_batch_size=4  --learning_rate=1e-6  --log_dir=logs/convdr_kd_cast19  --num_train_epochs=5  --model_type=rdot_nll --max_concat_length=256 --max_query_length=32 --overwrite_output_dir --cross_validate --warmup_steps=100

# inference 
python drivers/run_convdr_inference.py  --model_path=checkpoints/convdr-ssp-cast19  --eval_file=datasets/cast-19/eval_topics.jsonl  --query=no_res  --per_gpu_eval_batch_size=8  --cache_dir=../ann_cache_dir  --ann_data_dir=datasets/cast-shared/embeddings  --qrels=datasets/cast-19/qrels.tsv  --processed_data_dir=datasets/cast-shared  --raw_data_dir=datasets/cast-19   --output_dir=results/cast-19 --model_type=rdot_nll  --output_query_type=raw  --run_tag=ssp --use_gpu --cross_validate --max_concat_length=256 --max_query_length=32

# evaluate
python trec_eval.py --qrel datasets/cast-19/qrels.tsv --run results/cast-19/ssp.trec --tag cast19
```

## SSP on CAsT-20

```bash
# ssp warm-up
python drivers/run_convdr_train.py  --output_dir=checkpoints/ConvDR-KD-QRECC-postrain-20_1  --model_name_or_path=checkpoints/ad-hoc-ance-msmarco --teacher_model=checkpoints/ad-hoc-ance-msmarco --train_file=QRECC/query_refomulation_aug_20.json  --query=auto_can  --per_gpu_train_batch_size=32  --learning_rate=2e-5   --log_dir=logs/convdr_kd_cast20  --num_train_epochs=1 --model_type=rdot_nll  --overwrite_output_dir --max_concat_length=512 --max_query_length=32 --use_mark --use_topic --use_debias --no_mse
python drivers/run_convdr_train.py  --output_dir=checkpoints/ConvDR-KD-QRECC-postrain-20_1  --model_name_or_path=checkpoints/ConvDR-KD-QRECC-postrain-20_1 --teacher_model=checkpoints/ad-hoc-ance-msmarco --train_file=QRECC/query_refomulation.json  --query=auto_can  --per_gpu_train_batch_size=32  --learning_rate=2e-5   --log_dir=logs/convdr_kd_cast20  --num_train_epochs=1 --model_type=rdot_nll  --overwrite_output_dir --max_concat_length=512 --max_query_length=32 --no_mse

# cast finetune
python drivers/run_convdr_train.py  --output_dir=checkpoints/convdr-ssp-cast20  --model_name_or_path=checkpoints/ConvDR-KD-QRECC-postrain-20_1 --teacher_model=checkpoints/ad-hoc-ance-msmarco --train_file=datasets/cast-20/eval_topics.jsonl  --query=auto_can  --per_gpu_train_batch_size=4  --learning_rate=2e-5   --log_dir=logs/convdr_kd_cast20  --num_train_epochs=5  --model_type=rdot_nll  --cross_validate  --max_concat_length=512  --max_query_length=32 

# inference 
python drivers/run_convdr_inference.py  --model_path=checkpoints/convdr-ssp-cast20  --eval_file=datasets/cast-20/eval_topics.jsonl  --query=auto_can  --per_gpu_eval_batch_size=8  --cache_dir=../ann_cache_dir  --ann_data_dir=datasets/cast-shared/embeddings  --qrels=datasets/cast-20/qrels.tsv  --processed_data_dir=datasets/cast-shared --raw_data_dir=datasets/cast-20 --output_dir=results/cast-20 --model_type=rdot_nll  --output_query_type=raw  --use_gpu  --cross_validate --max_concat_length=512 --run_tag=ssp 

# evaluate
python trec_eval.py --qrel datasets/cast-20/qrels.tsv --run results/cast-20/ssp.trec --tag cast20
```


<!-- ## Download Trained Models

Three trained models can be downloaded with the following link: [CAsT19-KD-CV-Fold1](https://data.thunlp.org/convdr/convdr-kd-cast19-1.zip), [CAsT20-KD-Warmup-CV-Fold2](https://data.thunlp.org/convdr/convdr-kd-cast20-2.zip) and [ORQUAC-Multi](https://data.thunlp.org/convdr/convdr-multi-orquac.cp). -->

<!-- ## Results

[Download ConvDR and baseline runs on CAsT](https://drive.google.com/file/d/1F0RwA9sZscUAyE0IyQ7PMrgzNVqDnho5/view?usp=sharing) -->

