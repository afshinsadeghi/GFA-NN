# GFA-NN
## Embedding Knowledge Graphs Attentive to Positional and Centrality Qualities


### Setup

##### 1.  Make sure the Datasets are existent in "./data"

##### 2. Before executing the training, generate pre-preocessed files by
   running: ./extract_graph_features/process.sh

##### 3. With the current limitation on GPU memories, we made a multi core
   version (5 gpu cores) to allow running the model on the larger
   dataset (for example biokg). This version of the model is in the
   folder: ./5_gpu_version_of_model_for_large_datasets

##### 4. Please check the paper Appendix for the best hyper-parameters.

### Example run:

## Incremental training:
WN18RR_INC

run for the first dataset:
```
python run_incremental.py  --do_train --do_test -save ./experiments/kge_baselines_wn18rr_inc --data_path ./data/WN18RR_inc --data_path_train data/WN18RR_inc/train1.txt -data_path_entities data/WN18RR_inc/entity2id.txt -data_path_rels data/WN18RR_inc/relation2id.txt --model MDE  -n 500 -b 1000 -d 200 -g 4.0 -a 2.5 -adv -lr .0005 --max_steps 10000 --test_batch_size 2 --valid_steps 10000 --log_steps 10000 --do_valid  -node_feat_path ./data/WN18RR_inc/train_node_features --cuda -psi 14.0
```
run for the next incoming datasets:
for examplle train2.txt and new parameter:  -adding_data
```
python run_incremental.py --init_checkpoint -adding_data  --do_train --do_test -save ./experiments/kge_baselines_wn18rr_inc2 --data_path ./data/WN18RR_inc --data_path_train data/WN18RR_inc/train2.txt --model MDE  -n 500 -b 1000 -d 200 -g 4.0 -a 2.5 -adv -lr .0005 --max_steps 10000 --test_batch_size 2 --valid_steps 10000 --log_steps 10000 --do_valid  -node_feat_path ./data/WN18RR_inc/train_node_features --cuda -psi 14.0
```

## regular training:
WN18RR

```
python run.py  --do_train --do_test -save ./experiments/kge_baselines_wn18rr_inc --data_path ./data/WN18RR_inc  --model MDE  -n 500 -b 1000 -d 200 -g 4.0 -a 2.5 -adv -lr .0005 --max_steps 300000 --test_batch_size 2 --valid_steps 10000 --log_steps 10000 --do_valid  -node_feat_path ./data/WN18RR_inc/train_node_features --cuda -psi 14.0
```

FB15k237:
```
python run.py  --do_train --do_test -save ../experiments/kge_baselines_fb237 --data_path ../data/FB15K237  --model MDE  -n 1000 -b 1000 -d 200 -g 4.0 -a 2.5 -adv -lr .0005 --max_steps 300000 --test_batch_size 2 --valid_steps 10000 --log_steps 10000 --do_valid  -node_feat_path ../data/FB15K237/train_node_features --cuda -psi 15.0
```
biokg:

```
python run_with5_gpu.py --init_checkpoint ../experiments/kge_baselines_biokg_400_600_850_2 --do_train --do_test -save ../experiments/kge_baselines_biokg_400_600_850 --data_path ../data/biokg  --model MDE  -n 850 -b 600 -d 400 -g 2.5 -a 2.5 -adv -lr .0005 --max_steps 700000 --test_batch_size 2 --valid_steps 10000 --log_steps 10000 --do_valid  -node_feat_path ../data/biokg/train_node_features --cuda -psi 14.0
```


## Setup for incremental training:

#assume that data cleaning and dubplicates of entities and relations are done.
# all the new triples have at least a head or a tail in the known trained entities.


#for the first run:
#just train on the first coming set.
#save the trained model

#to make it you can just run for 5 epochs.

#incremental training iteration:
# load the saved model with its data put it in m1
# new data in new data folder arrives-> generate graph features and dictionary files entities.dic relations.dic 
#the names of coming files will be like train1.txt train2.txt etc
# read the new data:
# load them seperatedyly or putting them in one bigger train.txt file?
# in this step entity matching must be done, if they are same dedblicate and label the new ones with old ones.
#then make a larger dataset, still, only train on newly come entities? or their beghbours too? or all the network? 
#research question: 
# to what level of neighbours entites must be train?
# can these subgraphs be trained seperatedly? 

# make a new model name it m2 with size of m1 plus new entities in m2
# copy m1 into m2
# train on new triple t2 + triples of old dataset that have a common entity or relation with t2
# save new model and aggregated triples.  

### Dataset Generation for experimetns:
for experiments: run create_inc_dataset.py that randomly select triples from train.txt and generates several incoming train files as train1.txt ,train2.txt , ...
to run: python  create_inc_dataset.py  -data_path data/WN18RR_inc -divisions 5

# GFA-NN model training:

1. step extract features:
./extract_graph_features/process.sh


2. step run the embedding:
python run_incremental.py  --do_train --do_test -save ./experiments/kge_baselines_wn18rr_inc --data_path ./data/WN18RR_inc --train_file train1.txt  --model MDE  -n 500 -b 1000 -d 200 -g 4.0 -a 2.5 -adv -lr .0005 --max_steps 3000 --test_batch_size 2 --valid_steps 3000 --log_steps 3000 --do_valid  -node_feat_path ./data/WN18RR_inc/train_node_features --cuda -psi 14.0

then second run 


1. external step 1: data integration using entity matching and deduplicates: 
there 4 types of triples must be annoated by a 4th column:
1.new: both head tail are new
2.old
3.neghbour 1 : one of the head and tails are new
4.neghbour hop 2 : one of entites are connected to an entity that is old but neighbour to a new entity 


2. step extract features:
./extract_graph_features/process.sh

3. with --init_checkpoint to load the saved model and load new train_file:

python run_incremental.py  --init_checkpoint  --do_train --do_test -save ./experiments/kge_baselines_wn18rr_inc --data_path ./data/WN18RR_inc --train_file train2.txt  --model MDE  -n 500 -b 1000 -d 200 -g 4.0 -a 2.5 -adv -lr .0005 --max_steps 3000 --test_batch_size 2 --valid_steps 3000 --log_steps 3000 --do_valid  -node_feat_path ./data/WN18RR_inc/train_node_features --cuda -psi 14.0


Link to the paper on the [ECML conference website is here](https://2021.ecmlpkdd.org/wp-content/uploads/2021/07/sub_1096.pdf). 

### FAQ 
<strong>Q</strong>: How we reproduce the results of the model for the large dataset?

<strong>A</strong>: Large datasets similar to biokg require a large number of iterations.  Since the learning rate reduces during the training we do not suggest setting max_steps to a larger number, instead, we suggest storing the trained model using -save and rerunning the training iteration several times. In our evaluation it executed the training 3 times for biokg. 


<strong>Q</strong>: Is the model open for learning furthur features? 

<strong>A</strong>: Yes, simply by adding another score and a set of embedding weights to it. Please do not forget to normalize the graph features before learning them.


