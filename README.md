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

FB15k237:
```
python run.py  --do_train --do_test -save ../experiments/kge_baselines_fb237 --data_path ../data/FB15K237  --model MDE  -n 1000 -b 1000 -d 200 -g 4.0 -a 2.5 -adv -lr .0005 --max_steps 300000 --test_batch_size 2 --valid_steps 10000 --log_steps 10000 --do_valid  -node_feat_path ../data/FB15K237/train_node_features --cuda -psi 15.0
```
biokg:

```
python run_with5_gpu.py --init_checkpoint ../experiments/kge_baselines_biokg_400_600_850_2 --do_train --do_test -save ../experiments/kge_baselines_biokg_400_600_850 --data_path ../data/biokg  --model MDE  -n 850 -b 600 -d 400 -g 2.5 -a 2.5 -adv -lr .0005 --max_steps 700000 --test_batch_size 2 --valid_steps 10000 --log_steps 10000 --do_valid  -node_feat_path ../data/biokg/train_node_features --cuda -psi 14.0
```

### **Citation** :


If you use the model, please cite the following paper:
```

@inproceedings{gfa2021ECML,
  title={Embedding Knowledge Graphs Attentive to Positional and Centrality Qualities},
  author={Sadeghi, Afshin and Collarana, Diego and  Graux, Damien and Lehmann, Jens},
  booktitle={European Conference on Machine Learning and Data Mining, ECML PKDD 2021},
  year={2021}
}


```

Link to the paper on the [ECML conference website is here](https://2021.ecmlpkdd.org/wp-content/uploads/2021/07/sub_1096.pdf). 

### FAQ 
<strong>Q</strong>: How we reproduce the results of the model for the large dataset?

<strong>A</strong>: Large datasets similar to biokg require a large number of iterations.  Since the learning rate reduces during the training we do not suggest setting max_steps to a larger number, instead, we suggest storing the trained model using -save and rerunning the training iteration several times. In our evaluation it executed the training 3 times for biokg. 

<strong>Q</strong>: Is the model open for learning furthur features? 
<strong>A</strong> Yes, simply by adding another score and a set of embedding weights to it. Please do not forget to normalize the graph features before learning them.


