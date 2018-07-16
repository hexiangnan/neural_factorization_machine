# Neural Factorization Machines

This is our implementation for the paper:

Xiangnan He and Tat-Seng Chua (2017). [Neural Factorization Machines for Sparse Predictive Analytics.](http://www.comp.nus.edu.sg/~xiangnan/papers/sigir17-nfm.pdf) In Proceedings of SIGIR '17, Shinjuku, Tokyo,
Japan, August 07-11, 2017.

We have additionally released our TensorFlow implementation of Factorization Machines under our proposed neural network framework. 

**Please cite our SIGIR'17 paper if you use our codes. Thanks!** 

Author: Dr. Xiangnan He (http://www.comp.nus.edu.sg/~xiangnan/)

## Example to run the codes.

```
python NeuralFM.py --dataset frappe --hidden_factor 64 --layers [64] --keep_prob [0.8,0.5] --loss_type square_loss --activation relu --pretrain 0 --optimizer AdagradOptimizer --lr 0.05 --batch_norm 1 --verbose 1 --early_stop 1 --epoch 200
```
The instruction of commands has been clearly stated in the codes (see the  parse_args function). 

The current implementation supports two tasks: regression and binary classification. The regression task optimizes RMSE, and the binary classification task optimizes Log Loss. 

### Dataset
We use the same input format as the LibFM toolkit (http://www.libfm.org/). 

Split the data to train/test/validation files to run the codes directly (examples see data/frappe/). 



Last Update Date: May 11, 2017
