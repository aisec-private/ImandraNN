# Network compression

Network compression consists in reducing a network's size/inference computation time without
losing accuracy (or whichever measure we chose to maintain)

Compression techniques include :
* quantisation: convert network's weights into a smaller data type, e.g. from f32 to i8
* pruning: removing weights or nodes from the network 

In our case, compression is necessary due to limited scaling of our solution. 
For ACAS Xu, we do not have access to the training/testing data so we cannot evaluate the compressed networks's accuracy,
but we use proven compression technique. If we had access to such data, it would be easy to fine-tune 
the compression in order to have a verifiable network without losing too much accuracy.

quantisation: we use onnx's static quantization technique. For this technique to work, we need 
training data, so we devise a random data generator with "credible" values coming from the problem
domain. 

pruning: iterative pruning is impossible since we do not have training data on which to re-evaluate the 
pruned network. We use the simplest criterion to chose which weights to remove: weights magnitude. 
