#include <math.h>
#include <stdlib.h>
#include "image.h"
#include "matrix.h"

// Run an activation function on each element in a matrix,
// modifies the matrix in place
// matrix m: Input to activation function
// ACTIVATION a: function to run
void activate_matrix(matrix m, ACTIVATION a)
{
    int i, j;
    for(i = 0; i < m.rows; ++i){
        double sum = 0;
        for(j = 0; j < m.cols; ++j){
            double x = m.data[i][j];
            if(a == LOGISTIC){
                // TODO
				m.data[i][j] = 1.0/(1+exp(-x));
            } else if (a == RELU){
                // TODO
				m.data[i][j] = x>0?x :0;
            } else if (a == LRELU){
                // TODO
				m.data[i][j] = x>0?x:0.1*x;
            } else if (a == SOFTMAX){
                // TODO
				m.data[i][j] = exp(x);
            }
            sum += m.data[i][j];
        }
        if (a == SOFTMAX) {
            // TODO: have to normalize by sum if we are using SOFTMAX
			for(j=0;j<m.cols; ++j)
				m.data[i][j]/=sum;
        }
    }
}

// Calculates the gradient of an activation function and multiplies it into
// the delta for a layer
// matrix m: an activated layer output
// ACTIVATION a: activation function for a layer
// matrix d: delta before activation gradient
void gradient_matrix(matrix m, ACTIVATION a, matrix d)
{
    int i, j;
    for(i = 0; i < m.rows; ++i){
        for(j = 0; j < m.cols; ++j){
            double x = m.data[i][j];
            // TODO: multiply the correct element of d by the gradient
			double l = d.data[i][j];
			if(a==LOGISTIC){
				d.data[i][j] = x*(1-x)*l;
			}
			else if (a==RELU){
				d.data[i][j] = x>0?l:0;
			}
			else if (a==LRELU){
				d.data[i][j] = x>0?l:0.1*l;
			}
			else if (a==SOFTMAX){
				d.data[i][j] = l;
			}
        }
    }
	//printf("%i,%i,%i,%i\n", m.rows, m.cols, d.rows, d.cols);
	//d=matrix_mult_matrix(d,m);
}

// Forward propagate information through a layer
// layer *l: pointer to the layer
// matrix in: input to layer
// returns: matrix that is output of the layer
matrix forward_layer(layer *l, matrix in)
{

    l->in = in;  // Save the input for backpropagation


    // TODO: fix this! multiply input by weights and apply activation function.
    matrix out = make_matrix(in.rows, l->w.cols);
	out = matrix_mult_matrix(in,l->w);
	//activate_matrix(out, l->activation);

    free_matrix(l->out);// free the old output
    l->out = out;       // Save the current output for gradient calculation
	activate_matrix(out, l->activation);
    return out;
}

// Backward propagate derivatives through a layer
// layer *l: pointer to the layer
// matrix delta: partial derivative of loss w.r.t. output of layer
// returns: matrix, partial derivative of loss w.r.t. input to layer
matrix backward_layer(layer *l, matrix delta)
{
    // 1.4.1
    // delta is dL/dy
    // TODO: modify it in place to be dL/d(xw)
	gradient_matrix(l->out,l->activation,delta); 

    // 1.4.2
    // TODO: then calculate dL/dw and save it in l->dw
    free_matrix(l->dw);
	matrix dw = matrix_mult_matrix(transpose_matrix(l->in), delta);
    //matrix dw = make_matrix(l->w.rows, l->w.cols); // replace this
    l->dw = dw;

    
    // 1.4.3
    // TODO: finally, calculate dL/dx and return it.
	matrix dx =matrix_mult_matrix(delta, transpose_matrix(l->w));
    //matrix dx = make_matrix(l->in.rows, l->in.cols); // replace this

    return dx;
}

// Update the weights at layer l
// layer *l: pointer to the layer
// double rate: learning rate
// double momentum: amount of momentum to use
// double decay: value for weight decay
void update_layer(layer *l, double rate, double momentum, double decay)
{
    // TODO:
    // Calculate Δw_t = dL/dw_t - λw_t + mΔw_{t-1}
    // save it to l->v
	l->v = axpy_matrix(momentum, l->v, axpy_matrix(-decay, l->w, l->dw));
	//print_matrix(dw_t);
	//free_matrix(l->v);
	//l->v = dw_t;
	//print_matrix(l->v);

    // Update l->w
	l->w = axpy_matrix(rate, l->v, l->w);
	//free_matrix(l->v);
	//print_matrix(dw_t);
	//l->v = axpy_matrix(rate-1.0, dw_t,dw_t);
	//print_matrix(l->v);
	//free_matrix(dw_t);
    // Remember to free any intermediate results to avoid memory leaks

}

// Make a new layer for our model
// int input: number of inputs to the layer
// int output: number of outputs from the layer
// ACTIVATION activation: the activation function to use
layer make_layer(int input, int output, ACTIVATION activation)
{
    layer l;
    l.in  = make_matrix(1,1);
    l.out = make_matrix(1,1);
    l.w   = random_matrix(input, output, sqrt(2./input));
    l.v   = make_matrix(input, output);
    l.dw  = make_matrix(input, output);
    l.activation = activation;
    return l;
}

// Run a model on input X
// model m: model to run
// matrix X: input to model
// returns: result matrix
matrix forward_model(model m, matrix X)
{
    int i;
    for(i = 0; i < m.n; ++i){
        X = forward_layer(m.layers + i, X);
    }
    return X;
}

// Run a model backward given gradient dL
// model m: model to run
// matrix dL: partial derivative of loss w.r.t. model output dL/dy
void backward_model(model m, matrix dL)
{
    matrix d = copy_matrix(dL);
    int i;
    for(i = m.n-1; i >= 0; --i){
        matrix prev = backward_layer(m.layers + i, d);
        free_matrix(d);
        d = prev;
    }
    free_matrix(d);
}

// Update the model weights
// model m: model to update
// double rate: learning rate
// double momentum: amount of momentum to use
// double decay: value for weight decay
void update_model(model m, double rate, double momentum, double decay)
{
    int i;
    for(i = 0; i < m.n; ++i){
        update_layer(m.layers + i, rate, momentum, decay);
    }
}

// Find the index of the maximum element in an array
// double *a: array
// int n: size of a, |a|
// returns: index of maximum element
int max_index(double *a, int n)
{
    if(n <= 0) return -1;
    int i;
    int max_i = 0;
    double max = a[0];
    for (i = 1; i < n; ++i) {
        if (a[i] > max){
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
}

// Calculate the accuracy of a model on some data d
// model m: model to run
// data d: data to run on
// returns: accuracy, number correct / total
double accuracy_model(model m, data d)
{
    matrix p = forward_model(m, d.X);
    int i;
    int correct = 0;
    for(i = 0; i < d.y.rows; ++i){
        if(max_index(d.y.data[i], d.y.cols) == max_index(p.data[i], p.cols)) ++correct;
    }
    return (double)correct / d.y.rows;
}

// Calculate the cross-entropy loss for a set of predictions
// matrix y: the correct values
// matrix p: the predictions
// returns: average cross-entropy loss over data points, 1/n Σ(-ylog(p))
double cross_entropy_loss(matrix y, matrix p)
{
    int i, j;
    double sum = 0;
    for(i = 0; i < y.rows; ++i){
        for(j = 0; j < y.cols; ++j){
            sum += -y.data[i][j]*log(p.data[i][j]);
        }
    }
    return sum/y.rows;
}


// Train a model on a dataset using SGD
// model m: model to train
// data d: dataset to train on
// int batch: batch size for SGD
// int iters: number of iterations of SGD to run (i.e. how many batches)
// double rate: learning rate
// double momentum: momentum
// double decay: weight decay
void train_model(model m, data d, int batch, int iters, double rate, double momentum, double decay)
{
    int e;
    for(e = 0; e < iters; ++e){
        data b = random_batch(d, batch);
        matrix p = forward_model(m, b.X);
        fprintf(stderr, "%06d: Loss: %f\n", e, cross_entropy_loss(b.y, p));
        matrix dL = axpy_matrix(-1, p, b.y); // partial derivative of loss dL/dy
        backward_model(m, dL);
        update_model(m, rate/batch, momentum, decay);
        free_matrix(dL);
        free_data(b);
    }
}


// Questions 
//
// 5.2.2.1 Why might we be interested in both training accuracy and testing accuracy? What do these two numbers tell us about our current model?
// TODO
// Because accuracy inflects prediction ability of a model. If both training and testing accuracy are low, means the model can't predict well, instead, if training accuracy is high and test accuracy is low, means overfittng.
//
// 5.2.2.2 Try varying the model parameter for learning rate to different powers of 10 (i.e. 10^1, 10^0, 10^-1, 10^-2, 10^-3) and training the model. What patterns do you see and how does the choice of learning rate affect both the loss during training and the final model accuracy?
// TODO
// When learning rate is big(10), the model can't converge, loss is -nan and accuracy is 0.09. When lr is smaller(1), model converged quickly but accuracy is not high enough, 0.88, when lr is 0.1, loss is smaller and accuracy is higher, 0.916. But when lr is too small, 0.001, the model would get stuck in local optima, accuracy is not high enough, 0.87
//
// 5.2.2.3 Try varying the parameter for weight decay to different powers of 10: (10^0, 10^-1, 10^-2, 10^-3, 10^-4, 10^-5). How does weight decay affect the final model training and test accuracy?
// TODO
// When weight decay is 1, model accuracy is 0.89, and when weight decay is 0.1,0.01,0.001, the accuracy is about 0.915. Weight decay penalize big weight to prevent overfitting, but too big weight decay can prevent model from local optima.
//
// 5.2.3.1 Currently the model uses a logistic activation for the first layer. Try using a the different activation functions we programmed. How well do they perform? What's best?
// TODO
// After 1000 iters, logistic final accuracy is 0.939, relu : 0.956, lrelu :0.953. Relu performs best.
//
// 5.2.3.2 Using the same activation, find the best (power of 10) learning rate for your model. What is the training accuracy and testing accuracy?
// TODO
// relu, lr = 0.01, iteration = 3000, training accuracy = 0.950, test accuracy = 0.946
//
// 5.2.3.3 Right now the regularization parameter `decay` is set to 0. Try adding some decay to your model. What happens, does it help? Why or why not may this be?
// TODO
// I set decay = 0.001, the final accuracy is 0.957, 0.953. Weight decay penalize big weight and prevent overfitting, so it can improve test accuracy.
//
// 5.2.3.4 Modify your model so it has 3 layers instead of two. The layers should be `inputs -> 64`, `64 -> 32`, and `32 -> outputs`. Also modify your model to train for 3000 iterations instead of 1000. Look at the training and testing error for different values of decay (powers of 10, 10^-4 -> 10^0). Which is best? Why?
// TODO
// 0.001 is best. Weight decay can prevent overfitting and make mode learn the good pattern of data. So I think it is like find a balaned point for weight decay, we should try different value to figure out the best. 
//
// 5.3.2.1 How well does your network perform on the CIFAR dataset?
// TODO
// I added more layers, and the training process is very slow. After 2000 iterations, I achieved only 0.531 accuracy. 



