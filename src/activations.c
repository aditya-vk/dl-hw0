#include <assert.h>
#include <math.h>
#include "uwnet.h"

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
            double x = m.data[i*m.cols + j];
            if(a == LOGISTIC){
                x = 1.0/(1 + exp(-x));
            } else if (a == RELU){
                x = ((x < 0.0) ? 0.0 : x);
            } else if (a == LRELU){
                x = ((x < 0.0) ? 0.1*x : x);
            } else if (a == SOFTMAX){
                x = exp(x);
            }
            m.data[i*m.cols + j] = x;
            sum += m.data[i*m.cols + j];
        }
        if (a == SOFTMAX) {
            for(j = 0; j < m.cols; ++j){
                m.data[i*m.cols + j] /= sum;
            }
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
            double x = m.data[i*m.cols + j];
            if (a == LOGISTIC)
                d.data[i * d.cols + j] *= (1-x) * x;
            if (a == RELU)
                d.data[i * d.cols + j] *= ((x < 0.0) ? 0 : 1);
            if (a == LRELU)
                d.data[i * d.cols + j] *= ((x < 0.0) ? 0.1 : 1);
        }
    }
}
