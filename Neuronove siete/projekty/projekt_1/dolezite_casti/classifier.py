# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2017-2024
import numpy as np

from mlp import *
from util import *

# Vypracoval: Marian Kravec

class MLPClassifier(MLP):
    def __init__(self, dim_in, dim_hid1, dim_hid2, dim_out, lab_vals):
        super().__init__(dim_in, dim_hid1, dim_hid2, dim_out, lab_vals)

    # Activation functions & derivations

    def error(self, targets, outputs):
        """
        Cost / loss / error function
        """
        return np.sum((targets - outputs)**2, axis=0)

    # @override
    def f_hid1(self, x):
        return x * (x > 0)

        # @override
    def df_hid1(self, x):
        return 1. * (x > 0)

        # @override
    def f_hid2(self, x):
        return np.tanh(x)

    # @override
    def df_hid2(self, x):
        return 1/np.cosh(x)**2

    # @override
    def f_out(self, x):
        return 1/(1+np.exp(-x))

    # @override
    def df_out(self, x):
        return self.f_out(x)*(1-self.f_out(x))

    def predict(self, inputs):
        """
        Prediction = forward pass
        """
        outputs = np.stack([self.forward(x)[-1] for x in inputs.T]).T
        return outputs, onehot_decode(outputs, self.lab_vals)

    def predictOne(self, x):
        """
        Prediction = forward pass
        """
        return self.forward(x)[-1]


    def train(self, inputs, labels, alpha=0.1, beta_1=0.9, beta_2=0.99, epsy=0.9999, eps=100, verbose=True, comp_test=False, test_inputs=None, test_labels=None):
        (_, count) = inputs.shape
        targets = onehot_encode(labels, self.lab_vals)

        Errors = {"trainCEs": [],
                  "trainREs": [],
                  "testCEs": [],
                  "testREs": []}

        if comp_test:
            t_d = onehot_encode(test_labels, self.lab_vals)
            t_count = test_inputs.shape[1]

        for ep in range(eps):
            trCE = 0
            trRE = 0
            teCE = 0
            teRE = 0

            for idx in np.random.permutation(count):
                x = inputs[:, idx]
                d = targets[:, idx]

                a, h, b, i, c, y = self.forward(x)

                dW_hid1, dW_hid2, dW_out = self.backward(alpha, beta_1, beta_2, ep, x, a, h, b, i, c, y, d, False)

                self.W_hid1 += dW_hid1
                self.W_hid2 += dW_hid2
                self.W_out += dW_out

                # weight decay regularization
                self.W_hid1 *= epsy
                self.W_hid2 *= epsy
                self.W_out *= epsy

                trCE += labels[idx] != onehot_decode(y, self.lab_vals)
                trRE += self.error(d, y)

            trCE /= count
            trRE /= count
            Errors["trainCEs"].append(trCE)
            Errors["trainREs"].append(trRE)

            if comp_test:
                t_y, t_y_c = self.predict(test_inputs)
                teCE = np.sum(test_labels != t_y_c)
                teRE = np.sum(self.error(t_d, t_y))
                teCE /= t_count
                teRE /= t_count
                Errors["testCEs"].append(teCE)
                Errors["testREs"].append(teRE)

            if (ep+1) % 5 == 0 and verbose:
                print('Epoch {:3d}/{}, CE = {:6.2%}, RE = {:.5f}'.format(ep+1, eps, CE, RE))

        return Errors

    def trainBatch(self, inputs, labels, alpha=0.1, beta_1=0.9, beta_2=0.99, epsy=0.999, eps=100, verbose=True, comp_test=False, test_inputs=None, test_labels=None):
        (_, count) = inputs.shape
        targets = onehot_encode(labels, self.lab_vals)

        Errors = {"trainCEs": [],
                  "trainREs": [],
                  "testCEs": [],
                  "testREs": []}

        if comp_test:
            t_d = onehot_encode(test_labels, self.lab_vals)
            t_count = test_inputs.shape[1]

        for ep in range(eps):

            idx = np.random.permutation(count)
            x = inputs[:, idx]
            d = targets[:, idx]

            a, h, b, i, c, y = self.forward(x)

            dW_hid1, dW_hid2, dW_out = self.backward(alpha, beta_1, beta_2, ep, x, a, h, b, i, c, y, d, True)

            self.W_hid1 += dW_hid1
            self.W_hid2 += dW_hid2
            self.W_out += dW_out

            # weight decay regularization
            self.W_hid1 *= epsy
            self.W_hid2 *= epsy
            self.W_out *= epsy

            trCE = np.sum(labels[idx] != onehot_decode(y, self.lab_vals))
            trRE = np.sum(self.error(d, y))
            trCE /= count
            trRE /= count
            Errors["trainCEs"].append(trCE)
            Errors["trainREs"].append(trRE)

            if comp_test:
                t_y, t_y_c = self.predict(test_inputs)
                teCE = np.sum(test_labels != t_y_c)
                teRE = np.sum(self.error(t_d, t_y))
                teCE /= t_count
                teRE /= t_count
                Errors["testCEs"].append(teCE)
                Errors["testREs"].append(teRE)

            if (ep+1) % 100 == 0 and verbose:
                print('Epoch {:3d}/{}, CE = {:6.2%}, RE = {:.5f}'.format(ep+1, eps, CE, RE))

        return Errors

    def trainMiniBatch(self, inputs, labels, alpha=0.1, beta_1=0.9, beta_2=0.99, epsy=0.999, eps=100, batches=10, verbose=True, comp_test=False, test_inputs=None, test_labels=None):
        (_, count) = inputs.shape
        targets = onehot_encode(labels, self.lab_vals)

        Errors = {"trainCEs": [],
                  "trainREs": [],
                  "testCEs": [],
                  "testREs": []}

        if comp_test:
            t_d = onehot_encode(test_labels, self.lab_vals)
            t_count = test_inputs.shape[1]

        batchsize = count//batches
        for ep in range(eps):
            trCE = 0
            trRE = 0
            idx = np.random.permutation(count)

            for j in range(batches):
                x = inputs[:, idx[j*batchsize:(j+1)*batchsize]]
                d = targets[:, idx[j*batchsize:(j+1)*batchsize]]

                a, h, b, i, c, y = self.forward(x)

                dW_hid1, dW_hid2, dW_out = self.backward(alpha, beta_1, beta_2, ep, x, a, h, b, i, c, y, d, True)

                self.W_hid1 += dW_hid1
                self.W_hid2 += dW_hid2
                self.W_out += dW_out

                # weight decay regularization
                self.W_hid1 *= epsy
                self.W_hid2 *= epsy
                self.W_out *= epsy

                trCE += np.sum(labels[idx[j*batchsize:(j+1)*batchsize]] != onehot_decode(y, self.lab_vals))
                trRE += np.sum(self.error(d, y))


            trCE /= count
            trRE /= count
            Errors["trainCEs"].append(trCE)
            Errors["trainREs"].append(trRE)

            if comp_test:
                t_y, t_y_c = self.predict(test_inputs)
                teCE = np.sum(test_labels != t_y_c)
                teRE = np.sum(self.error(t_d, t_y))
                teCE /= t_count
                teRE /= t_count
                Errors["testCEs"].append(teCE)
                Errors["testREs"].append(teRE)

            if (ep+1) % 10 == 0 and verbose:
                print('Epoch {:3d}/{}, CE = {:6.2%}, RE = {:.5f}'.format(ep+1, eps, CE, RE))

        return Errors

    def getBoundry(self, mesh, thres=0.1, mid=0):
        boundry = []
        for m in mesh:
            #print(m.ndim)
            y = abs(self.predictOne(m)-mid)
            if y < thres:
                boundry.append(m)
        #print(boundry)
        return np.array(boundry)