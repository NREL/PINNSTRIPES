import sys

import numpy as np
import tensorflow as tf


def completeDataset(xDataList, x_params_dataList, yDataList):

    nList = [xData.shape[0] for xData in xDataList]
    maxN = max(nList)
    toFillList = [max(maxN - n, 0) for n in nList]

    for i, toFill in enumerate(toFillList):
        if toFill < nList[i]:
            xDataList[i] = np.vstack((xDataList[i], xDataList[i][:toFill, :]))
            x_params_dataList[i] = np.vstack(
                (x_params_dataList[i], x_params_dataList[i][:toFill, :])
            )
            yDataList[i] = np.vstack((yDataList[i], yDataList[i][:toFill, :]))
        if toFill > nList[i]:
            nrep = toFill // nList[i]
            nrep_res = toFill - nList[i] * nrep
            xData_tmp = xDataList[i].copy()
            x_params_data_tmp = x_params_dataList[i].copy()
            yData_tmp = yDataList[i].copy()
            for _ in range(nrep):
                xDataList[i] = np.vstack((xDataList[i], xData_tmp))
                x_params_dataList[i] = np.vstack(
                    (x_params_dataList[i], x_params_data_tmp)
                )
                yDataList[i] = np.vstack((yDataList[i], yData_tmp))
            xDataList[i] = np.vstack((xDataList[i], xData_tmp[:nrep_res, :]))
            x_params_dataList[i] = np.vstack(
                (x_params_dataList[i], x_params_data_tmp[:nrep_res, :])
            )
            yDataList[i] = np.vstack((yDataList[i], yData_tmp[:nrep_res, :]))
    return maxN


def checkDataShape(xData, x_params_data, yData):
    if not len(xData.shape) == 2:
        print("Expected tensor of rank 2 for xData")
        print("xData shape =", xData.shape)
        sys.exit()
    if not len(x_params_data.shape) == 2:
        print("Expected tensor of rank 2 for x_params_data")
        print("x_params_data shape =", x_params_data.shape)
        sys.exit()
    if not len(yData.shape) == 2:
        print("Expected tensor of rank 2 for yData")
        print("yData shape =", yData.shape)
        sys.exit()
    # if not (xData.shape[1]==self.ndim):
    #     print('Expected xData.shape[1] =',ndim)
    #     print('xData shape =',xData.shape)
    #     sys.exit()


def check_loss_component_dim(terms, string):
    for i, term in enumerate(terms):
        if i == 0:
            refShape = tf.shape(term)
            print("Shape %s " % string, refShape)
        shape = tf.shape(term)
        if (
            not bool(tf.shape(shape) == 2)
            or not bool(shape[1] == 1)
            or not bool((shape == refShape).numpy().all())
        ):
            print("Shape of term %d in %s unexpected" % (i, string))
            print("shape = ", shape)
            sys.exit()


def check_loss_dim(self, intTerms, boundTerms, dataTerms, regTerms):
    if self.activeInt:
        check_loss_component_dim(intTerms, "intLoss")

    if self.activeBound:
        check_loss_component_dim(boundTerms, "boundLoss")

    if self.activeData:
        check_loss_component_dim(dataTerms, "dataLoss")

    if self.activeReg:
        check_loss_component_dim(regTerms, "regLoss")
