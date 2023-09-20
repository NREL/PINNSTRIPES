# pulled from https://github.com/yaroslavvb/stuff/blob/master/eager_lbfgs/eager_lbfgs.py

import os
import sys
import time

import numpy as np
import tensorflow as tf


def dot(a, b):
    """Dot product function since TensorFlow doesn't have one."""
    return tf.reduce_sum(a * b)


def verbose_func(s):
    print(s)


final_loss = None
times = []


def safe_save(model, weight_path, overwrite=False):
    saved = False
    ntry = 0
    while not saved:
        try:
            model.save_weights(weight_path, overwrite=overwrite)
            saved = True
        except BlockingIOError:
            ntry += 1
        if ntry > 10000:
            sys.exit(f"ERROR: could not save {weight_path}")


def logLosses(
    logLossFolder, epoch, interiorTerms, boundaryTerms, dataTerms, regTerms
):
    interiorTermsArray = [
        np.mean(tf.square(interiorTerm)) for interiorTerm in interiorTerms
    ]
    interiorTermsArrayPercent = [
        round(term / (1e-16 + sum(interiorTermsArray)), 2)
        for term in interiorTermsArray
    ]
    f = open(os.path.join(logLossFolder, "interiorTerms.csv"), "a+")
    f.write(str(int(epoch)) + ";" + str(interiorTermsArrayPercent) + "\n")
    f.write(str(int(epoch)) + ";" + str(interiorTermsArray) + "\n")
    f.close()
    boundaryTermsArray = [
        np.mean(tf.square(boundaryTerm)) for boundaryTerm in boundaryTerms
    ]
    boundaryTermsArrayPercent = [
        round(term / (1e-16 + sum(boundaryTermsArray)), 2)
        for term in boundaryTermsArray
    ]
    f = open(os.path.join(logLossFolder, "boundaryTerms.csv"), "a+")
    f.write(str(int(epoch)) + ";" + str(boundaryTermsArrayPercent) + "\n")
    f.write(str(int(epoch)) + ";" + str(boundaryTermsArray) + "\n")
    f.close()
    dataTermsArray = [np.mean(tf.square(dataTerm)) for dataTerm in dataTerms]
    dataTermsArrayPercent = [
        round(term / (1e-16 + sum(dataTermsArray)), 2)
        for term in dataTermsArray
    ]
    f = open(os.path.join(logLossFolder, "dataTerms.csv"), "a+")
    f.write(str(int(epoch)) + ";" + str(dataTermsArrayPercent) + "\n")
    f.write(str(int(epoch)) + ";" + str(dataTermsArray) + "\n")
    regTermsArray = [np.mean(tf.square(regTerm)) for regTerm in regTerms]
    regTermsArrayPercent = [
        round(term / (1e-16 + sum(regTermsArray)), 2) for term in regTermsArray
    ]
    f = open(os.path.join(logLossFolder, "regTerms.csv"), "a+")
    f.write(str(int(epoch)) + ";" + str(regTermsArrayPercent) + "\n")
    f.write(str(int(epoch)) + ";" + str(regTermsArray) + "\n")


def lbfgs(
    opfunc,
    x,
    state,
    model,
    bestLoss,
    modelFolder,
    maxIter=100,
    learningRate=1,
    do_verbose=True,
    dynamicAttention=False,
    logLossFolder="Log",
    nEpochDoneLBFGS=0,
    nEpochDoneSGD=0,
    nBatchSGD=0,
    nEpochs_start_lbfgs=0,
):
    """port of lbfgs.lua, using TensorFlow eager mode."""

    global final_loss, times

    maxEval = maxIter * 1.25
    tolFun = 1e-5
    tolX = 1e-15
    nCorrection = max(nEpochs_start_lbfgs, 50)
    isverbose = False
    target_lr = learningRate

    # verbose function
    if isverbose:
        verbose = verbose_func
    else:
        verbose = lambda x: None

    if dynamicAttention:
        (
            f,
            g,
            f_unweighted,
            intL,
            boundL,
            dataL,
            regL,
            int_loss,
            bound_loss,
            data_loss,
            reg_loss,
        ) = opfunc(x)
    else:
        (
            f,
            g,
            intL,
            boundL,
            dataL,
            regL,
            int_loss,
            bound_loss,
            data_loss,
            reg_loss,
        ) = opfunc(x)

    f_hist = [f]
    currentFuncEval = 1
    state.funcEval = state.funcEval + 1
    p = g.shape[0]

    # check optimality of initial point
    tmp1 = tf.abs(g)
    if tf.reduce_sum(tmp1) <= tolFun:
        verbose("optimality condition below tolFun")
        return x, f_hist, currentFuncEval, bestLoss

    # optimize for a max of maxIter iterations
    nIter = 0
    times = []
    window_ave_time = 0

    counter_success = 0
    counter_failure = 0

    while nIter < maxIter:
        start_time = time.time()

        # keep track of nb of iterations
        nIter = nIter + 1
        state.nIter = state.nIter + 1

        ############################################################
        ## compute gradient descent direction
        ############################################################
        if state.nIter == 1:
            d = -g
            old_dirs = []
            old_stps = []
            Hdiag = 1
        else:
            # do lbfgs update (update memory)
            y = g - g_old
            s = d * t
            ys = dot(y, s)

            if ys > 1e-10:
                # updating memory
                if len(old_dirs) == nCorrection:
                    # shift history by one (limited-memory)
                    del old_dirs[0]
                    del old_stps[0]

                # store new direction/step
                old_dirs.append(s)
                old_stps.append(y)

                # update scale of initial Hessian approximation
                Hdiag = ys / dot(y, y)

            # compute the approximate (L-BFGS) inverse Hessian
            # multiplied by the gradient
            k = len(old_dirs)

            # need to be accessed element-by-element, so don't re-type tensor:
            ro = [0] * nCorrection
            for i in range(k):
                ro[i] = 1 / dot(old_stps[i], old_dirs[i])

            # iteration in L-BFGS loop collapsed to use just one buffer
            # need to be accessed element-by-element, so don't re-type tensor:
            al = [0] * nCorrection

            q = -g
            for i in range(k - 1, -1, -1):
                al[i] = dot(old_dirs[i], q) * ro[i]
                q = q - al[i] * old_stps[i]

            # multiply by initial Hessian
            r = q * Hdiag
            for i in range(k):
                be_i = dot(old_stps[i], r) * ro[i]
                r += (al[i] - be_i) * old_dirs[i]

            d = r
            # final direction is in r/d (same object)

        g_old = g
        f_old = f
        x_old = x

        ############################################################
        ## compute step length
        ############################################################
        # directional derivative
        gtd = dot(g, d)

        # check that progress can be made along that direction
        if gtd > -tolX:
            verbose("Can not make progress along direction.")
            break

        # reset initial guess for step size
        if state.nIter == 1:
            tmp1 = tf.abs(g)
            t = min(learningRate, 1 / tf.reduce_sum(tmp1))
        else:
            t = learningRate
        # reduce learning rate for a few iterations before we are confident in the Hessian
        if nIter <= nEpochs_start_lbfgs:
            t /= max(learningRate / 1e-12, 100)

        x += t * d

        if nIter != maxIter:
            # re-evaluate function only if not in last iteration
            # the reason we do this: in a stochastic setting,
            # no use to re-evaluate that function here
            if dynamicAttention:
                (
                    f,
                    g,
                    f_unweighted,
                    intL,
                    boundL,
                    dataL,
                    regL,
                    int_loss,
                    bound_loss,
                    data_loss,
                    reg_loss,
                ) = opfunc(x)
            else:
                (
                    f,
                    g,
                    intL,
                    boundL,
                    dataL,
                    regL,
                    int_loss,
                    bound_loss,
                    data_loss,
                    reg_loss,
                ) = opfunc(x)
        # ~~~ Dynamic learning rate

        # Check if failure of iteration

        if nIter > nEpochs_start_lbfgs:
            if f.numpy() > f_hist[-1] * 2 or np.isnan(f.numpy()):
                counter_failure += 1
                counter_success = 0
                f = f_old
                g = g_old
                x = x_old
            else:
                counter_success += 1
                counter_failure = 0

            # Loss increase
            if counter_failure >= 1:
                # Decrease learning rate
                if learningRate > 1e-6:
                    learningRate /= 2
                else:
                    learningRate /= 1.1
                if state.nIter == 1:
                    tmp1 = tf.abs(g)
                    t = min(learningRate, 1 / tf.reduce_sum(tmp1))
                else:
                    t = learningRate
                if nIter < 10:
                    t /= max(learningRate / 1e-12, 100)
                x = x_old
                counter_failure = 0
            if counter_success >= 10:
                if learningRate < 1e-6:
                    learningRate = min(target_lr, learningRate * 4)
                elif learningRate < 1e-4:
                    learningRate = min(target_lr, learningRate * 2)
                else:
                    learningRate = min(target_lr, learningRate * 1.1)
                counter_success = 0

        lsFuncEval = 1
        f_hist.append(f)

        # update func eval
        currentFuncEval = currentFuncEval + lsFuncEval
        state.funcEval = state.funcEval + lsFuncEval

        ############################################################
        ## check conditions
        ############################################################
        if nIter == maxIter:
            break

        if currentFuncEval >= maxEval:
            # max nb of function evals
            print("max nb of function evals")
            break

        tmp1 = tf.abs(g)
        if tf.reduce_sum(tmp1) <= tolFun:
            # check optimality
            print("optimality condition below tolFun")
            break

        tmp1 = tf.abs(d * t)
        if nIter > nEpochs_start_lbfgs and tf.reduce_sum(tmp1) <= tolX:
            # step size below tolX
            print("step size below tolX")
            break
        if learningRate < 1e-15:
            # step size below tolX
            print("training appears stuck")
            break

        # if tf.abs(f, f_old) < tolX:
        #     # function value changing less than tolX
        #     print(
        #         "function value changing less than tolX"
        #         + str(tf.abs(f - f_old))
        #     )
        #     break

        if do_verbose:
            # Log the loss
            logFile = open(os.path.join(logLossFolder, "log.csv"), "a+")
            if dynamicAttention:
                logFile.write(
                    str(int(nEpochDoneLBFGS + nEpochDoneSGD + nIter))
                    + ";"
                    + str(
                        int(
                            nEpochDoneLBFGS + nEpochDoneSGD * nBatchSGD + nIter
                        )
                    )
                    + ";"
                    + str(f.numpy())
                    + ";"
                    + str(f_unweighted.numpy())
                    + "\n"
                )
            else:
                logFile.write(
                    str(int(nEpochDoneLBFGS + nEpochDoneSGD + nIter))
                    + ";"
                    + str(
                        int(
                            nEpochDoneLBFGS + nEpochDoneSGD * nBatchSGD + nIter
                        )
                    )
                    + ";"
                    + str(f.numpy())
                    + "\n"
                )
            logFile.close()
            logLosses(
                logLossFolder,
                (nEpochDoneSGD * nBatchSGD) + nEpochDoneLBFGS + int(nIter),
                intL,
                boundL,
                dataL,
                regL,
            )

            # Save the weights
            currentLoss = f.numpy()
            if bestLoss is None or currentLoss < bestLoss:
                bestLoss = currentLoss
                safe_save(
                    model, os.path.join(modelFolder, "best.h5"), overwrite=True
                )
            if nIter % 10 == 0:
                safe_save(
                    model,
                    os.path.join(modelFolder, "lastLBFGS.h5"),
                    overwrite=True,
                )
            if (
                nIter + nEpochDoneLBFGS + nEpochDoneSGD * nBatchSGD
            ) % 1000 == 0:
                totalStep = nIter + nEpochDoneLBFGS + nEpochDoneSGD * nBatchSGD
                safe_save(
                    model,
                    os.path.join(modelFolder, f"step_{totalStep}.h5"),
                    overwrite=True,
                )
                print("\nSaved weights")

            end_time = time.time()
            window_ave_time += end_time - start_time

            # output to screen
            if nIter % 1 == 0:
                print(
                    "Step %3d loss %6.5f lr %.4g iL %.2f bL %.2f dL %.2f rL %.2f t/step %.2f ms "
                    % (
                        nIter,
                        f.numpy(),
                        learningRate,
                        int_loss.numpy() / (f.numpy() + 1e-12),
                        bound_loss.numpy() / (f.numpy() + 1e-12),
                        data_loss.numpy() / (f.numpy() + 1e-12),
                        reg_loss.numpy() / (f.numpy() + 1e-12),
                        1000 * (window_ave_time) / 1,
                    )
                )
                sys.stdout.flush()
                window_ave_time = 0

        if nIter == maxIter - 1:
            final_loss = f.numpy()
            safe_save(
                model,
                os.path.join(modelFolder, f"lastLBFGS.h5"),
                overwrite=True,
            )

    # save state
    state.old_dirs = old_dirs
    state.old_stps = old_stps
    state.Hdiag = Hdiag
    state.g_old = g_old
    state.f_old = f_old
    state.t = t
    state.d = d

    return x, f_hist, currentFuncEval, bestLoss


# dummy/Struct gives Lua-like struct object with 0 defaults
class dummy(object):
    pass


class Struct(dummy):
    def __getattribute__(self, key):
        if key == "__dict__":
            return super(dummy, self).__getattribute__("__dict__")
        return self.__dict__.get(key, 0)
