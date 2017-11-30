import numpy

recall = lambda tp, fn: tp/(tp + fn)
precision = lambda tp, fp: tp/(tp + fp)
f_mean = lambda p, r: 2*p*r/(p+r)

def calc_fp(pred, true):
    p1 = numpy.where(pred == 1)
    t0 = numpy.where(true == 0)
    
    fp = set(p1[0].tolist()) & set(t0[0].tolist())
    return len(fp)

def calc_fn(pred, true):
    p0 = numpy.where(pred == 0)
    t1 = numpy.where(true == 1)
    
    fn = set(p0[0].tolist()) & set(t1[0].tolist())
    return len(fn)

def calc_tp(pred, true):
    p1 = numpy.where(pred == 1)
    t1 = numpy.where(true == 1)
    
    tp = set(p1[0].tolist()) & set(t1[0].tolist())
    return len(tp)