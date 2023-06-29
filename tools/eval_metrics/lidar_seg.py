import numpy as np

def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    bin_count = np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2)
    return bin_count[:n ** 2].reshape(n, n)
def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def compute_iou(seg_result,n=17):
    hist_list = []
    for seg_i in seg_result:
        pred = seg_i['lidar_pred']
        label = seg_i['lidar_label']
        assert pred.shape[0]==label.shape[0]
        hist = fast_hist(pred, label, n)
        hist_list.append(hist)
    iou = per_class_iu(sum(hist_list))
    return iou