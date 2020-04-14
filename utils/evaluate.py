import logging

logger = logging.getLogger(__name__)

try:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import matthews_corrcoef, f1_score, classification_report
    _has_sklearn = True
except (AttributeError, ImportError) as e:
    logger.warning("To use data.metrics please install scikit-learn. See https://scikit-learn.org/stable/index.html")
    _has_sklearn = False

def is_sklearn_available():
    return _has_sklearn

if _has_sklearn:

    def simple_accuracy(preds, labels):
        return (preds == labels).mean()


    def acc_and_f1(preds, labels):
        acc = simple_accuracy(preds, labels)
        f1 = f1_score(y_true=labels, y_pred=preds)
        return {
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
        }


    def pearson_and_spearman(preds, labels):
        pearson_corr = pearsonr(preds, labels)[0]
        spearman_corr = spearmanr(preds, labels)[0]
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
        }


    def compute_metrics(method_name, preds, labels,label_name=[]):
        assert len(preds) == len(labels)
        if method_name == "matthews_corrcoef":
            return {"mcc": matthews_corrcoef(labels, preds)}
        elif method_name == "acc":
            return {"acc": simple_accuracy(preds, labels)}
        elif method_name == "acc_and_f1":
            return acc_and_f1(preds, labels)
        elif method_name == "report":
            return classification_report(preds, labels, target_names=label_name,digits=4), classification_report(preds, labels, target_names=label_name,output_dict=True,digits=4)
        else:
            raise KeyError(method_name)