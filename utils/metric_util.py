from utils.evaluate import compute_metrics


def write_result(output_golden_file, golden, predict,label_name):
    with open(output_golden_file, encoding='utf-8', mode='a') as fw:
        for i in range(len(golden)):
            fw.write(str(golden[i]) + "\t" + str(predict[i]) + "\n")
        _result, result = compute_metrics("report", golden, predict, label_name=label_name)
        for key in sorted(result.keys()):
            fw.write("%s = %s\n" % (key, str(result[key])))
        fw.write("\n")