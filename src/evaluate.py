import math


def topNRecall(ground_truth, test_result, topn):
    count = 0
    total = len(ground_truth)
    for paper in test_result[:topn]:
        if paper in ground_truth:
            count += 1
    return count / total


def topNPrecision(ground_truth, test_result, topn=None):
    count = 0
    if topn is None:
        topn = min(len(ground_truth), len(test_result))
    for paper in test_result[:topn]:
        if paper in ground_truth:
            count += 1
    return count / topn


def binaryNDCG(ground_truth, test_result, topn):
    dcg, pdcg = 0.0, 0.0
    for i, item in enumerate(test_result[:topn]):
        if item in ground_truth:
            dcg += 1 / math.log(i + 2)
    for i in range(min(len(ground_truth), topn)):
        pdcg += 1 / math.log(i + 2)
    return dcg / pdcg


def MAP(ground_truth, test_result):
    relevant_num, sums = 0, 0.0
    for i, paper in enumerate(test_result):
        if paper in ground_truth:
            relevant_num += 1
            sums += relevant_num / (i + 1)
    return sums / len(ground_truth)


def get_mean_measure(measure_func):
    def new_measure(ground_truths, test_results, *args, **kwargs):
        value, num = 0.0, 0
        for ground_truth, test_result in zip(ground_truths, test_results):
            if len(ground_truth) > 0:
                value += measure_func(ground_truth, test_result, *args, **kwargs)
                num += 1
        return value / num

    return new_measure


def get_list_measure(measure_func):
    def new_measure(ground_truths, test_results, *args, **kwargs):
        values = []
        for ground_truth, test_result in zip(ground_truths, test_results):
            if len(ground_truth) > 0:
                values.append(measure_func(ground_truth, test_result, *args, **kwargs))
        return values

    return new_measure


def multi_mean_measure(evaluation_data, measures):
    scores, num = [0.0 for _ in range(len(measures))], 0
    for ground_truth, test_result in evaluation_data:
        num += 1
        for i in range(len(measures)):
            scores[i] += measures[i](ground_truth, test_result)
    return list(map(lambda x: x / num, scores))


meanTopnPrecision = get_mean_measure(topNPrecision)
listTopnPrecision = get_list_measure(topNPrecision)
