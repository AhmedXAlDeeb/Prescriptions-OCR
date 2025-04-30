import jiwer


def compute_cer(predictions, labels):
    total_cer = 0
    for pred, label in zip(predictions, labels):
        pred = pred.lower()
        label = label.lower()

        distance = jiwer.compute_measures(label, pred)['substitutions'] + \
                  jiwer.compute_measures(label, pred)['deletions'] + \
                  jiwer.compute_measures(label, pred)['insertions']

        total_chars = len(label)
        if total_chars > 0:
            total_cer += distance / total_chars

    return total_cer / len(predictions)

def compute_wer(predictions, labels):
    return jiwer.wer(labels, predictions)