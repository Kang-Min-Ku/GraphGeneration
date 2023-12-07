import os
import logging
import torch
from eval import eval_acc

def test(params, data):
    logging.info('Loading existing model from %s' % os.path.join(params.exp_dir, 'best_graph_classifier.pth'))
    graph_classifier = torch.load(os.path.join(params.exp_dir, 'best_graph_classifier.pth')).to(device=params.device)
    graph_classifier.eval()
    with torch.no_grad():
        score = graph_classifier(data)
        test_acc = eval_acc(y_true = data.y[data.test_mask], y_pred = score[data.test_mask])
        logging.info(f'Test Set Performance: {test_acc}')
        