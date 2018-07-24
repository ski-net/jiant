import re
import sys
import datetime

# Pro tip: To copy the results of this script from the terminal in Mac OS, use command-alt-shift-c. That'll copy the tabs as tabs, not spaces.

# pro tip 2: generously use wildcards and do the collating in spreadsheet
# e.g.
#     python jiant/major_experiment_scripts/extract_edge_results.py /nfs/jsalt/exp/patrick-*/elmo-*/run*/log.log

if len(sys.argv) < 2:
  print("Usage: python extract_results.py log.log")
  exit(0)

col_order = ['date', 'train_tasks', 'dropout', 'elmo',
             'edges-dpr_mcc', 'edges-dpr_acc', 'edges-dpr_precision', 'edges-dpr_recall', 'edges-dpr_f1',
             'edges-spr2_mcc', 'edges-spr2_acc', 'edges-spr2_precision', 'edges-spr2_recall', 'edges-spr2_f1',
             'edges-srl-conll2005_mcc', 'edges-srl-conll2005_acc', 'edges-srl-conll2005_precision', 'edges-srl-conll2005_recall', 'edges-srl-conll2005_f1',
             'edges-coref-ontonotes_mcc', 'edges-coref-ontonotes_acc', 'edges-coref-ontonotes_precision', 'edges-coref-ontonotes_recall', 'edges-coref-ontonotes_f1',
             'edges-dep-labeling_mcc', 'edges-dep-labeling_acc', 'edges-dep-labeling_precision', 'edges-dep-labeling_recall', 'edges-dep-labeling_f1',
             'edges-ner-conll2003_mcc', 'edges-ner-conll2003_acc', 'edges-ner-conll2003_precision', 'edges-ner-conll2003_recall', 'edges-ner-conll2003_f1',
             'edges-constituent-ptb_mcc', 'edges-constituent-ptb_acc', 'edges-constituent-ptb_precision', 'edges-constituent-ptb_recall', 'edges-constituent-ptb_f1',
             'path']

today = datetime.datetime.now()

# looking at all lines is overkill, but just in case we change the format later, 
# or if there is more junk after the eval line

for path in sys.argv[1:]:
  try:
    cols = {c : '' for c in col_order}
    cols['date'] =  today.strftime("%m/%d/%Y")
    cols['path'] = path
    results_line = None
    found_eval = False
    train_tasks = None
    dropout = None
    elmo = None

    with open(path) as f:
      for line in f:
        line = line.strip()

        if line == 'Evaluating...':
          found_eval = True
        else:
          if found_eval:
            # safe number to prune out lines we don't care about. we usually have at least 10 fields in those lines
            if len(line.strip().split()) > 10:
              if results_line is not None:
                pass
                # print("WARNING: Multiple GLUE evals found. Skipping all but last.")
              results_line = line.strip()
              found_eval = False

        train_m = re.match('Training model on tasks: (.*)', line)
        if train_m:
          found_tasks = train_m.groups()[0]
          if train_tasks is not None and found_tasks != train_tasks:
            print("WARNING: Multiple sets of training tasks found. Skipping %s and reporting last."%(found_tasks))
          train_tasks = found_tasks

        do_m = re.match('"dropout": (.*),', line)
        if do_m:
          do = do_m.groups()[0]
          if dropout is None:
            # This is a bit of a hack: Take the first instance of dropout, which will come from the overall config.
            # Later matches will appear for model-specific configs.
            dropout = do

        el_m = re.match('"elmo_chars_only": (.*),', line)
        if el_m:
          el = el_m.groups()[0]
          if elmo is not None:
            assert (elmo == el), "Multiple elmo flags set, but settings don't match: %s vs. %s."%(elmo, el)
          elmo = el

    if train_tasks is None:
      train_tasks = ''
    cols['train_tasks'] = train_tasks
    cols['dropout'] = dropout
    cols['elmo'] = 'Y' if elmo == '0' else 'N'

    assert results_line is not None, "No GLUE eval results line found. Still training?"
    for mv in results_line.strip().split(','):
      metric, value = mv.split(':')
      cols[metric.strip()] = '%.02f'%(100*float(value.strip()))
    print(','.join([str(cols[c]) for c in col_order]))
  except BaseException as e:
   print("Error:", e, path)

