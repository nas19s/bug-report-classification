import pandas as pd
import numpy as np
import re
import os
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import wilcoxon
from scipy.sparse import hstack, csr_matrix
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


# inline stopwords so no nltk download required
STOP_WORDS = set((
    "i me my myself we our ours ourselves you your yours yourself yourselves "
    "he him his himself she her hers herself it its itself they them their theirs "
    "themselves what which who whom this that these those am is are was were be "
    "been being have has had having do does did doing a an the and but if or "
    "because as until while of at by for with about against between into through "
    "during before after above below to from up down in out on off over under "
    "again further then once here there when where why how all both each few more "
    "most other some such no nor not only own same so than too very s t can will "
    "just don should now d ll m o re ve y ain aren couldn didn doesn hadn hasn "
    "haven isn ma mightn mustn needn shan shouldn wasn weren won wouldn"
).split())

# performance-related terms that tend to appear in perf bug reports
PERF_KEYWORDS = {
    'slow', 'fast', 'speed', 'latency', 'throughput', 'memory', 'cpu', 'gpu',
    'performance', 'efficient', 'inefficient', 'bottleneck', 'overhead', 'lag',
    'delay', 'timeout', 'oom', 'leak', 'benchmark', 'profile', 'inference',
    'training', 'runtime', 'compute', 'parallelism', 'batch', 'optimize',
    'fps', 'ms', 'seconds', 'minutes', 'hours', 'mb', 'gb', 'ram', 'vram'
}


def remove_html(text):
    return re.sub(r'<.*?>', '', str(text))


def remove_emoji(text):
    pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return pattern.sub('', text)


def clean_str(text):
    text = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", str(text))
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip().lower()


def remove_stopwords(text):
    return " ".join([w for w in str(text).split() if w not in STOP_WORDS])


def preprocess(text):
    text = remove_html(text)
    text = remove_emoji(text)
    text = remove_stopwords(text)
    text = clean_str(text)
    return text


def extract_handcrafted_features(texts):
    """
    5 handcrafted features per report:
    - performance keyword count
    - log of text length
    - digit character count (perf reports often have benchmark numbers)
    - average word length
    - unique word ratio
    """
    feats = []
    for text in texts:
        tokens = text.split()
        n = max(len(tokens), 1)
        kw   = sum(1 for t in tokens if t in PERF_KEYWORDS)
        tlen = np.log1p(len(text))
        digs = sum(c.isdigit() for c in text)
        awl  = np.mean([len(t) for t in tokens]) if tokens else 0
        uniq = len(set(tokens)) / n
        feats.append([kw, tlen, digs, awl, uniq])
    return np.array(feats, dtype=np.float32)


PROJECTS = ['pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe']
REPEAT = 30
RESULTS = {}

print("=" * 60)
print("Bug Report Classification")
print("=" * 60)

for project in PROJECTS:
    path = f'data/{project}.csv'
    if not os.path.exists(path):
        print(f"[SKIP] {project}.csv not found")
        continue

    print(f"\n--- {project} ---")

    df = pd.read_csv(path)
    df = df.sample(frac=1, random_state=999).reset_index(drop=True)

    df['text'] = df.apply(
        lambda r: r['Title'] + '. ' + r['Body'] if pd.notna(r['Body']) else r['Title'],
        axis=1
    )
    df['text']  = df['text'].apply(preprocess)
    df['label'] = df['class']

    texts  = df['text'].values
    labels = df['label'].values

    nb_res  = {k: [] for k in ['acc', 'prec', 'rec', 'f1', 'auc', 'time']}
    abl_res = {k: [] for k in ['acc', 'prec', 'rec', 'f1', 'auc', 'time']}
    xgb_res = {k: [] for k in ['acc', 'prec', 'rec', 'f1', 'auc', 'time']}

    for seed in range(REPEAT):
        idx = np.arange(len(texts))
        tr_idx, te_idx = train_test_split(
            idx, test_size=0.3, random_state=seed, stratify=labels
        )

        tr_text, te_text = texts[tr_idx], texts[te_idx]
        y_train, y_test  = labels[tr_idx], labels[te_idx]

        tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=3000, sublinear_tf=True)
        X_tr = tfidf.fit_transform(tr_text)
        X_te = tfidf.transform(te_text)

        neg = np.sum(y_train == 0)
        pos = np.sum(y_train == 1)
        spw = neg / max(pos, 1)

        # baseline - naive bayes
        t0 = time.time()
        nb = GaussianNB()
        nb.fit(X_tr.toarray(), y_train)
        nb_pred = nb.predict(X_te.toarray())
        t_nb = time.time() - t0

        fpr, tpr, _ = roc_curve(y_test, nb_pred, pos_label=1)
        nb_res['acc'].append(accuracy_score(y_test, nb_pred))
        nb_res['prec'].append(precision_score(y_test, nb_pred, average='macro', zero_division=0))
        nb_res['rec'].append(recall_score(y_test, nb_pred, average='macro', zero_division=0))
        nb_res['f1'].append(f1_score(y_test, nb_pred, average='macro', zero_division=0))
        nb_res['auc'].append(auc(fpr, tpr))
        nb_res['time'].append(t_nb)

        # ablation - xgboost with tfidf only, no handcrafted features
        t0 = time.time()
        clf_abl = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw, eval_metric='logloss',
            random_state=seed, verbosity=0
        )
        clf_abl.fit(X_tr, y_train)
        abl_pred = clf_abl.predict(X_te)
        t_abl = time.time() - t0

        fpr, tpr, _ = roc_curve(y_test, abl_pred, pos_label=1)
        abl_res['acc'].append(accuracy_score(y_test, abl_pred))
        abl_res['prec'].append(precision_score(y_test, abl_pred, average='macro', zero_division=0))
        abl_res['rec'].append(recall_score(y_test, abl_pred, average='macro', zero_division=0))
        abl_res['f1'].append(f1_score(y_test, abl_pred, average='macro', zero_division=0))
        abl_res['auc'].append(auc(fpr, tpr))
        abl_res['time'].append(t_abl)

        # proposed - xgboost + tfidf + handcrafted features
        t0 = time.time()
        tr_hand = extract_handcrafted_features(tr_text)
        te_hand = extract_handcrafted_features(te_text)

        scaler = MinMaxScaler()
        tr_hand = scaler.fit_transform(tr_hand)
        te_hand = scaler.transform(te_hand)

        X_tr_full = hstack([X_tr, csr_matrix(tr_hand)])
        X_te_full = hstack([X_te, csr_matrix(te_hand)])

        clf = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw, eval_metric='logloss',
            random_state=seed, verbosity=0
        )
        clf.fit(X_tr_full, y_train)
        xgb_pred = clf.predict(X_te_full)
        t_xgb = time.time() - t0

        fpr, tpr, _ = roc_curve(y_test, xgb_pred, pos_label=1)
        xgb_res['acc'].append(accuracy_score(y_test, xgb_pred))
        xgb_res['prec'].append(precision_score(y_test, xgb_pred, average='macro', zero_division=0))
        xgb_res['rec'].append(recall_score(y_test, xgb_pred, average='macro', zero_division=0))
        xgb_res['f1'].append(f1_score(y_test, xgb_pred, average='macro', zero_division=0))
        xgb_res['auc'].append(auc(fpr, tpr))
        xgb_res['time'].append(t_xgb)

    RESULTS[project] = {'nb': nb_res, 'abl': abl_res, 'xgb': xgb_res}

    for name, res in [('NB (Baseline)', nb_res),
                      ('XGB-TF-IDF (Ablation)', abl_res),
                      ('XGBoost (Proposed)', xgb_res)]:
        print(f"  {name}:")
        for k in ['acc', 'prec', 'rec', 'f1', 'auc']:
            print(f"    {k.upper():5s}: {np.mean(res[k]):.4f} +/- {np.std(res[k]):.4f}")


# wilcoxon tests
print("\n" + "=" * 60)
print("Wilcoxon: Proposed vs NB")
print("=" * 60)

stat_results = {}
for project, res in RESULTS.items():
    stat_results[project] = {}
    print(f"\n{project}:")
    for m in ['acc', 'prec', 'rec', 'f1', 'auc']:
        try:
            _, p = wilcoxon(res['xgb'][m], res['nb'][m])
        except Exception:
            p = float('nan')
        stat_results[project][m] = p
        print(f"  {m.upper():5s}: p={p:.4f} ({'significant' if p < 0.05 else 'not significant'})")

print("\n" + "=" * 60)
print("Wilcoxon: Proposed vs Ablation")
print("=" * 60)

for project, res in RESULTS.items():
    print(f"\n{project}:")
    for m in ['acc', 'prec', 'rec', 'f1', 'auc']:
        try:
            _, p = wilcoxon(res['xgb'][m], res['abl'][m])
        except Exception:
            p = float('nan')
        print(f"  {m.upper():5s}: p={p:.4f} ({'significant' if p < 0.05 else 'not significant'})")


# save to csv
os.makedirs('results', exist_ok=True)
projects_done = list(RESULTS.keys())

rows = []
for project in projects_done:
    res = RESULTS[project]
    for metric in ['acc', 'prec', 'rec', 'f1', 'auc', 'time']:
        for i in range(REPEAT):
            rows.append({
                'project': project,
                'repeat': i,
                'metric': metric,
                'NB': res['nb'][metric][i],
                'XGBoost_ablation': res['abl'][metric][i],
                'XGBoost_proposed': res['xgb'][metric][i]
            })

pd.DataFrame(rows).to_csv('results/raw_results.csv', index=False)

summary = []
for project in projects_done:
    res = RESULTS[project]
    for key, name in [('nb', 'NB'), ('abl', 'XGB-Ablation'), ('xgb', 'XGBoost')]:
        row = {'Project': project, 'Model': name}
        for m in ['acc', 'prec', 'rec', 'f1', 'auc']:
            vals = res[key][m]
            row[m.upper()] = f"{np.mean(vals):.4f}+/-{np.std(vals):.4f}"
        summary.append(row)

pd.DataFrame(summary).to_csv('results/summary_table.csv', index=False)
print("\nResults saved to results/")


# plots
os.makedirs('figures', exist_ok=True)
sns.set_style("whitegrid")

c = {'nb': '#4878CF', 'abl': '#F5A623', 'xgb': '#D65F5F'}
metrics = ['acc', 'prec', 'rec', 'f1', 'auc']
mlabels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']

# fig 1 - metrics bar chart
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
for ax, metric, mlabel in zip(axes, metrics, mlabels):
    x = np.arange(len(projects_done))
    w = 0.25
    for i, (key, name) in enumerate([('nb', 'NB'), ('abl', 'XGB-TF-IDF'), ('xgb', 'XGBoost+Feats')]):
        means = [np.mean(RESULTS[p][key][metric]) for p in projects_done]
        stds  = [np.std(RESULTS[p][key][metric])  for p in projects_done]
        ax.bar(x + (i - 1) * w, means, w, yerr=stds, color=c[key],
               capsize=2, error_kw={'linewidth': 0.7})
    ax.set_title(mlabel, fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([p[:5] for p in projects_done], fontsize=7, rotation=20)
    ax.set_ylim(0, 1.05)
    if metric == 'acc':
        ax.set_ylabel('Score')

handles = [
    mpatches.Patch(color=c['nb'],  label='NB (Baseline)'),
    mpatches.Patch(color=c['abl'], label='XGB-TF-IDF (Ablation)'),
    mpatches.Patch(color=c['xgb'], label='XGBoost+Features (Proposed)'),
]
fig.legend(handles=handles, loc='upper center', ncol=3, fontsize=8, bbox_to_anchor=(0.5, 1.04))
plt.tight_layout()
plt.savefig('figures/fig1_metric_comparison.png', bbox_inches='tight', dpi=150)
plt.close()

# fig 2 - f1 boxplots
fig, axes = plt.subplots(1, len(projects_done), figsize=(4 * len(projects_done), 4), sharey=True)
if len(projects_done) == 1:
    axes = [axes]

for ax, project in zip(axes, projects_done):
    data = [
        RESULTS[project]['nb']['f1'],
        RESULTS[project]['abl']['f1'],
        RESULTS[project]['xgb']['f1']
    ]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops={'color': 'black', 'linewidth': 2})
    for patch, col in zip(bp['boxes'], [c['nb'], c['abl'], c['xgb']]):
        patch.set_facecolor(col)
    ax.set_title(project[:10], fontsize=9, fontweight='bold')
    ax.set_xticklabels(['NB', 'XGB-TF-IDF', 'XGB+Feats'], fontsize=7, rotation=10)
    ax.set_ylim(0, 1.0)
    if project == projects_done[0]:
        ax.set_ylabel('F1 Score (Macro)')
    p_val = stat_results.get(project, {}).get('f1', float('nan'))
    ax.text(0.5, 0.02, f"p={p_val:.4f}", ha='center',
            transform=ax.transAxes, fontsize=7, color='grey')

plt.suptitle('F1 Score Distribution Across 30 Repeats', fontsize=10, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/fig2_f1_boxplot.png', bbox_inches='tight', dpi=150)
plt.close()

# fig 3 - training time
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(projects_done))
w = 0.25
for i, (key, name) in enumerate([('nb', 'NB'), ('abl', 'XGB-TF-IDF'), ('xgb', 'XGBoost+Feats')]):
    means = [np.mean(RESULTS[p][key]['time']) for p in projects_done]
    ax.bar(x + (i - 1) * w, means, w, label=name, color=c[key])

ax.set_xticks(x)
ax.set_xticklabels(projects_done, rotation=15, fontsize=8)
ax.set_ylabel('Avg Training Time (s)')
ax.set_title('Training Time Comparison', fontweight='bold')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig('figures/fig3_training_time.png', bbox_inches='tight', dpi=150)
plt.close()

print("Figures saved to figures/")
print("Done.")