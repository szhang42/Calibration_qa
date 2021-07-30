"""
Temporary heatmap code.

"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


diffs = np.array( [     [36.66, 35.60, 26.77, 14.97, 3.24],
                        [33.64, 36.45, 26.23, 14.85, 2.83],
                        [33.05, 31.83, 28.68, 14.51, 4.69],
                        [29.36, 29.77, 24.50, 15.65, 3.79],
                        [22.23, 21.02, 20.33,  9.75, 9.93]], dtype=np.float64 )

datasets = ['TriviaQA', 'SearchQA', 'NewsQA', 'NQ', 'HotpotQA']

ordering = [0, 1, 2, 3, 4]

rearranged_diffs = np.zeros((5, 5))

for i, o1 in enumerate(ordering):
    for j, o2 in enumerate(ordering):
        rearranged_diffs[i][j] = diffs[o1, o2]

rearranged_datasets = [datasets[i] for i in ordering]

fig, ax = plt.subplots()
im = ax.imshow(diffs)

# We want to show all ticks...
ax.set_xticks(np.arange(len(datasets)))
ax.set_yticks(np.arange(len(datasets)))
# ... and label them with the respective list entries
ax.set_xticklabels(datasets)
ax.set_yticklabels(datasets)
ax.set_xlabel('Test Dataset (+SQuAD)')
ax.set_ylabel('Train Dataset (+SQuAD)')

# Loop over data dimensions and create text annotations.
for i in range(len(rearranged_datasets)):
    for j in range(len(rearranged_datasets)):
        if rearranged_diffs[i, j] > 30:
            text = ax.text(j, i, rearranged_diffs[i, j], size='large',
                        ha='center', va='center', color='k')
        else:
            text = ax.text(j, i, rearranged_diffs[i, j], size='large',
                        ha='center', va='center', color='w')

ax.set_title("Percentage reduction towards Best Possible AUC")
fig.tight_layout()
plt.savefig('heatmap.png', dpi=400)

